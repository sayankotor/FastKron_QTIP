import itertools
import math
import os
from functools import cache

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from lib.codebook import kdict
from lib.utils.kernel_check import has_kernel
from lib.utils.kernel_decompress import decode_compressed
from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda

# Opt-in CUDA-graph capture of the bs=1 eval decode path (steady-state generation).
# Each QuantizedLinear captures its own graph on first bs=1 forward; weights/SU/SV/
# hadK/trellis are constant per instance, so the graph is valid for any input vector
# (proven: probe_cudagraph.py -> rel_err 0, 10.3x/chain). Off by default so the
# quantize / prefill / bs>1 paths are byte-for-byte unchanged.
_CUDAGRAPH = os.environ.get("QTIP_CUDAGRAPH", "0") == "1"

# Global gate that must be True for any BitshiftLinear to enter the graph path.
# Off during prefill (and quantize/bs>1) -- a routed MoE expert can legitimately
# receive exactly 1 token during PREFILL, so numel()==n alone is NOT enough to
# distinguish steady-state decode. If capture were begun mid-prefill it would try
# to capture the surrounding forward pass's still-pending async GPU work and
# DEADLOCK. The generation driver turns this on only around the decode loop.
_CAPTURE_ENABLED = False


_GRAPH_POOL = None


def _graph_pool():
    """One shared CUDA-graph memory pool for ALL per-instance decode graphs.
    ~900 QuantizedLinear graphs (8 experts x2 + shared, x48 layers) each holding a
    private pool would waste allocator reservations; a shared pool is safe here
    because every graph's static output is clone()'d right after replay (never read
    across another graph's replay)."""
    global _GRAPH_POOL
    if _GRAPH_POOL is None:
        _GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _GRAPH_POOL


def set_graph_capture(enabled: bool):
    """Enable/disable CUDA-graph capture+replay for the bs=1 eval decode path.
    Must be True only when the model is doing genuine single-token decode (not
    prefill / not bs>1), otherwise capture can deadlock. No-op when QTIP_CUDAGRAPH
    is off, so the eager path stays byte-for-byte unchanged."""
    global _CAPTURE_ENABLED
    _CAPTURE_ENABLED = bool(enabled)


def decode_1mad(x):
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * 34038481 + 76625530
    x = x & ((1 << 32) - 1)
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    y = y - 510
    y = y.to(torch.float32)
    y = y / 147.800537109375
    return y


def decode_2mad(x):
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * 264435761 + 1013904223
    x = x & ((1 << 32) - 1)
    x = ((x * 1664525) >> 32) + x
    x = x & ((1 << 32) - 1)
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    y = y - 510
    y = y.to(torch.float32)
    y = y / 147.800537109375
    return y


def decode_3inst(x):

    def bfe16_to_fp16(x):
        x[torch.where(x >= 2**15)] -= 2**16
        return torch.tensor(x.to(torch.int16).numpy().view(np.float16))

    a = 89226354
    b = 64248484
    fpmask = 996162400
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * a + b
    mask = (1 << 15) + ((1 << 12) - 1)
    mask = (mask << 16) + mask
    res = (mask & x) ^ fpmask
    top = bfe16_to_fp16(res >> 16)
    bottom = bfe16_to_fp16(res & ((1 << 16) - 1))
    return (top + bottom).float()


def quantlut(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L)
        lut = (lut + 1) * lut
        lut = (lut >> (16 - nbits)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    return lut


def quantlut_sym(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp = 1 - ((lut >> 15) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    lut[:, 0] = lut[:, 0] * sflp
    return lut


class bitshift_codebook(nn.Module):

    def __init__(self,
                 L=16,
                 K=2,
                 V=2,
                 tlut_bits=16,
                 decode_mode='lut',
                 tlut=None):
        super(bitshift_codebook, self).__init__()
        self.idx_dtype = torch.int32
        self.opt_scale = 1

        self.L = L
        self.K = K
        self.V = V
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode

        if decode_mode == 'lut':
            if tlut is None:
                if tlut_bits > 0:
                    assert tlut_bits == L
                self.register_buffer('tlut', torch.randn(2**L, V))
                self.register_buffer('lut', self.tlut.T.contiguous())
            else:
                self.tlut = tlut
                self.recons_lut()

        elif decode_mode == '1mad':
            assert V == 1
            self.register_buffer('lut',
                                 decode_1mad(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == '2mad':
            assert V == 1
            self.register_buffer('lut',
                                 decode_2mad(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == '3inst':
            assert V == 1
            self.register_buffer('lut',
                                 decode_3inst(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == 'quantlut':
            if tlut is None:
                assert tlut_bits > 0
                if V == 1:
                    tlut = torch.erfinv((torch.arange(1 << tlut_bits) + 0.5) /
                                        (1 << tlut_bits) * 2 -
                                        1) * torch.tensor(2.0).sqrt()
                elif V == 2:
                    n = 2**tlut_bits
                    tlut = torch.zeros(n)
                    R = ((n / (n - torch.arange(n))).log() * 2).sqrt()
                    tlut = torch.stack(
                        [R * torch.arange(n).sin(), R * torch.arange(n).cos()],
                        dim=-1)
                else:
                    raise Exception
                self.register_buffer('tlut', tlut.unsqueeze(-1))
                self.register_buffer(
                    'lut',
                    quantlut(self.tlut, L, tlut_bits).T.contiguous())
            else:
                self.tlut = tlut
                self.recons_lut()
        elif decode_mode == 'quantlut_sym':
            if tlut is None:
                assert tlut_bits > 0
                if V == 2:
                    fname = f'/tmp/kmeans_{tlut_bits}_{V}.pt'
                    if not os.path.exists(fname):
                        tlut = torch.randn(2**tlut_bits, V)
                        import scipy
                        data = torch.randn(1 << 20, 2)
                        clusters = scipy.cluster.vq.kmeans(data, tlut)
                        tlut = torch.tensor(clusters[0])
                        tlut = (tlut /
                                tlut.std(unbiased=False)) * 0.9682458365518543
                        torch.save(tlut, fname)
                    else:
                        tlut = torch.load(fname)
                else:
                    raise Exception
                self.register_buffer('tlut', tlut)
                self.register_buffer(
                    'lut',
                    quantlut_sym(self.tlut, L, tlut_bits).T.contiguous())
            else:
                self.tlut = tlut
                self.recons_lut()
        else:
            raise Exception

        self.fakeinf = torch.tensor(torch.inf)

        self.register_buffer('sumdelta',
                             torch.arange(2**(K * V)) << (L - K * V))
        self.sumdelta = self.sumdelta.view(1, 1, -1)

        self.register_buffer('state', torch.arange(2**L).unsqueeze(0))
        self.register_buffer('state_cand',
                             (self.state >>
                              (K * V))[0, ::2**(K * V)].unsqueeze(-1) +
                             self.sumdelta)
        self.register_buffer('recons_state', self.recons(self.state))

        dtype = torch.int32
        self.state = self.state.to(dtype)
        self.sumdelta = self.sumdelta.to(dtype)

        self.version = 0

    def recons_lut(self):
        if self.decode_mode == 'lut':
            self.lut = self.tlut.T.contiguous()
        elif self.decode_mode == 'quantlut':
            self.lut = quantlut(self.tlut, self.L,
                                self.tlut_bits).T.contiguous()
        elif self.decode_mode == 'quantlut_sym':
            self.lut = quantlut_sym(self.tlut, self.L,
                                    self.tlut_bits).T.contiguous()

    def recons(self, encoded, **kwargs):
        return self.lut[:,
                        encoded.int().to(self.lut.device)].to(encoded.device)

    @torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=True)
    def update(self, cost, thing):
        state_err = (self.recons_state -
                     thing.unsqueeze(-1)).square().sum(dim=0)
        cand_cost = torch.gather(
            cost.unsqueeze(-2).expand(-1, self.state_cand.shape[1], -1), -1,
            self.state_cand.expand(len(cost), -1, 2**(self.K * self.V)))
        best = torch.min(cand_cost, dim=-1)
        cost = state_err + best.values.unsqueeze(-1).expand(
            -1, -1, 2**(self.K * self.V)).reshape(state_err.shape)
        prev_state = torch.gather(
            self.state_cand.expand(thing.shape[1], -1, -1), -1,
            best.indices.unsqueeze(-1))[..., 0]
        return prev_state, cost

    @torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=True)
    def update_loop(self, from_state_buf, Xbuf, cost, buf_sz, start_idx):
        for j in range(start_idx, buf_sz):
            from_state_buf[j], cost = self.update(
                cost, Xbuf[j * self.V:(j + 1) * self.V])
        return from_state_buf, cost

    def viterbi(self, X, overlap=None):
        T, B = X.shape
        assert T % self.V == 0
        # cost is (B, 2**L)
        cost = (self.recons_state -
                X[:self.V].unsqueeze(-1)).square().sum(dim=0)

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * self.fakeinf
            allow = (overlap <<
                     (self.K * self.V)).unsqueeze(-1) + torch.arange(
                         2**(self.K * self.V)).to(X.device).view(1, 1, -1)
            mask.scatter_(1, allow[0].long(), 0)
            cost = torch.min(cost + mask, self.fakeinf)

        from_state = torch.zeros(T // self.V,
                                 B,
                                 2**(self.L - self.K * self.V),
                                 dtype=self.state.dtype,
                                 device=self.state.device)

        buf_sz = 4
        for i in range(T // self.V // buf_sz):
            Xbuf = X[i * buf_sz * self.V:(i + 1) * buf_sz * self.V]
            from_state_buf = from_state[i * buf_sz:(i + 1) * buf_sz]
            from_state_buf, cost = self.update_loop(from_state_buf, Xbuf, cost,
                                                    buf_sz, 1 if
                                                    (i == 0) else 0)
            from_state[i * buf_sz:(i + 1) * buf_sz] = from_state_buf

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * self.fakeinf
            allow = (overlap.unsqueeze(-1) + self.sumdelta.unsqueeze(0))
            mask.scatter_(1, allow[0, 0].long(), 0)
            cost = torch.min(cost + mask, self.fakeinf)

        final_state = torch.zeros(T // self.V,
                                  B,
                                  dtype=self.idx_dtype,
                                  device=X.device)
        final_state[T // self.V - 1] = torch.argmin(cost, dim=-1)
        final_state = self.gather_loop(final_state, from_state, T)
        return final_state

    @torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=True)
    def gather_loop(self, final_state, from_state, T):
        for i in range(T // self.V - 1, 0, -1):
            final_state[i - 1] = torch.gather(
                from_state[i], -1, (final_state[i].long().unsqueeze(-1)) >>
                (self.K * self.V))[..., 0]
        return final_state

    def quantize_seq(self, X, overlap=None, **kwargs):
        T, NO = X.shape
        bs = min(2**(int(os.environ.get("QTIP_BS_LOG2", "20")) - self.L), NO)
        pad_amt = math.ceil(NO / bs) * bs - NO
        X = torch.nn.functional.pad(X, (0, pad_amt))
        T, N = X.shape
        X = X.reshape(T, N // bs, bs).transpose(0, 1).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap, (0, pad_amt))
            overlap = overlap.reshape(N // bs, bs)

        Qidxs = torch.zeros(N // bs,
                            T // self.V,
                            bs,
                            dtype=self.idx_dtype,
                            device=X.device)
        for i in range(len(X)):
            b_overlap = None if overlap is None else overlap[i]
            Qidxs[i] = self.viterbi(X[i], overlap=b_overlap)
        Qidxs = Qidxs.transpose(0, 1).reshape(T // self.V, N)[:, :NO]
        return Qidxs

    def quantize(self, X, **kwargs):
        X = X.T.contiguous().to(torch.float16)
        T = X.shape[0]
        roll_X = torch.roll(X, T // (2 * self.V) * self.V, 0)
        state = self.quantize_seq(roll_X, overlap=None)
        overlap = state[T // (2 * self.V)] >> self.K * self.V
        state = self.quantize_seq(X, overlap=overlap)
        hatX = self.recons(state).transpose(0, 1).reshape(X.shape)
        return hatX.T.contiguous().to(X.device), state.T.contiguous().to(
            X.device)

    def pack_trellis(self, trellis):
        # T is really T // self.V here
        B, T = trellis.shape
        bf = torch.zeros(B,
                         T * self.K * self.V + self.L - self.K * self.V,
                         dtype=bool,
                         device=trellis.device)
        bf[:, :self.L] = (trellis[:, 0].unsqueeze(-1) & (2**torch.arange(
            self.L, device=trellis.device).flip(dims=(-1, ))).unsqueeze(0)) > 0
        K_mask = 2**torch.arange(
            self.K * self.V,
            device=trellis.device).flip(dims=(-1, )).unsqueeze(0)
        for i in range(1, T):
            assert ((trellis[:, i - 1] &
                     ((1 << (self.L - self.K * self.V)) - 1)) == (
                         trellis[:, i] >> (self.K * self.V))).all()
            bf[:,
               (self.L +
                (i - 1) * self.K * self.V):(self.L + i * self.K * self.V)] = (
                    (trellis[:, i] &
                     ((1 <<
                       (self.K * self.V)) - 1)).unsqueeze(-1) & K_mask) > 0

        bf = bf[:, :-(self.L - self.K * self.V)]
        pad_amt = math.ceil(
            T * self.K * self.V / 16) * 16 - T * self.K * self.V
        bf = torch.nn.functional.pad(bf, (0, pad_amt)).reshape(
            -1, (T * self.K * self.V + pad_amt) // 16, 16)

        uint_mask = (2**torch.arange(
            16, dtype=torch.int32,
            device=bf.device)).flip(dims=(-1, )).unsqueeze(0).unsqueeze(0)
        bf_sum = (bf.to(torch.int32) * uint_mask).sum(dim=-1)
        return bf_sum.to(torch.uint16)

    def unpack_trellis(self, packed, T):
        packed = packed.view(torch.uint16).to(torch.int32)
        uint_mask = (2**torch.arange(
            16, dtype=torch.int32,
            device=packed.device)).flip(dims=(-1, )).unsqueeze(0).unsqueeze(0)
        bf = (packed.unsqueeze(-1) & uint_mask) > 0
        pad_amt = math.ceil(T * self.K / 16) * 16 - T * self.K
        bf = bf.reshape(-1, (T * self.K + pad_amt))[:, :T * self.K]
        bf = torch.concat([bf, bf[:, :self.L - self.K * self.V]], dim=-1)
        L_mask = (2**torch.arange(
            self.L, dtype=torch.int32,
            device=packed.device).flip(dims=(-1, ))).unsqueeze(0)
        K_mask = (2**torch.arange(
            self.K * self.V, dtype=torch.int32,
            device=packed.device).flip(dims=(-1, ))).unsqueeze(0)
        trellis = torch.zeros(bf.shape[0],
                              T // self.V,
                              dtype=torch.int32,
                              device=bf.device)
        trellis[:, 0] = (bf[:, :self.L].int() * L_mask).sum(dim=-1)
        for i in range(1, T // self.V):
            trellis[:, i] = ((trellis[:, i-1] << (self.K*self.V)) & ((1 << self.L) - 1)) + \
                (bf[:, self.L + (i-1)*self.K*self.V : self.L + i*self.K*self.V].int() * K_mask).sum(dim=-1)

        return trellis


# Building bitshift_codebook (the 2**L decode LUT + recons tables) costs ~0.5s and is
# identical for every module that shares (L, K, V, tlut_bits, decode_mode, tlut, device).
# Under the MoE fork there are ~20480 per-expert QuantizedLinear; rebuilding the codebook
# for each on its first forward was ~3h of CPU (looked like a hang, GPU idle). Memoize it
# so all identical experts reuse one codebook. hatW stays per-instance on BitshiftLinear,
# so sharing self.cb (read-only at inference: decode/get_hatW) is safe. Quantize path uses
# bitshift_codebook directly and is unaffected.
_CB_CACHE = {}


def _cached_bitshift_codebook(L, K, V, tlut_bits, decode_mode, tlut):
    if tlut is None:
        key = (L, K, V, tlut_bits, decode_mode, None, None)
    else:
        key = (L, K, V, tlut_bits, decode_mode, str(tlut.device),
               hash(tlut.detach().to('cpu', torch.float32).contiguous().numpy().tobytes()))
    cb = _CB_CACHE.get(key)
    if cb is None:
        cb = bitshift_codebook(L, K, V, tlut_bits, decode_mode, tlut=tlut)
        _CB_CACHE[key] = cb
    return cb


class BitshiftLinear(nn.Module):

    def __init__(self,
                 td_x,
                 td_y,
                 L,
                 K,
                 V,
                 tlut_bits,
                 decode_mode,
                 dtype=torch.float16,
                 tlut=None,
                 has_kernel=False):
        super().__init__()
        self.td_x = td_x
        self.td_y = td_y
        self.V = V
        self.cb = _cached_bitshift_codebook(L, K, V, tlut_bits, decode_mode, tlut)
        self.internal_dtype = dtype
        self.has_kernel = has_kernel
        self.scale = 32

    def get_hatW(self, unpacked_trellis, m, n):
        return self.cb.recons(unpacked_trellis).transpose(0, 1).transpose(
            1, 2).reshape(m // self.td_x, n // self.td_y, self.td_x,
                          self.td_y).transpose(1, 2).reshape(m, n)

    def get_hatW_kernel(self, trellis, m, n):
        out = decode_compressed(self.cb.L, self.cb.tlut_bits, self.cb.K,
                                int(math.log2(self.V)), m, n, trellis.view(-1),
                                self.cb.lut.T)
        return out

    def cache_hatW(self, packed_trellis, had_left, had_right, K_left, K_right,
                   m, n, rcp, tp_rank):
        if self.has_kernel:
            hatW = self.get_hatW_kernel(packed_trellis, m, n)
        else:
            hatW = self.get_hatW(
                self.cb.unpack_trellis(packed_trellis, self.td_x * self.td_y),
                m, n)
        hatW = hatW.float() / self.scale

        if rcp == 1:
            self.hatW = matmul_hadU_cuda(
                matmul_hadU_cuda(hatW.reshape(tp_rank * m, n // tp_rank),
                                 had_left, K_left).reshape(m, n).T, had_right,
                K_right).T.contiguous().to(self.internal_dtype)
        elif rcp == 2:
            self.hatW = matmul_hadU_cuda(
                matmul_hadU_cuda(hatW, had_left,
                                 K_left).T.reshape(tp_rank * n,
                                                   m // tp_rank), had_right,
                K_right).reshape(n, m).T.contiguous().to(self.internal_dtype)
        else:
            self.hatW = matmul_hadU_cuda(
                matmul_hadU_cuda(hatW, had_left, K_left).T, had_right,
                K_right).T.contiguous().to(self.internal_dtype)

    def forward(self,
                input,
                trellis,
                SU,
                SV,
                had_left,
                had_right,
                K_left,
                K_right,
                rcp,
                tp_rank,
                mode='eval',
                **kwargs):
        n, m = len(SU), len(SV)

        # Steady-state decode: replay a per-instance CUDA graph instead of relaunching
        # the ~10 tiny kernels (Hadamard x2 + qtip matvec) of this linear every token.
        if (_CUDAGRAPH and _CAPTURE_ENABLED and mode == 'eval'
                and self.has_kernel and input.is_cuda and input.numel() == n):
            return self._graph_forward(input, trellis, SU, SV, had_left,
                                       had_right, K_left, K_right, rcp, tp_rank,
                                       n, m)

        x = input.view(-1, n).to(torch.float32)
        x = x * SU

        if mode == 'train-fixW':
            x = (x.to(self.internal_dtype) @ self.hatW.T).float()
        else:
            bs = x.shape[0]

            if rcp == 1:
                x = matmul_hadUt_cuda(x.reshape(-1, n // tp_rank), had_left,
                                      K_left).reshape(x.shape) / self.scale
            else:
                x = matmul_hadUt_cuda(x, had_left, K_left) / self.scale

            if bs == 1 and self.has_kernel and os.environ.get('QTIP_BS1_TORCH','0') != '1':
                # Fast path: bypass torch.ops.quip_lib dispatch.
                # Eliminates per-call: f-string + getattr on torch.ops
                # (~10 us) + torch.library dispatcher (~10 us) + Python
                # impl + torch.zeros alloc+memset (~10 us) + repeated
                # reshape/view (~5 us). Cached once per BitshiftLinear
                # instance -- each instance only sees one (m, n, K) shape,
                # one trellis tensor, one tlut.
                if not hasattr(self, "_qtip_fast"):
                    import qtip_kernels as _qk
                    self._qtip_fast = getattr(
                        _qk,
                        f"decompress_matvec_16_9_{self.cb.K}_1_{m}_1_{x.numel()}")
                    # Kernel writes (does not accumulate) every output row
                    # under the current grid sizing, so empty() is safe --
                    # no memset needed per call.
                    self._qtip_out = torch.empty((m, 1),
                                                 dtype=torch.float32,
                                                 device=x.device)
                    self._qtip_trellis = trellis.reshape(-1).view(torch.int32)
                    self._qtip_tlut = self.cb.tlut.reshape(-1)

                self._qtip_fast(
                    self._qtip_out,
                    self._qtip_trellis,
                    x.to(torch.float16).T,
                    self._qtip_tlut,
                )
                x = self._qtip_out.T

            else:
                if mode == 'train-recons':
                    self.cb.recons_lut()

                if self.has_kernel:
                    # Fast path for small batches (MoE experts): the fused
                    # decode+matmat kernel shares one weight decode across all bs
                    # tokens in a single launch (grid.y tiles by NTILE), instead of
                    # materializing the full hatW every forward. Only used when a
                    # matmat instance exists for this (K, m, n) and bs is below the
                    # crossover where torch's amortized decode wins (~256).
                    bs = x.shape[0]
                    NTILE = 4
                    _mm = None
                    # Speed-bench (decode-only measurement): the fused matmat
                    # kernel illegal-accesses for the 35B attention/dense shapes
                    # (qtip-kernels inference.cu matvec launch), and no matmat
                    # instance is compiled for the 35B MoE expert shapes either.
                    # For any bs>1 eval call, take the correct torch get_hatW
                    # dequant path below instead (prefill only; slow but safe).
                    # The bs==1 fast matvec decode kernel path is untouched, so
                    # the measured decode speed still uses the kernel.
                    if os.environ.get("QTIP_NO_MATMAT", "0") == "1":
                        _mm = None
                    elif 1 < bs <= 256:
                        import qtip_kernels as _qk
                        _mm = getattr(
                            _qk,
                            f"decompress_matmat_16_9_{self.cb.K}_1_{m}_{NTILE}_{n}",
                            None)
                    if _mm is not None:
                        pad = (-bs) % NTILE
                        xin = x if pad == 0 else torch.cat(
                            [x, x.new_zeros(pad, n)], 0)
                        xin = xin.to(torch.float16).contiguous()
                        out = x.new_empty((xin.shape[0], m), dtype=torch.float32)
                        _mm(out, trellis.reshape(-1).view(torch.int32), xin,
                            self.cb.tlut.reshape(-1))
                        x = out[:bs]
                    else:
                        # No matmat kernel compiled for this (K, m, n) shape.
                        # The BitshiftLinearKernelAG path illegal-accesses for
                        # these dims (e.g. 35B MoE experts), so for bs>1 eval
                        # fall back to the torch get_hatW dequant path (correct,
                        # slow, only hit during prefill). The fast bs==1 matvec
                        # decode kernel path above is untouched.
                        _tr = self.cb.unpack_trellis(
                            trellis, self.td_x * self.td_y)
                        hatW = self.get_hatW(_tr, m, n)
                        x = (x.to(hatW.dtype) @ hatW.T).float()
                else:
                    if mode == 'eval':
                        trellis = self.cb.unpack_trellis(
                            trellis, self.td_x * self.td_y)
                    hatW = self.get_hatW(trellis, m, n)
                    x = (x.to(hatW.dtype) @ hatW.T).float()

            if rcp == 2:
                x = matmul_hadU_cuda(x.reshape(-1, m // tp_rank), had_right,
                                     K_right).reshape(x.shape)
            else:
                x = matmul_hadU_cuda(x, had_right, K_right)

        x = x.to(SV.device) * (SV * self.scale)
        return x.view(*input.shape[:-1], m).to(input.dtype)

    def _bs1_eval(self, input, trellis, SU, SV, had_left, had_right, K_left,
                  K_right, rcp, tp_rank, n, m):
        """Pure-compute bs=1 eval path (SU -> hadUt -> qtip matvec -> hadU -> SV).
        Mirrors the bs==1 fast branch of forward; used inside CUDA-graph capture."""
        x = input.view(-1, n).to(torch.float32) * SU
        if rcp == 1:
            x = matmul_hadUt_cuda(x.reshape(-1, n // tp_rank), had_left,
                                  K_left).reshape(x.shape) / self.scale
        else:
            x = matmul_hadUt_cuda(x, had_left, K_left) / self.scale
        if not hasattr(self, "_qtip_fast"):
            import qtip_kernels as _qk
            self._qtip_fast = getattr(
                _qk, f"decompress_matvec_16_9_{self.cb.K}_1_{m}_1_{x.numel()}")
            self._qtip_out = torch.empty((m, 1), dtype=torch.float32,
                                         device=x.device)
            self._qtip_trellis = trellis.reshape(-1).view(torch.int32)
            self._qtip_tlut = self.cb.tlut.reshape(-1)
        self._qtip_fast(self._qtip_out, self._qtip_trellis,
                        x.to(torch.float16).T, self._qtip_tlut)
        x = self._qtip_out.T
        if rcp == 2:
            x = matmul_hadU_cuda(x.reshape(-1, m // tp_rank), had_right,
                                 K_right).reshape(x.shape)
        else:
            x = matmul_hadU_cuda(x, had_right, K_right)
        x = x.to(SV.device) * (SV * self.scale)
        return x.view(*input.shape[:-1], m).to(input.dtype)

    def _graph_forward(self, input, trellis, SU, SV, had_left, had_right,
                       K_left, K_right, rcp, tp_rank, n, m):
        """Lazily capture, then replay, this linear's bs=1 eval graph."""
        if getattr(self, "_cg", None) is None:
            self._cg_in = torch.empty((1, n), dtype=input.dtype,
                                      device=input.device)
            self._cg_in.copy_(input.reshape(1, n))
            args = (trellis, SU, SV, had_left, had_right, K_left, K_right,
                    rcp, tp_rank, n, m)
            # warm up (build hadK cache, qtip fast handle, allocator) off-stream
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._bs1_eval(self._cg_in, *args)
            torch.cuda.current_stream().wait_stream(s)
            # Drain ALL in-flight work before capturing. During generation the
            # decode step's other kernels (attention, other experts, routing) may
            # still be queued on the default stream; capture must start from a
            # quiescent device.
            torch.cuda.synchronize()
            self._cg = torch.cuda.CUDAGraph()
            # NOTE: we deliberately do NOT use the torch.cuda.graph(...) context
            # manager here. Its __enter__ calls gc.collect(), which -- when we are
            # nested inside a live generation forward pass holding hundreds of CUDA
            # tensors in reference cycles -- runs tensor finalizers (cudaFree) that
            # deadlock against the just-issued synchronize/capture setup. We drive
            # capture_begin/capture_end directly on our own side stream instead
            # (mirrors probe_cudagraph.py, minus gc). capture_error_mode
            # "thread_local" keeps cudaStreamBeginCapture from handshaking with the
            # inductor compile-worker / allocator background threads (another hang
            # source under "global" mode).
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                self._cg.capture_begin(_graph_pool(),
                                       capture_error_mode="thread_local")
                self._cg_out = self._bs1_eval(self._cg_in, *args)
                self._cg.capture_end()
            torch.cuda.current_stream().wait_stream(capture_stream)
        self._cg_in.copy_(input.reshape(1, n))
        self._cg.replay()
        # clone: the static out buffer is overwritten on the next replay
        return self._cg_out.view(*input.shape[:-1], m).clone()


class BitshiftLinearKernelAG(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, trellis, m, n, L, tlut_bits, K, V, lut):
        ctx.save_for_backward(trellis, lut)
        ctx.L = L
        ctx.tlut_bits = tlut_bits
        ctx.K = K
        ctx.V = V
        ctx.m = m
        ctx.n = n

        hatW = decode_compressed(L, tlut_bits, K, int(math.log2(V)), m, n,
                                 trellis.view(-1), lut.T)
        return input.to(hatW.dtype) @ hatW.T

    @staticmethod
    def backward(ctx, grad_output):
        trellis, lut = ctx.saved_tensors
        L = ctx.L
        tlut_bits = ctx.tlut_bits
        K = ctx.K
        V = ctx.V
        m = ctx.m
        n = ctx.n

        hatW = decode_compressed(L, tlut_bits, K, int(math.log2(V)), m, n,
                                 trellis.view(-1), lut.T)

        grad_input = grad_output.to(hatW.dtype) @ hatW
        return grad_input, None, None, None, None, None, None, None, None
