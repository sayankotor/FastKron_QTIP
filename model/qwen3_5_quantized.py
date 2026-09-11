"""
Thin quantized fork of Qwen3.5 (hybrid DeltaNet + full-attn, multimodal).

Instead of hand-reimplementing ~2000 lines of modeling (like model/qwen2.py does for
Qwen2), this SUBCLASSES the real transformers Qwen3_5ForConditionalGeneration and, in
__init__, swaps the TEXT projection nn.Linear modules for QuantizedLinear according to
config.quip_params. Everything else — the hybrid DeltaNet/full-attn logic, the vision
encoder, the hybrid cache, rope — is inherited unchanged.

Result: a self-loading COMPRESSED checkpoint. from_pretrained builds QuantizedLinear
where the projections are quantized (trellis buffers loaded by name from the saved
state_dict) and keeps the vision encoder + embeddings + lm_head + norms + DeltaNet
gates/conv in fp16. The trellis is stored compressed (~4/2 bits); QuantizedLinear
decodes on the fly (CUDA kernel when has_kernel, else torch fallback).

Quantized text projections (per config.quip_params, minus skip_list):
  full_attention  layer -> self_attn.{q,k,v,o}_proj
  linear_attention layer -> linear_attn.{in_proj_qkv, in_proj_z, out_proj}
  both                   -> mlp.{gate,up,down}_proj
Left fp16 (never quantized): router/gates in_proj_a/b, conv1d, norms, q/k_norm,
embed_tokens, lm_head, and the entire vision tower.
"""
import torch
from torch import nn

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

from lib.linear.quantized_linear import QuantizedLinear
from lib.utils import get_hadK


def materialize_hadK(model):
    """Recompute the non-persistent Hadamard buffers of every QuantizedLinear.

    QuantizedLinear registers had_left/had_right as persistent=False buffers built
    in __init__. Under from_pretrained(low_cpu_mem_usage=True) the module is created
    on the meta device, so those buffers are meta/empty and are NOT restored from the
    checkpoint (they're not in the state_dict) -> NaN in the Hadamard transform.
    Call this after loading a compressed checkpoint to materialize them on the right
    device.
    """
    for mod in model.modules():
        if isinstance(mod, QuantizedLinear):
            dev = mod.SU.device
            hl, kl = get_hadK(mod.in_features)
            hr, kr = get_hadK(mod.out_features)
            mod.had_left = None if hl is None else hl.to(dev)
            mod.had_right = None if hr is None else hr.to(dev)
            mod.K_left, mod.K_right = kl, kr
    # Install the bs=1 decode CUDA-graph fast paths (idempotent, env-gated). Only
    # active under QTIP_CUDAGRAPH; the install functions themselves are no-ops when
    # already installed. install_dense_mlp_graph graphs the dense MLP; the
    # DeltaNet graph covers the linear_attention token-mixer.
    if _bs._CUDAGRAPH:
        install_dense_mlp_graph()
        install_deltanet_graph()
    return model


# save-name for each projection -> used to check skip_list ("{idx}_{name}")
_FULL_ATTN = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}
_DELTA_ATTN = {"in_proj_qkv": "in_proj_qkv", "in_proj_z": "in_proj_z",
               "out_proj": "out_proj"}
_MLP = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def _swap(parent, attr, qp, dtype):
    """Replace parent.<attr> (an nn.Linear) with a QuantizedLinear of same shape."""
    lin = getattr(parent, attr)
    q = QuantizedLinear(
        lin.in_features, lin.out_features,
        qp["td_x"], qp["td_y"], qp["L"], qp["K"], qp["V"],
        qp["tlut_bits"], qp["decode_mode"],
        dtype=dtype, bias=(lin.bias is not None),
    )
    # The trellis was packed with one layout (kernel vs plain) at quantize time;
    # decode MUST use the same. Honor qp['packed_for_kernel'] (default True — old
    # checkpoints were packed when kernels worked). The kernel-format decode
    # (decode_compressed) is pure torch, so this is correct even if the CUDA .so
    # can't import; the fused-kernel matvec is only taken in eval mode when the
    # .so is actually present.
    q.has_kernel = bool(qp.get("packed_for_kernel", True))
    setattr(parent, attr, q)


class Qwen3_5QuantizedForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 MM model with text projections swapped to QuantizedLinear."""

    def __init__(self, config):
        super().__init__(config)
        qp = getattr(config, "quip_params", None)
        if qp is None:
            return                                   # behaves as the vanilla model
        skip = set(qp.get("skip_list") or [])
        dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""))

        layers = self.model.language_model.layers
        for idx, layer in enumerate(layers):
            # attention: full vs DeltaNet, mutually exclusive
            if hasattr(layer, "self_attn"):
                for attr, name in _FULL_ATTN.items():
                    if hasattr(layer.self_attn, attr) and f"{idx}_{name}" not in skip:
                        _swap(layer.self_attn, attr, qp, dtype)
            elif hasattr(layer, "linear_attn"):
                for attr, name in _DELTA_ATTN.items():
                    if hasattr(layer.linear_attn, attr) and f"{idx}_{name}" not in skip:
                        _swap(layer.linear_attn, attr, qp, dtype)
            # dense MLP (27B). MoE experts are not handled here (35B path).
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
                for attr, name in _MLP.items():
                    if hasattr(layer.mlp, attr) and f"{idx}_{name}" not in skip:
                        _swap(layer.mlp, attr, qp, dtype)


# --- CUDA-graph fast path for the dense MLP (27B) -----------------------------
# Direct port of the MoE shared-expert MLP graph (see qwen3_5_moe_quantized.py):
# the 3 projections are quantized with FIXED weights and the chain shape is static,
# so ONE graph per Qwen3_5MLP instance is valid for any bs=1 input vector. Collapses
# 3 per-linear graph replays + the python act/mul into a single replay. Env-gated
# via QTIP_CUDAGRAPH and only active inside a bs=1 decode capture window; otherwise
# byte-identical to eager.
from lib.codebook import bitshift as _bs


def _dense_mlp_chain(mlp, x):
    """bs=1 compute: down(act(gate(x))*up(x)). Inner linears run their plain eager
    kernel path (capture disabled) so kernels record into the enclosing per-MLP graph
    rather than trying to replay their own graph (no illegal nested capture)."""
    prev = _bs._CAPTURE_ENABLED
    _bs._CAPTURE_ENABLED = False
    try:
        return mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))
    finally:
        _bs._CAPTURE_ENABLED = prev


def _dense_mlp_graphed_forward(mlp, x):
    if (_bs._CUDAGRAPH and _bs._CAPTURE_ENABLED and x.dim() == 2 and x.shape[0] == 1
            and isinstance(mlp.gate_proj, QuantizedLinear)):
        cg = getattr(mlp, "_cg", None)
        if cg is None:
            cin = torch.empty_like(x)
            cin.copy_(x)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    _dense_mlp_chain(mlp, cin)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            cap = torch.cuda.Stream(); cap.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(cap):
                g.capture_begin(_bs._graph_pool(), capture_error_mode="thread_local")
                cout = _dense_mlp_chain(mlp, cin)
                g.capture_end()
            torch.cuda.current_stream().wait_stream(cap)
            mlp._cg = (g, cin, cout)
            cg = mlp._cg
        g, cin, cout = cg
        cin.copy_(x)
        g.replay()
        return cout.clone()
    return _dense_mlp_orig_forward(mlp, x)


_dense_mlp_orig_forward = None


def install_dense_mlp_graph():
    """Monkeypatch Qwen3_5MLP.forward with the graphed bs=1 fast path (idempotent)."""
    global _dense_mlp_orig_forward
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
    if _dense_mlp_orig_forward is not None:
        return
    _dense_mlp_orig_forward = Qwen3_5MLP.forward
    Qwen3_5MLP.forward = _dense_mlp_graphed_forward


# === DELTANET GRAPHED DECODE ================================================
# Direct port of the MoE DeltaNet graph (see qwen3_5_moe_quantized.py). Graph the
# linear_attention (DeltaNet) token-mixer at bs=1 steady-state decode (48 of the
# 64 layers are linear_attention). The dense Qwen3_5GatedDeltaNet.forward is
# byte-identical to the MoE Qwen3_5MoeGatedDeltaNet.forward (same signature, same
# attribute/method names, same use_precomputed_states branch), and the dense cache
# Qwen3_5DynamicCache exposes conv_states / recurrent_states / has_previous_state
# identically -- so this is a mechanical class-name-only port.
#
# WHY IT COULDN'T BE GRAPHED BEFORE: HF's DynamicCache stores per-layer conv/
# recurrent state in python lists and REASSIGNS the slot each decode step (the
# fused recurrent kernel returns a freshly-allocated final state). A CUDA graph
# replays fixed addresses, so a captured graph would read stale pointers on the
# next token. FIX: give each DeltaNet a pair of static, fixed-address buffers
# (_dn_conv, _dn_rec); the conv kernel already updates its state IN-PLACE, and for
# the recurrent state we pass the static buffer as initial_state and copy_ the
# fresh final_state back into it -- so the address never moves across steps. We
# keep the cache's list slots pointing at these same static buffers so
# has_previous_state / prefill->decode handoff is intact.
import torch.nn.functional as _Fdn


def _deltanet_decode_compute(self, hidden_states):
    """Pure bs=1, seq_len==1 DeltaNet decode compute using this module's STATIC
    conv/recurrent buffers (self._dn_conv / self._dn_rec). Mirrors the
    use_precomputed_states branch of Qwen3_5GatedDeltaNet.forward exactly, but
    reads/writes fixed-address buffers so it is CUDA-graph capturable. The quantized
    in/out projections run their plain eager kernel path (capture disabled) so their
    kernels record directly into the enclosing graph (no illegal nested capture)."""
    prev = _bs._CAPTURE_ENABLED
    _bs._CAPTURE_ENABLED = False
    try:
        batch_size, seq_len, _ = hidden_states.shape  # (1, 1, hidden)
        conv_state = self._dn_conv
        recurrent_state = self._dn_rec

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # in-place conv-state update (causal_conv1d_update writes conv_state.copy_)
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * _Fdn.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # write fresh final state back into the STATIC recurrent buffer (fixed addr)
        recurrent_state.copy_(last_recurrent_state)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output
    finally:
        _bs._CAPTURE_ENABLED = prev


def _deltanet_graphed_forward(self, hidden_states, cache_params=None,
                              cache_position=None, attention_mask=None):
    """Graphed bs=1 decode fast path for Qwen3_5GatedDeltaNet.

    Active only under QTIP_CUDAGRAPH capture window at genuine steady-state decode
    (seq_len==1, cache has previous state). Otherwise falls through to the original
    forward (prefill, bs>1, flag off) -> byte-for-byte unchanged eager behaviour.
    One graph per DeltaNet instance (weights fixed) valid for any input vector."""
    use_decode = (
        _bs._CUDAGRAPH and _bs._CAPTURE_ENABLED
        and cache_params is not None and cache_params.has_previous_state
        and hidden_states.dim() == 3 and hidden_states.shape[0] == 1
        and hidden_states.shape[1] == 1 and cache_position is not None
        and isinstance(self.in_proj_qkv, QuantizedLinear)
    )
    if not use_decode:
        return _deltanet_orig_forward(
            self, hidden_states, cache_params=cache_params,
            cache_position=cache_position, attention_mask=attention_mask)

    li = self.layer_idx
    # (Re)initialise the static buffers from whatever the cache currently holds.
    # This runs once per generate() call (buffers absent OR cache slot changed
    # identity because a new prefill built a fresh state). After that the cache
    # slots are pinned to our static buffers and never reassigned by us.
    cur_conv = cache_params.conv_states[li]
    cur_rec = cache_params.recurrent_states[li]
    need_init = (
        getattr(self, "_dn_conv", None) is None
        or self._dn_conv.shape != cur_conv.shape
        or (cur_conv.data_ptr() != self._dn_conv.data_ptr())
        or (cur_rec is not None and cur_rec.data_ptr() != self._dn_rec.data_ptr())
    )
    if need_init:
        self._dn_conv = cur_conv.clone()
        self._dn_rec = cur_rec.clone()
        cache_params.conv_states[li] = self._dn_conv
        cache_params.recurrent_states[li] = self._dn_rec
        # invalidate any stale graph (buffers moved)
        self._dn_cg = None

    cg = getattr(self, "_dn_cg", None)
    if cg is None:
        cin = torch.empty_like(hidden_states)
        cin.copy_(hidden_states)
        # snapshot buffers so warmup (which mutates them in-place) doesn't corrupt
        # the real decode state; restore before capture.
        conv_save = self._dn_conv.clone()
        rec_save = self._dn_rec.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._dn_conv.copy_(conv_save); self._dn_rec.copy_(rec_save)
                _deltanet_decode_compute(self, cin)
        torch.cuda.current_stream().wait_stream(s)
        self._dn_conv.copy_(conv_save); self._dn_rec.copy_(rec_save)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        cap = torch.cuda.Stream(); cap.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap):
            g.capture_begin(_bs._graph_pool(), capture_error_mode="thread_local")
            cout = _deltanet_decode_compute(self, cin)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(cap)
        self._dn_cg = (g, cin, cout)
        cg = self._dn_cg
    g, cin, cout = cg
    cin.copy_(hidden_states)
    g.replay()
    return cout.clone()


_deltanet_orig_forward = None


def install_deltanet_graph():
    """Monkeypatch Qwen3_5GatedDeltaNet.forward with the graphed decode path."""
    global _deltanet_orig_forward
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    if _deltanet_orig_forward is not None:
        return
    _deltanet_orig_forward = Qwen3_5GatedDeltaNet.forward
    Qwen3_5GatedDeltaNet.forward = _deltanet_graphed_forward
# === END DELTANET GRAPHED DECODE ============================================
