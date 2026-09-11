"""
Thin quantized fork of Qwen3.5-MoE (35B-A3B: hybrid DeltaNet + full-attn, MoE, MM).

Same idea as model/qwen3_5_quantized.py (the 27B dense-MLP fork), but this model is a
Mixture-of-Experts: every decoder layer's MLP is a Qwen3_5MoeSparseMoeBlock holding
  - shared_expert : a normal MLP (gate/up/down nn.Linear)   -> quantized as BODY (K)
  - experts       : 256 routed experts as 3D nn.Parameters   -> quantized at EXPERT_BITS
  - gate / shared_expert_gate : router -> kept fp16, never quantized

MIXED PRECISION: the body (attention + shared_expert) is quantized at quip_params['K']
(=4 bit); the routed experts at quip_params['expert_bits'] (=2 bit). Two things are
swapped in __init__:
  1. every text projection nn.Linear (attn q/k/v/o or DeltaNet in_proj_*/out_proj, and
     mlp.shared_expert.{gate,up,down}_proj) -> QuantizedLinear  (K bit)
  2. mlp.experts (Qwen3_5MoeExperts, 3D params) -> QuantizedExperts, which holds one
     QuantizedLinear per (expert, projection) at EXPERT_BITS and reproduces the exact
     routed-expert forward.

Result: a self-loading COMPRESSED checkpoint. from_pretrained builds the QuantizedLinear
tree (trellis buffers loaded by name from the saved state_dict) and keeps the vision
encoder + embeddings + lm_head + norms + router gates in fp16. Everything else — the
hybrid DeltaNet/full-attn logic, the vision encoder, the hybrid cache, rope — is
inherited unchanged from the real Qwen3_5MoeForConditionalGeneration.

Left fp16 (never quantized): router mlp.gate.weight, shared_expert_gate, conv1d, norms,
q/k_norm, embed_tokens, lm_head, and the entire vision tower.
"""
import os
import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)

from lib.linear.quantized_linear import QuantizedLinear
from lib.utils import get_hadK
from lib.codebook import bitshift as _bs


def _mlp_chain(mlp, x):
    """bs=1 compute of a Qwen3_5MoeMLP (shared expert): down(act(gate(x))*up(x)).
    Linears run their plain eager kernel path (capture disabled) so the kernels are
    recorded directly into the enclosing per-MLP graph -- no illegal nested capture."""
    prev = _bs._CAPTURE_ENABLED
    _bs._CAPTURE_ENABLED = False
    try:
        return mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))
    finally:
        _bs._CAPTURE_ENABLED = prev


def _mlp_graphed_forward(mlp, x):
    """Graph-replay fast path for the shared-expert MLP at bs=1 decode.

    The 3 projections are quantized (K=4) with FIXED weights, and the chain shape is
    static, so one graph per MLP instance is valid for any input vector -- same proof
    as the per-linear / per-expert graphs. Collapses 3 per-linear replays + the python
    act/mul into ONE graph replay. Falls back to the original chain when not in a bs=1
    decode capture window (prefill / bs>1 / flag off). Byte-identical to eager."""
    if (_bs._CUDAGRAPH and _bs._CAPTURE_ENABLED and x.dim() == 2 and x.shape[0] == 1
            and isinstance(mlp.gate_proj, QuantizedLinear)):
        cg = getattr(mlp, "_cg", None)
        if cg is None:
            cin = torch.empty_like(x)
            cin.copy_(x)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    _mlp_chain(mlp, cin)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            cap = torch.cuda.Stream(); cap.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(cap):
                g.capture_begin(_bs._graph_pool(), capture_error_mode="thread_local")
                cout = _mlp_chain(mlp, cin)
                g.capture_end()
            torch.cuda.current_stream().wait_stream(cap)
            mlp._cg = (g, cin, cout)
            cg = mlp._cg
        g, cin, cout = cg
        cin.copy_(x)
        g.replay()
        return cout.clone()
    return _mlp_orig_forward(mlp, x)


_mlp_orig_forward = None


def install_shared_mlp_graph():
    """Monkeypatch Qwen3_5MoeMLP.forward with the graphed bs=1 fast path (idempotent)."""
    global _mlp_orig_forward
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP
    if _mlp_orig_forward is not None:
        return
    _mlp_orig_forward = Qwen3_5MoeMLP.forward
    Qwen3_5MoeMLP.forward = _mlp_graphed_forward


def _mod_dtype(mod):
    """Expected input dtype of an expert linear: QuantizedLinear carries `.dtype`
    (SU/SV buffers), a dead expert is a plain nn.Linear with a `.weight`."""
    dt = getattr(mod, "dtype", None)
    if dt is not None:
        return dt
    return mod.weight.dtype


def materialize_hadK(model):
    """Recompute the non-persistent Hadamard buffers of every QuantizedLinear.

    QuantizedLinear registers had_left/had_right as persistent=False buffers built in
    __init__. Under from_pretrained(low_cpu_mem_usage=True) modules are created on the
    meta device, so those buffers are meta/empty and are NOT restored from the
    checkpoint (they're not in the state_dict) -> NaN in the Hadamard transform. Call
    this after loading a compressed checkpoint to rebuild them on the right device.
    (Covers body AND expert QuantizedLinear — it iterates every module.)
    """
    for mod in model.modules():
        if isinstance(mod, QuantizedLinear):
            dev = mod.SU.device
            hl, kl = get_hadK(mod.in_features)
            hr, kr = get_hadK(mod.out_features)
            mod.had_left = None if hl is None else hl.to(dev)
            mod.had_right = None if hr is None else hr.to(dev)
            mod.K_left, mod.K_right = kl, kr
    # Install the shared-expert MLP graph fast path (no-op unless QTIP_CUDAGRAPH=1 and
    # in a bs=1 decode capture window; the eager path is byte-for-byte unchanged).
    install_shared_mlp_graph()
    install_deltanet_graph()
    # install_fullattn_graph()  # DISABLED: static-KV graph diverged + slower; DeltaNet-only is the win
    return model


# save-name for each projection -> used to check skip_list ("{idx}_{name}")
_FULL_ATTN = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}
_DELTA_ATTN = {"in_proj_qkv": "in_proj_qkv", "in_proj_z": "in_proj_z",
               "out_proj": "out_proj"}
# MoE body MLP = the shared expert (router mlp.gate stays fp16).
_SHARED_MLP = {"gate_proj": "shared_gate", "up_proj": "shared_up",
               "down_proj": "shared_down"}


def _resolve_dtype(config):
    dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.replace("torch.", ""))
    return dtype


def _make_qlin(in_f, out_f, qp, K, dtype):
    q = QuantizedLinear(
        in_f, out_f,
        qp["td_x"], qp["td_y"], qp["L"], K, qp["V"],
        qp["tlut_bits"], qp["decode_mode"],
        dtype=dtype, bias=False,
    )
    # The trellis was packed with one layout (kernel vs plain) at quantize time; decode
    # MUST use the same. Honor qp['packed_for_kernel'] (default True — old checkpoints
    # were packed when kernels worked). The kernel-format decode is pure torch, so this
    # is correct even when the CUDA .so can't import.
    q.has_kernel = bool(qp.get("packed_for_kernel", True))
    return q


def _swap(parent, attr, qp, K, dtype):
    """Replace parent.<attr> (an nn.Linear) with a QuantizedLinear of the same shape."""
    lin = getattr(parent, attr)
    setattr(parent, attr, _make_qlin(lin.in_features, lin.out_features, qp, K, dtype))



# === DELTANET GRAPHED DECODE ================================================
# Graph the linear_attention (DeltaNet) token-mixer at bs=1 steady-state decode.
# WHY IT COULDN'T BE GRAPHED BEFORE: HF's Qwen3_5MoeDynamicCache stores the
# per-layer conv/recurrent state in python lists and REASSIGNS the slot each
# decode step (the fused recurrent kernel returns a freshly-allocated final
# state). A CUDA graph replays fixed addresses, so a captured graph would read
# stale pointers on the next token. FIX: give each DeltaNet a pair of static,
# fixed-address buffers (_dn_conv, _dn_rec); the conv kernel already updates its
# state IN-PLACE, and for the recurrent state we pass the static buffer as
# initial_state and copy_ the fresh final_state back into it -- so the address
# never moves across steps. We keep the cache's list slots pointing at these
# same static buffers so has_previous_state / prefill->decode handoff is intact.
# Verified standalone: both causal_conv1d_update and fused_recurrent_gated_delta_rule
# capture and replay byte-identically.
import torch.nn.functional as _Fdn


def _deltanet_decode_compute(self, hidden_states):
    """Pure bs=1, seq_len==1 DeltaNet decode compute using this module's STATIC
    conv/recurrent buffers (self._dn_conv / self._dn_rec). Mirrors the
    use_precomputed_states branch of Qwen3_5MoeGatedDeltaNet.forward exactly, but
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
    """Graphed bs=1 decode fast path for Qwen3_5MoeGatedDeltaNet.

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
    """Monkeypatch Qwen3_5MoeGatedDeltaNet.forward with the graphed decode path."""
    global _deltanet_orig_forward
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet
    if _deltanet_orig_forward is not None:
        return
    _deltanet_orig_forward = Qwen3_5MoeGatedDeltaNet.forward
    Qwen3_5MoeGatedDeltaNet.forward = _deltanet_graphed_forward
# === END DELTANET GRAPHED DECODE ============================================



# === FULLATTN GRAPHED DECODE ================================================
# Graph the full_attention token-mixer (every 4th layer) at bs=1 steady-state
# decode. HF grows the KV cache by torch.cat each step (address + length change),
# which a CUDA graph can't follow. FIX: per-attn-instance STATIC KV buffers of
# fixed capacity CAP (= prompt_len at first decode + decode budget). Each step we
# index_copy_ the new k/v at a static position tensor and SDPA over the full CAP
# with a STATIC additive mask that reveals only positions <= current pos. pos and
# the mask are updated in-place OUTSIDE the graph (cheap elementwise, no host
# sync), so the graph's baked addresses stay valid. Verified standalone: byte
# identical across a multi-step sequence. Env QTIP_ATTN_DECODE_BUDGET (default
# 1024) sizes the extra KV room; if generation would exceed CAP we fall back to
# the eager forward for that step (correctness preserved).
import torch.nn.functional as _Ffa
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    apply_rotary_pos_emb as _apply_rope, repeat_kv as _repeat_kv)

_ATTN_BUDGET = int(os.environ.get("QTIP_ATTN_DECODE_BUDGET", "1024"))


def _fullattn_decode_compute(self, hidden_states, cos, sin):
    """Pure bs=1 seq_len==1 full-attn decode compute against this module's STATIC
    KV buffers (self._kv_k / self._kv_v) at static write index self._kv_pos, over a
    fixed CAP with static additive mask self._kv_mask. Quantized q/k/v/o run their
    eager kernel path (capture disabled) so kernels record into the enclosing graph."""
    prev = _bs._CAPTURE_ENABLED
    _bs._CAPTURE_ENABLED = False
    try:
        input_shape = hidden_states.shape[:-1]           # (1, 1)
        hidden_shape = (*input_shape, -1, self.head_dim)
        q, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = _apply_rope(query, key, cos, sin)
        # in-place KV write at static position
        self._kv_k.index_copy_(2, self._kv_pos, key.to(self._kv_k.dtype))
        self._kv_v.index_copy_(2, self._kv_pos, value.to(self._kv_v.dtype))
        k_all = _repeat_kv(self._kv_k, self.num_key_value_groups)
        v_all = _repeat_kv(self._kv_v, self.num_key_value_groups)
        attn = _Ffa.scaled_dot_product_attention(
            query, k_all.to(query.dtype), v_all.to(query.dtype),
            attn_mask=self._kv_mask, scale=self.scaling)
        attn = attn.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn = attn * torch.sigmoid(gate)
        return self.o_proj(attn)
    finally:
        _bs._CAPTURE_ENABLED = prev


def _fullattn_graphed_forward(self, hidden_states, position_embeddings,
                              attention_mask=None, past_key_values=None,
                              cache_position=None, **kwargs):
    cos, sin = position_embeddings
    use_decode = (
        _bs._CUDAGRAPH and _bs._CAPTURE_ENABLED
        and past_key_values is not None
        and hidden_states.dim() == 3 and hidden_states.shape[0] == 1
        and hidden_states.shape[1] == 1 and cache_position is not None
        and isinstance(self.q_proj, QuantizedLinear))
    if use_decode:
        li = self.layer_idx
        kc = past_key_values.key_cache[li]
        vc = past_key_values.value_cache[li]
        use_decode = kc is not None and vc is not None
    if not use_decode:
        return _fullattn_orig_forward(
            self, hidden_states, position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values,
            cache_position=cache_position, **kwargs)

    li = self.layer_idx
    kc = past_key_values.key_cache[li]
    vc = past_key_values.value_cache[li]
    past_len = kc.shape[2]                       # tokens already in KV (this step's not yet added)
    cur_pos = int(cache_position[-1].item())     # absolute index for this token
    # (Re)initialise static KV buffers once per generate() (fresh prefill KV).
    need_init = (
        getattr(self, "_kv_k", None) is None
        or self._kv_k.shape[1] != kc.shape[1] or self._kv_k.shape[3] != kc.shape[3]
        or getattr(self, "_kv_base", None) is None)
    if need_init:
        cap = past_len + _ATTN_BUDGET
        b, hkv, _, hd = kc.shape
        self._kv_k = torch.zeros(b, hkv, cap, hd, dtype=kc.dtype, device=kc.device)
        self._kv_v = torch.zeros(b, hkv, cap, hd, dtype=vc.dtype, device=vc.device)
        self._kv_k[:, :, :past_len].copy_(kc)
        self._kv_v[:, :, :past_len].copy_(vc)
        self._kv_cap = cap
        self._kv_base = past_len          # first decode writes at index past_len
        self._kv_pos = torch.zeros(1, dtype=torch.long, device=kc.device)
        # mask dtype MUST match query dtype (SDPA additive-bias requirement); q is bf16.
        self._kv_mask = torch.full((1, 1, 1, cap), float("-inf"),
                                   dtype=kc.dtype, device=kc.device)
        self._kv_cg = None
    if cur_pos >= self._kv_cap:
        # out of budget -> eager fallback (and drop graph so a later re-init is clean)
        return _fullattn_orig_forward(
            self, hidden_states, position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values,
            cache_position=cache_position, **kwargs)
    # update static position + mask (reveal 0..cur_pos) BEFORE replay (outside graph)
    self._kv_pos.fill_(cur_pos)
    self._kv_mask[..., cur_pos] = 0.0
    # keep HF's cache object consistent so downstream length bookkeeping works:
    # write this step's k/v placeholder into the real cache via its normal update
    # would re-cat; instead we mirror our static buffer view back into the slot.
    cg = getattr(self, "_kv_cg", None)
    if cg is None:
        cos_s = cos.clone(); sin_s = sin.clone()
        cin = torch.empty_like(hidden_states); cin.copy_(hidden_states)
        # snapshot KV + mask so warmup writes don't corrupt real state
        k_save = self._kv_k.clone(); v_save = self._kv_v.clone()
        m_save = self._kv_mask.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._kv_k.copy_(k_save); self._kv_v.copy_(v_save)
                _fullattn_decode_compute(self, cin, cos_s, sin_s)
        torch.cuda.current_stream().wait_stream(s)
        self._kv_k.copy_(k_save); self._kv_v.copy_(v_save); self._kv_mask.copy_(m_save)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        cap_s = torch.cuda.Stream(); cap_s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap_s):
            g.capture_begin(_bs._graph_pool(), capture_error_mode="thread_local")
            cout = _fullattn_decode_compute(self, cin, cos_s, sin_s)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(cap_s)
        self._kv_cg = (g, cin, cos_s, sin_s, cout)
        cg = self._kv_cg
    g, cin, cos_s, sin_s, cout = cg
    cin.copy_(hidden_states); cos_s.copy_(cos); sin_s.copy_(sin)
    g.replay()
    # Advance HF cache length by appending this step's k/v (kept for bookkeeping /
    # get_seq_length); values already live in our static buffer, but transformers
    # tracks length via key_cache[li].shape[2]. Append a 1-token slice so lengths
    # stay in sync WITHOUT a cat of the whole KV each step being on the hot path.
    past_key_values.key_cache[li] = self._kv_k[:, :, :cur_pos + 1]
    past_key_values.value_cache[li] = self._kv_v[:, :, :cur_pos + 1]
    return cout.clone(), None


_fullattn_orig_forward = None


def install_fullattn_graph():
    global _fullattn_orig_forward
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeAttention
    if _fullattn_orig_forward is not None:
        return
    _fullattn_orig_forward = Qwen3_5MoeAttention.forward
    Qwen3_5MoeAttention.forward = _fullattn_graphed_forward
# === END FULLATTN GRAPHED DECODE ============================================


class QuantizedExperts(nn.Module):
    """Drop-in replacement for Qwen3_5MoeExperts with per-expert QuantizedLinear.

    The original stores gate_up_proj [E, 2*inter, hidden] and down_proj [E, hidden,
    inter] as 3D nn.Parameters and, per hit expert, does F.linear(state, W[e]). Here
    each expert's two matrices are a QuantizedLinear (K=expert_bits) so the checkpoint
    stays compressed on disk; the forward is otherwise identical.
    """

    def __init__(self, text_config, qp, dtype, dead=None):
        super().__init__()
        E = text_config.num_experts
        hidden = text_config.hidden_size
        inter = text_config.moe_intermediate_size
        self.num_experts = E
        self.hidden_dim = hidden
        self.intermediate_dim = inter
        self.top_k = text_config.num_experts_per_tok
        self.act_fn = ACT2FN[text_config.hidden_act]
        Ke = qp.get("expert_bits") or qp["K"]
        # dead = {(e, 'gate_up_proj'|'down_proj')} kept as fp16 nn.Linear so their
        # ORIGINAL bf16 weight loads (variant a: dead experts left unquantized).
        dead = dead or set()

        def _mk(e, proj, in_f, out_f):
            if (e, proj) in dead:
                return nn.Linear(in_f, out_f, bias=False, dtype=dtype)
            return _make_qlin(in_f, out_f, qp, Ke, dtype)

        # gate_up: hidden -> 2*inter ; down: inter -> hidden   (out = weight.shape[0])
        self.gate_up = nn.ModuleList(
            [_mk(e, 'gate_up_proj', hidden, 2 * inter) for e in range(E)])
        self.down = nn.ModuleList(
            [_mk(e, 'down_proj', inter, hidden) for e in range(E)])

    def _expert_chain(self, e, x):
        """Full bs=1 compute of expert e: gate_up -> chunk -> act(gate)*up -> down.
        Returns [1, hidden] fp32. Pure compute (no python-visible sync); used both
        eagerly and inside a per-expert CUDA-graph capture. The two linears run their
        plain bs=1 eager kernel path (NOT their own _graph_forward) -- capture of THIS
        fused graph forces _CAPTURE_ENABLED off so there is no illegal nested capture;
        the linear kernels are recorded directly into this graph instead."""
        prev = _bs._CAPTURE_ENABLED
        _bs._CAPTURE_ENABLED = False
        try:
            gate_up = self.gate_up[e]
            down = self.down[e]
            gu = gate_up(x.to(_mod_dtype(gate_up)))
            gate, up = gu.chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            return down(h.to(_mod_dtype(down)))
        finally:
            _bs._CAPTURE_ENABLED = prev

    def _expert_graphed(self, e, x):
        """Replay (capturing on first use) expert e's fused gate_up->act->down graph.
        One graph per expert *instance* (weights fixed) -> valid for any input vector,
        exactly like the per-linear graphs (proof: probe_cudagraph.py). Collapses the
        two per-linear replays + the python chunk/act/mul into a SINGLE graph replay,
        cutting per-token launch/python overhead. Byte-identical to _expert_chain."""
        cg = self._cg_expert[e]
        if cg is None:
            hidden = self.hidden_dim
            cin = torch.empty((1, hidden), dtype=x.dtype, device=x.device)
            cin.copy_(x.reshape(1, hidden))
            # warm up (build each linear's qtip fast handle, hadK cache, allocator)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._expert_chain(e, cin)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            cap = torch.cuda.Stream(); cap.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(cap):
                # same context-manager-free capture technique as bitshift._graph_forward
                # (avoid torch.cuda.graph()'s gc.collect + global-mode handshake hangs)
                g.capture_begin(_bs._graph_pool(), capture_error_mode="thread_local")
                cout = self._expert_chain(e, cin)
                g.capture_end()
            torch.cuda.current_stream().wait_stream(cap)
            self._cg_expert[e] = (g, cin, cout)
            cg = self._cg_expert[e]
        g, cin, cout = cg
        cin.copy_(x.reshape(1, self.hidden_dim))
        g.replay()
        return cout.clone()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # ---- capture-safe bs=1 decode fast path -----------------------------
        # At steady-state decode there is exactly ONE token and top_k selected
        # experts (top_k_index is [1, top_k]). The generic path below uses
        # one_hot / permute / nonzero / where / index_add_ + a python loop over
        # the (data-dependent, dynamic-shaped) hit set -- nonzero/where force a
        # device->host sync and produce dynamic shapes, so that path can NOT be
        # CUDA-graph-captured and also carries heavy per-token launch/python
        # overhead x48 layers. For the single-token case we instead loop over
        # the FIXED top_k positions directly, driving each selected expert's two
        # QuantizedLinear (which own their own per-instance bs=1 decode graphs).
        # No nonzero/where/one_hot/index_add -> no host sync, static control
        # flow, ~half the eager routing kernels. Correctness: identical math --
        # sum_k weight_k * down(act(gate)*up) for the single token.
        if _bs._CUDAGRAPH and top_k_index.numel() == self.top_k and hidden_states.shape[0] == 1:
            # One host sync for the whole layer to turn the 8 selected-expert ids
            # into python ints for ModuleList indexing (the generic path syncs per
            # hit expert anyway); the heavy per-linear compute is graph-replayed.
            idx = top_k_index.reshape(-1).tolist()  # [top_k] python ints
            w = top_k_weights.reshape(-1)           # [top_k] on device
            x = hidden_states.reshape(1, -1)        # [1, hidden]
            if getattr(self, "_cg_expert", None) is None:
                self._cg_expert = [None] * self.num_experts
            out = None
            for k in range(self.top_k):
                e = idx[k]
                # dead experts are plain bf16 nn.Linear (no per-instance qtip graph) ->
                # run them eagerly; quantized experts use the fused per-expert graph.
                if _bs._CAPTURE_ENABLED and isinstance(self.gate_up[e], QuantizedLinear) \
                        and isinstance(self.down[e], QuantizedLinear):
                    h = self._expert_graphed(e, x)
                else:
                    h = self._expert_chain(e, x)
                h = h.to(hidden_states.dtype) * w[k]
                out = h if out is None else out + h
            return out.reshape_as(hidden_states)
        # ---------------------------------------------------------------------
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = self.gate_up[expert_idx]
            down = self.down[expert_idx]
            # QuantizedLinear/fused kernel emits fp32; dead experts are bf16 nn.Linear.
            # Cast each linear's input to that module's own weight dtype so the
            # quantized <-> dead mix stays dtype-consistent.
            gate, up = gate_up(current_state.to(_mod_dtype(gate_up))).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = down(
                current_hidden_states.to(_mod_dtype(down)))
            current_hidden_states = current_hidden_states * top_k_weights[
                token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class Qwen3_5MoeQuantizedForConditionalGeneration(
        Qwen3_5MoeForConditionalGeneration):
    """Qwen3.5-MoE MM model with text body + routed experts swapped to QuantizedLinear."""

    def __init__(self, config):
        super().__init__(config)
        qp = getattr(config, "quip_params", None)
        if qp is None:
            return                                   # behaves as the vanilla model
        qp = dict(qp)
        skip = set(qp.get("skip_list") or [])
        K = qp["K"]
        dtype = _resolve_dtype(config)
        tc = config.text_config if hasattr(config, "text_config") else config

        # Dead routed experts (no collected factor) kept in fp16 nn.Linear so their
        # ORIGINAL bf16 weight loads (variant a). qp['dead_experts'] = [[idx, e, proj], ...].
        dead_by_layer = {}
        for entry in (qp.get("dead_experts") or []):
            li, e, proj = entry
            dead_by_layer.setdefault(int(li), set()).add((int(e), proj))

        layers = self.model.language_model.layers
        for idx, layer in enumerate(layers):
            # attention: full vs DeltaNet, mutually exclusive
            if hasattr(layer, "self_attn"):
                for attr, name in _FULL_ATTN.items():
                    if hasattr(layer.self_attn, attr) and f"{idx}_{name}" not in skip:
                        _swap(layer.self_attn, attr, qp, K, dtype)
            elif hasattr(layer, "linear_attn"):
                for attr, name in _DELTA_ATTN.items():
                    if hasattr(layer.linear_attn, attr) and f"{idx}_{name}" not in skip:
                        _swap(layer.linear_attn, attr, qp, K, dtype)
            # MoE block: shared_expert MLP = body (K); routed experts = QuantizedExperts.
            mlp = layer.mlp
            if hasattr(mlp, "shared_expert"):
                for attr, name in _SHARED_MLP.items():
                    if hasattr(mlp.shared_expert, attr) and f"{idx}_{name}" not in skip:
                        _swap(mlp.shared_expert, attr, qp, K, dtype)
            if hasattr(mlp, "experts"):
                mlp.experts = QuantizedExperts(tc, qp, dtype,
                                               dead=dead_by_layer.get(idx))
