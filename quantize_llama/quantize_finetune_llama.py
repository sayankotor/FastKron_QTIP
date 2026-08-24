import argparse
import os
import time

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
import sys
_FASTKRON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FASTKRON_ROOT not in sys.path:
    sys.path.insert(0, _FASTKRON_ROOT)

from operator import attrgetter

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils
from lib.algo import finetune
from lib.codebook import bitshift

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--hess_path', type=str)
parser.add_argument('--in_hess_path', type=str, default=None)
parser.add_argument('--out_hess_path', type=str, default=None)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=3e-6, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=1, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=5, type=int)
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--td_x', default=16, type=int)
parser.add_argument('--td_y', default=16, type=int)
parser.add_argument('--L', default=16, type=int)
parser.add_argument('--K', default=2, type=int)
parser.add_argument('--V', default=2, type=int)
parser.add_argument('--tlut_bits', default=0, type=int)
parser.add_argument('--decode_mode', default='lut', type=str)
parser.add_argument('--ft_train_lut', action='store_true')
parser.add_argument('--split_for_tp', action='store_true')
parser.add_argument('--tp_rank', default=8, type=int)
parser.add_argument('--skip_list', default=None, type=str)
# Mixed-precision MoE: body (attn + shared_expert) uses --K; the 256 routed
# experts use --expert_bits. expert_bits<=0 → no special expert handling (dense).
parser.add_argument('--expert_bits', default=0, type=int,
                    help='K (bits) for MoE routed experts; 0 = dense / no experts')
parser.add_argument('--num_experts', default=256, type=int,
                    help='routed experts per MoE layer')
parser.add_argument('--skip_experts', action='store_true',
                    help='Body-only: quantize attn + shared_expert (+layernorm) and '
                         'SKIP the routed-expert phase. Do experts separately via '
                         'quantize_experts_second_wave.py. Required when some experts '
                         'are dead (no factor) — the in-process expert phase opens the '
                         'factor unconditionally and would crash on the first dead one.')


def check_exist(idx, args, names, skip_list):
    # names = per-layer save-name list (depends on layer type) + layernorm
    for nm in list(names) + ['layernorm']:
        test = f'{args.save_path}/{idx}_{nm}.pt'
        if not (os.path.exists(test) or f'{idx}_{nm}' in skip_list):
            return False
    return True


def quantize_llama_decoder(layer, idx, cb, args, device, pre_orig_emb,
                           orig_emb, model_config, skip_list,
                           position_embeddings, cb_expert=None):

    if skip_list is None:
        skip_list = []

    # Attention projections depend on the layer type (Qwen3.5 hybrid LLLF):
    #   full_attention            -> self_attn.{q,k,v,o}_proj
    #   linear_attention (DeltaNet) -> linear_attn.{in_proj_qkv,in_proj_z,out_proj}
    # tuple = (linear_attr, save_name, in_hess, out_hess, rcp, skip_ft, one_sided)
    if hasattr(layer, 'self_attn'):
        attn_things = [
            ('self_attn.v_proj', 'v', 'v', 'v', 'col', False, False),
            ('self_attn.q_proj', 'q', 'q', 'q', 'col', False, False),
            ('self_attn.k_proj', 'k', 'k', 'k', 'col', False, False),
            ('self_attn.o_proj', 'o', 'o', 'o', 'row', False, False),
        ]
    elif hasattr(layer, 'linear_attn'):
        attn_things = [
            ('linear_attn.in_proj_qkv', 'in_proj_qkv', 'in_proj_qkv', 'in_proj_qkv', 'col', False, False),
            ('linear_attn.in_proj_z', 'in_proj_z', 'in_proj_z', 'in_proj_z', 'col', False, False),
            ('linear_attn.out_proj', 'out_proj', 'out_proj', 'out_proj', 'row', False, False),
        ]
    else:
        raise ValueError(f'layer {idx}: neither self_attn nor linear_attn found')

    # MoE layer (Qwen3.5-MoE): body MLP = shared expert (router mlp.gate stays
    # fp16); the 256 routed experts are quantized separately at expert_bits.
    is_moe = hasattr(layer.mlp, 'experts')
    if is_moe:
        mlp_things = [
            ('mlp.shared_expert.up_proj', 'shared_up', 'shared_up', 'shared_up', 'col', False, False),
            ('mlp.shared_expert.gate_proj', 'shared_gate', 'shared_gate', 'shared_gate', 'col', False, False),
            ('mlp.shared_expert.down_proj', 'shared_down', 'shared_down', 'shared_down', 'row', False, False),
        ]
    else:
        mlp_things = [
            ('mlp.up_proj', 'up', 'up', 'up', 'col', False, False),
            ('mlp.gate_proj', 'gate', 'gate', 'gate', 'col', False, False),
            ('mlp.down_proj', 'down', 'down', 'down', 'row', False, False),
        ]
    all_things = attn_things + mlp_things

    body_names = [t[1] for t in all_things]
    expert_names = []
    if is_moe and not args.skip_experts:
        expert_names = [f'experts_{e}_{p}'
                        for e in range(args.num_experts)
                        for p in ('gate_up_proj', 'down_proj')]
    if check_exist(idx, args, body_names + expert_names, skip_list):
        return

    quant_order = []
    for thing in all_things:
        if f'{idx}_{thing[1]}' not in skip_list:
            quant_order.append(thing)
        else:
            attrgetter(thing[0])(layer).weight.requires_grad = False
            print(f'skipping {idx}_{thing[1]}')

    finetune.quantize_finetune_decoder_layer(layer, quant_order, idx, cb, args,
                                             device, pre_orig_emb, orig_emb,
                                             position_embeddings)
    if is_moe and not args.skip_experts:
        # routed experts at expert_bits (mixed precision)
        finetune.quantize_moe_experts(layer, idx, cb_expert, args, device,
                                      args.num_experts)
    torch.save(
        {
            'input_layernorm': layer.input_layernorm.weight,
            'post_attention_layernorm': layer.post_attention_layernorm.weight,
        }, f'{args.save_path}/{idx}_layernorm.pt')


def main(args):
    if args.skip_list is not None:
        args.skip_list = args.skip_list.split(',')

    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    print ("args.tlut_bits", args.tlut_bits)

    cb = bitshift.bitshift_codebook(L=args.L,
                                    K=args.K,
                                    V=args.V,
                                    tlut_bits=args.tlut_bits,
                                    decode_mode=args.decode_mode)
    # Mixed-precision MoE: separate codebook for the routed experts at expert_bits.
    cb_expert = None
    if args.expert_bits and args.expert_bits > 0:
        cb_expert = bitshift.bitshift_codebook(L=args.L,
                                               K=args.expert_bits,
                                               V=args.V,
                                               tlut_bits=args.tlut_bits,
                                               decode_mode=args.decode_mode)
        glog.info(f'mixed precision: body K={args.K}, experts K={args.expert_bits}')
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True)

    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'L': args.L,
        'K': args.K,
        'V': args.V,
        'tlut_bits': args.tlut_bits,
        'decode_mode': args.decode_mode,
        'td_x': args.td_x,
        'td_y': args.td_y,
        'split_for_tp': args.split_for_tp,
        'skip_list': args.skip_list,
        'expert_bits': args.expert_bits,
        'num_experts': args.num_experts,
    }
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    glog.info('loaded model')

    # Pure PTQ from precomputed Kronecker factors — NO calibration dataset.
    # With ft_epochs=0 there is no per-layer finetune, so layer input/output
    # activations are unnecessary: LDLQ uses only each W plus the loaded XF/YF
    # factors. We therefore do NOT sample any dataset and do NOT forward the model
    # through its layers (that forward was also the rotary_emb crash on the hybrid
    # model). Each layer is quantized independently from its factor files.
    nproc = torch.cuda.device_count()
    cur_device = 0
    proc_list = [None for _ in range(nproc)]
    for i in range(len(model.model.layers)):
        glog.info(f'quantizing layer {i} on gpu {cur_device}')
        if proc_list[cur_device] is not None:
            proc_list[cur_device][0].join()
            model.model.layers[proc_list[cur_device][1]] = None
            utils.clean()
        proc_list[cur_device] = (mp.Process(
            target=quantize_llama_decoder,
            args=(
                model.model.layers[i],
                i,
                cb,
                args,
                cur_device,
                None,   # pre_orig_emb — unused (no finetune)
                None,   # orig_emb     — unused (no finetune)
                all_config['model_config'],
                args.skip_list,
                None,   # position_embeddings — unused (no finetune)
                cb_expert,  # K=expert_bits codebook for MoE routed experts
            )), i)
        proc_list[cur_device][0].start()
        cur_device = (cur_device + 1) % nproc

    for p in proc_list:
        if p is not None:
            p[0].join()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
