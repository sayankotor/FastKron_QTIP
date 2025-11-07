# functions in this file cause circular imports so they cannot be loaded into __init__

import json
import os

import accelerate
import torch
import transformers

from model.llama import LlamaForCausalLM
from model.qwen import Qwen3ForCausalLM
#from model.llama4 import Llama4ForCausalLM
#from model.llama4_orig import Llama4ForCausalLM as Llama4ForCausalLMOrig

def _normalize_config(cfg):
    # --- гарантируем наличие quip_params ---
    if not hasattr(cfg, "quip_params") or cfg.quip_params is None:
        cfg.quip_params = {}
    cfg.quip_params.setdefault("fused", True)
    cfg.quip_params.setdefault("skip_list", [])

    # --- sliding_window и layer_types ---
    if not hasattr(cfg, "sliding_window"):
        cfg.sliding_window = None

    lt = getattr(cfg, "layer_types", None)
    if lt is None:
        default_type = "sliding_attention" if cfg.sliding_window else "full_attention"
        lt = [default_type] * cfg.num_hidden_layers
    else:
        lt = list(lt)
        if len(lt) < cfg.num_hidden_layers:
            lt += [lt[-1]] * (cfg.num_hidden_layers - len(lt))
        elif len(lt) > cfg.num_hidden_layers:
            lt = lt[:cfg.num_hidden_layers]
    cfg.layer_types = lt

    return cfg



def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):

    # AutoConfig fails to read name_or_path correctly
    if 'qwen3' in path:
        bad_config = torch.load('../yaqa-quantization/qwen3_sketchA_2048_qw_2_vika/config.pt', weights_only=False)
        model_config = bad_config['model_config']
        is_quantized = hasattr(model_config, 'quip_params')
        model_type = model_config.model_type
        model_config = _normalize_config(model_config)

    else:
        bad_config = transformers.AutoConfig.from_pretrained(path)
        is_quantized = hasattr(bad_config, 'quip_params')
        model_type = bad_config.model_type

    if is_quantized:
        if model_type == 'llama':
            model_str = transformers.LlamaConfig.from_pretrained(
                path)._name_or_path
            model_cls = LlamaForCausalLM
        elif model_type.startswith('llama4'):
            model_str = transformers.LlamaConfig.from_pretrained(
                path)._name_or_path
            model_cls = Llama4ForCausalLM
        elif model_type.startswith('qwen3'):
            model_str = "Qwen3ForCausalLM"#transformers.Qwen3Config.from_pretrained(
                #path)._name_or_path
            model_cls = Qwen3ForCausalLM
        else:
            raise Exception
    else:
        if model_type.startswith('llama4'):
            model_str = transformers.LlamaConfig.from_pretrained(
                path)._name_or_path
            model_cls = Llama4ForCausalLMOrig
        else:
            model_cls = transformers.AutoModelForCausalLM
            model_str = path

    print(model_cls)
    if device_map is None:
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        if model_type.startswith('qwen3'):
            model = model_cls.from_pretrained(path,
                                                  torch_dtype='auto',
                                                  low_cpu_mem_usage=True,
                                                  trust_remote_code=True,
                                                  config=model_config)
            print ("loadad qwen!")
        else:
            model = model_cls.from_pretrained(path,
                                          torch_dtype='auto',
                                          low_cpu_mem_usage=True,
                                          attn_implementation='sdpa')
        device_map = accelerate.infer_auto_device_map(
            model,
            no_split_module_classes=[
                'LlamaDecoderLayer', 'Llama4TextDecoderLayer'
            ],
            max_memory=mmap)
    if model_type.startswith('qwen3'):
        model = model_cls.from_pretrained(path,
                                              torch_dtype='auto',
                                              low_cpu_mem_usage=True,
                                              trust_remote_code=True,
                                              config=model_config,
                                              device_map=device_map)
        print ("loadad qwen!")

    else:
        model = model_cls.from_pretrained(path,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      attn_implementation='sdpa',
                                      device_map=device_map)

    return model, model_str
