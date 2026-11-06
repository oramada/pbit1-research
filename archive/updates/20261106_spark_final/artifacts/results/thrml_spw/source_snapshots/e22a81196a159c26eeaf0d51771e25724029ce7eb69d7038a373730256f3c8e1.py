"""CPU model helpers; native runtime never imports Torch."""
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .sequential_transformer import SequentialRunner
from .calibrated_transformer_controls import decode
from .transformer_block_gpu import sha

def runner(p,mean=False):
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}";r=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),p['targets_in_forward_order'])
    if mean:set_means(r,p)
    return r

def set_means(r,p):
    with torch.no_grad():
        for module,c in zip(r.modules,p['control_artifacts']):
            assert sha(c['path'])==c['sha256'];a=np.load(c['path']);prob=1/(1+np.exp(-2*decode(a['codes']).astype(float)));w=(np.repeat(a['lower'],128,0)+np.repeat(a['span'],128,0)*prob).astype(np.float32);module.weight.copy_(torch.from_numpy(w))
