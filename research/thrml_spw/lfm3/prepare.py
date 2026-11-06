"""Pinned LFM3 architecture inventory and deterministic baseline smoke run.

This preparation does not claim probabilistic conversion or task-quality evidence.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = Path('data/thrml_spw/lfm3_1p7b/checkpoint')
OUT = Path('results/thrml_spw/lfm3_1p7b')
REVISION = 'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261029)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True, local_files_only=True)
    with torch.device('meta'):
        skeleton = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    params = [{'name': n, 'shape': list(p.shape), 'numel': p.numel()}
              for n, p in skeleton.named_parameters()]
    count = sum(p['numel'] for p in params)
    inventory = {
        'checkpoint': 'LiquidAI/LFM3-1.7B', 'revision': REVISION,
        'source_sha256': sha(Path(__file__)),
        'checkpoint_files': {p.name: sha(p) for p in CHECKPOINT.iterdir() if p.is_file()},
        'unique_parameters': count, 'parameters': params,
        'logical_binary_state_bytes': (count + 7) // 8,
        'FP32_probability_bytes': 4 * count,
        'BF16_weight_bytes': 2 * count,
        'retained_node_evaluations_per_full_sweep': {str(t): count*t for t in [1,16,64,256,1024,4096]},
        'scope': 'Architecture inventory only; binary states exclude programming controls, metadata, activations, counters and simulator object overhead. Tied embeddings counted once. No stochastic inference yet.',
    }
    (OUT/'preparation_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    del skeleton
    print('architecture inventory complete:', count, 'unique parameters', flush=True)
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager').to(device).eval()
    prompt = 'A probability encodes a number between zero and one. The capital of France is'
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits.float().cpu().numpy()
    assert np.isfinite(logits).all()
    np.savez_compressed(OUT/'baseline_smoke.npz', input_ids=inputs['input_ids'].cpu().numpy(), logits=logits)
    result = {'revision': REVISION, 'dtype':'BF16', 'device': device,
              'prompt': prompt, 'input_tokens': inputs['input_ids'].shape[-1],
              'logits_shape': list(logits.shape), 'all_finite': True,
              'elapsed_seconds': time.monotonic()-start,
              'artifact_sha256':sha(OUT/'baseline_smoke.npz'),
              'scope':'Deterministic loading/forward smoke only. No perplexity, quality or p-bit equivalence claim.'}
    (OUT/'baseline_smoke.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
