"""Frozen 512-token LFM3 controls; simulated affine weight-only quantization.

All learned tensors, including norms and tied embedding/head, are covered.
Dequantized BF16 execution is a quality control, not an integer-kernel benchmark.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'language_controls_batch'
    out.mkdir(exist_ok=True)
    selection = root / 'language_subset'
    manifest = json.loads((selection / 'manifest.json').read_text())
    assert sha(selection / 'tokens.npz') == manifest['tokens_sha256']
    assert sha(selection / 'protocol.json') == manifest['protocol_sha256']
    protocol = {
        'source_sha256': sha(Path(__file__)),
        'selection_manifest_sha256': sha(selection / 'manifest.json'),
        'tokens_sha256': manifest['tokens_sha256'],
        'revision': 'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7',
        'conditions': ['BF16', 'INT2_affine_g128', 'INT4_affine_g128', 'INT8_affine_g128'],
        'grouping': 'Per output row, consecutive groups of128 along fan-in; seven-weight depthwise kernels stay one group; norm vectors grouped128.',
        'quantizer': 'Min/max affine, levels=2**bits-1, nearest-even rounding. Constant groups exactly preserved. Dequantized values rounded to BF16 before original operators. No activation quantization.',
        'batching': 'All eight65-token contexts evaluated together; batch BF16 must exactly match independent saved batch preflight.',
        'scope': 'Exploratory frozen eight contexts /512 predicted tokens. All learned tensors quantized, tied embeddings/head counted once. BF16 arithmetic on dequantized weights; no integer-kernel timing, physical memory or equivalence claim.',
    }
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        'data/thrml_spw/lfm3_1p7b/checkpoint', trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager').to(device).eval()
    code = sys.modules[type(model.model.layers[0].conv).__module__]
    code.can_use_lfm3_conv = lambda *args: False
    originals = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    assert sum(p.numel() for p in originals.values()) == 1728801536
    tokens = np.load(selection / 'tokens.npz')['tokens']
    assert tokens.shape == (8, 65)
    metrics = []
    for condition in protocol['conditions']:
        bits = None if condition == 'BF16' else int(condition.split('_')[0][3:])
        inventory = []
        for name, p in model.named_parameters():
            original = originals[name].to(device)
            rows = original.shape[0] if original.ndim > 1 else 1
            fanin = original.numel() // rows
            size = min(128, fanin)
            assert fanin % size == 0
            groups = original.numel() // size
            if bits is not None:
                w = original.float().reshape(-1, size)
                lo = w.amin(1, keepdim=True)
                span = w.amax(1, keepdim=True) - lo
                safe = torch.where(span == 0, torch.ones_like(span), span)
                q = torch.round((w - lo) / safe * ((1 << bits) - 1))
                reconstructed = lo + span * q / ((1 << bits) - 1)
                p.data.copy_(reconstructed.reshape(original.shape).to(torch.bfloat16))
            else:
                p.data.copy_(original)
            inventory.append({'name': name, 'shape': list(p.shape),
                              'numel': p.numel(), 'groups': groups, 'group_size': size})
        assert model.lm_head.weight is model.model.embed_tokens.weight
        logits_path = out / (condition + '_logits.npy')
        logits = np.lib.format.open_memmap(logits_path, mode='w+', dtype=np.float32,
                                          shape=(8, 64, 128000))
        losses = []
        start = time.monotonic()
        with torch.inference_mode():
            ids = torch.from_numpy(tokens).to(device)
            batch_y = model(input_ids=ids, use_cache=False).logits[:, :-1].float().cpu().numpy()
            if condition == 'BF16':
                np.testing.assert_array_equal(batch_y, np.load(root / 'language_batch_preflight/logits.npy')[:, :-1])
            for window, tok in enumerate(tokens):
                y = batch_y[window]
                assert np.isfinite(y).all()
                logits[window] = y
                z = y.astype(np.float64)
                mx = z.max(-1)
                nll = mx + np.log(np.exp(z - mx[:, None]).sum(-1)) - z[np.arange(64), tok[1:]]
                losses.append(nll.tolist())
                (out / 'progress.json').write_text(json.dumps({'condition': condition, 'completed_windows': window + 1}) + '\n')
        del batch_y
        logits.flush()
        del logits
        ce = float(np.mean(losses))
        metric = {'condition': condition, 'mean_NLL_FP64': ce,
                  'subset_perplexity': float(np.exp(ce)), 'token_NLL': losses,
                  'predicted_tokens': 512, 'logits_sha256': sha(logits_path),
                  'elapsed_seconds': time.monotonic() - start,
                  'parameters': sum(p['numel'] for p in inventory),
                  'groups': sum(p['groups'] for p in inventory),
                  'bits': bits, 'inventory': inventory}
        with (out / (condition + '_metrics.json')).open('x') as f:
            json.dump(metric, f, indent=2)
        metrics.append(metric)
        print(json.dumps({k: metric[k] for k in ['condition', 'mean_NLL_FP64', 'subset_perplexity', 'parameters']}), flush=True)
    with (out / 'results.json').open('x') as f:
        json.dump({'protocol_sha256': sha(out / 'protocol.json'), 'metrics': metrics}, f, indent=2)


if __name__ == '__main__':
    main()
