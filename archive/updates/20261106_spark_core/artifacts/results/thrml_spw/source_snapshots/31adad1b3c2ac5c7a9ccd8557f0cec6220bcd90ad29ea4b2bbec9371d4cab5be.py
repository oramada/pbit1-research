"""Independent saved-logit loss, token selection and tensor-coverage audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'language_controls_batch'
    protocol = json.loads((out / 'protocol.json').read_text())
    results = json.loads((out / 'results.json').read_text())
    assert sha(out / 'protocol.json') == results['protocol_sha256']
    assert sha(Path(__file__).with_name('language_controls_batch.py')) == protocol['source_sha256']
    selection = root / 'language_subset'
    assert sha(selection / 'manifest.json') == protocol['selection_manifest_sha256']
    assert sha(selection / 'tokens.npz') == protocol['tokens_sha256']
    tokens = np.load(selection / 'tokens.npz')['tokens']
    meta = json.loads((root / 'meta_inventory.json').read_text())
    expected = {p['name']: p for p in meta['parameters']}
    assert [m['condition'] for m in results['metrics']] == protocol['conditions']
    checks = []
    for metric in results['metrics']:
        condition = metric['condition']
        path = out / (condition + '_logits.npy')
        assert sha(path) == metric['logits_sha256']
        logits = np.load(path, mmap_mode='r')
        assert logits.shape == (8, 64, 128000) and logits.dtype == np.float32
        if condition == 'BF16':
            np.testing.assert_array_equal(logits, np.load(root / 'language_batch_preflight/logits.npy')[:, :-1])
        losses = []
        for window in range(8):
            assert np.isfinite(logits[window]).all()
            nll = F.cross_entropy(torch.from_numpy(logits[window].astype(np.float64)),
                                  torch.from_numpy(tokens[window, 1:]), reduction='none').numpy()
            np.testing.assert_allclose(nll, metric['token_NLL'][window], atol=1e-12, rtol=1e-12)
            losses.extend(nll.tolist())
        ce = float(np.mean(losses))
        np.testing.assert_allclose(ce, metric['mean_NLL_FP64'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(np.exp(ce), metric['subset_perplexity'], atol=1e-12, rtol=1e-12)
        inventory = {p['name']: p for p in metric['inventory']}
        assert set(inventory) == set(expected)
        groups = 0
        for name, p in inventory.items():
            assert p['shape'] == expected[name]['shape'] and p['numel'] == expected[name]['numel']
            fanin = int(np.prod(p['shape'][1:])) if len(p['shape']) > 1 else p['numel']
            size = min(128, fanin)
            assert fanin % size == 0 and p['group_size'] == size
            assert p['groups'] == p['numel'] // size
            groups += p['groups']
        assert sum(p['numel'] for p in inventory.values()) == metric['parameters'] == 1728801536
        assert groups == metric['groups'] == 13531430
        assert metric['predicted_tokens'] == len(losses) == 512
        checks.append({'condition': condition, 'mean_NLL_FP64': ce,
                       'subset_perplexity': float(np.exp(ce)), 'all_unique_parameters': metric['parameters'],
                       'logits_sha256': metric['logits_sha256']})
    audit = {'status': 'all_saved_token_losses_and_unique_tensor_coverage_verified',
             'batch_BF16_matches_independent_preflight_exact': True,
             'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
             'checks': checks, 'scope': 'Saved-logit numerical and inventory audit. Quantizer implementation inspected in source; no independent full quantized-model rerun. Small exploratory subset, not standard WikiText perplexity or equivalence proof.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
