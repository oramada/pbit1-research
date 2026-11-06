"""Validation-only layer sensitivity; no test images or task scoring."""
import datetime
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
from results.thrml_spw.fashion_cnn_native.layer_allocation.common import (
    ROOT, TRAIN, OUT, KEY_ROOT, PREFIXES, COUNTS, sha, write, guard, weights_for, torch_forward, losses, solve)


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    train = json.loads((TRAIN/'manifest.json').read_text())
    assert train['complete'] and len(train['rows']) == 5
    paths = [TRAIN/'manifest.json', TRAIN/'validation.npz']+[Path(r['path']) for r in train['rows']]
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += [ROOT/'gaussian_control/common.py', ROOT/'gaussian_control/preflight.py']
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': {str(path): sha(path) for path in sources}, 'input_sha256': {str(path): sha(path) for path in paths},
         'models': 5, 'layers': PREFIXES, 'MC_seeds': 10, 'images': 5000, 'banks': 625, 'bank_size': 8,
         'chunk': 16, 'isolated_samples': 256, 'case_count': 150, 'key_root': KEY_ROOT,
         'minimum_free_bytes': 130000000, 'output_bound_bytes': 80000000,
         'allocation': 'Separate schedule for each of all5fixed starting checkpoints. S_l=max(mean validationCE with isolated-layer Gaussian noise minus nominal-mean validationCE,1e-4). Minimize sum_l S_l*256/T_l over every integer T_l in[64,1024] subject to 72T0+1152T1+1440T2=2664*256. Exact exhaustive search; lexicographic schedule tie-break for equal float objectives. No outcome-based switch of estimator, floor, bounds or objective.',
         'scope': 'Validation-only conventional Gaussian sensitivity surrogate, not nativeTHRML sampling, measured neural quality gain or an error bound. All5000training-validation images were used for earlier checkpoint selection. Noise preserves eight-image/spatial kernel sharing and original finite-field nominal moments, ACT8 and fixed metadata. Other layers remain nominal-mean deterministic. Fresh keys, no test data or labels in allocation. The selected schedules must be sealed and independently checked before any new full native test scoring. Uniform and allocated native test conditions must use equal scheduled updates; no favorable schedule may be selected on test. Existing test outcomes remain consumed descriptive evidence.'}
    guard(p)
    write(OUT/'protocol.json', p)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'manifest.json').exists()
    assert shutil.disk_usage('.').free >= p['minimum_free_bytes']
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dataset = np.load(TRAIN/'validation.npz')
    x, labels = dataset['inputs'].astype(float).reshape(625, 8, 1, 14, 14), dataset['labels']
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    (OUT/'cases').mkdir(exist_ok=True)
    manifest = {'complete': False, 'rows': [], 'baseline_rows': [], 'protocol_sha256': sha(OUT/'protocol.json'), 'new_native_updates': 0}
    with torch.no_grad():
        for mi in range(5):
            model = np.load(train[mi]['path'])
            for selected in [-1, 0, 1, 2]:
                for seed in ([-1] if selected == -1 else range(10)):
                    guard(p)
                    logits = np.empty((5000, 10))
                    for first in range(0, 625, 16):
                        n = min(16, 625-first)
                        weights = weights_for(model, mi, seed, selected, first, n)
                        output, _, _, _ = torch_forward(x[first:first+n], model, weights)
                        logits[first*8:(first+n)*8] = output.reshape(n*8, 10)
                    assert np.isfinite(logits).all()
                    name = f'model{mi}_layer{selected}_seed{seed}'
                    path = OUT/'cases'/(name+'.npz')
                    with path.open('wb') as stream:
                        np.savez_compressed(stream, logits=logits)
                    row = {'name': name, 'model': mi, 'layer': selected, 'seed': seed, 'path': str(path), 'sha256': sha(path), 'mean_CE': float(losses(logits, labels).mean())}
                    manifest['baseline_rows' if selected == -1 else 'rows'].append(row)
                    write(OUT/'manifest.json', manifest)
                    print(json.dumps({'isolated_cases': len(manifest['rows']), 'baselines': len(manifest['baseline_rows'])}), flush=True)
    assert len(manifest['rows']) == 150 and len(manifest['baseline_rows']) == 5
    manifest['complete'] = True
    guard(p)
    write(OUT/'manifest.json', manifest)


def select():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'allocation.json').exists()
    audit = json.loads((OUT/'evidence_audit.json').read_text())
    m = json.loads((OUT/'manifest.json').read_text())
    assert m['complete'] and audit['passed'] and audit['manifest_sha256'] == sha(OUT/'manifest.json')
    rows = []
    for mi in range(5):
        baseline = next(r['mean_CE'] for r in m['baseline_rows'] if r['model'] == mi)
        means = np.array([np.mean([r['mean_CE'] for r in m['rows'] if r['model'] == mi and r['layer'] == li]) for li in range(3)])
        coefficients = np.maximum(means-baseline, 1e-4)
        chosen, objective, feasible = solve(coefficients)
        rows.append({'model': mi, 'baseline_CE': baseline, 'isolated_mean_CE': means.tolist(), 'coefficients': coefficients.tolist(),
                     'samples': chosen.tolist(), 'surrogate': objective, 'uniform_surrogate': float(coefficients.sum()),
                     'feasible_schedules': feasible, 'scheduled_updates_per_bank': int(chosen@COUNTS)})
    write(OUT/'allocation.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'manifest_sha256': sha(OUT/'manifest.json'),
                                 'rows': rows, 'scope': p['scope'], 'native_test_scored': False})
    print('All5validation-only schedules frozen; separate optimum verification pending', flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'select': select}[sys.argv[1]]()
