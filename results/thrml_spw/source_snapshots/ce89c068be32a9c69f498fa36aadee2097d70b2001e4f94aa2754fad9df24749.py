"""Predeclared full frozen-CNN variance-matched Gaussian negative controls."""
import datetime
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np

from results.thrml_spw.fashion_cnn_native.gaussian_control.common import (
    ROOT, TRAIN, BASE, OUT, MODES, KEY_ROOT, sha, write, guard, weights_for, torch_forward)


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    completion = json.loads((BASE/'completion.json').read_text())
    for path, h in completion['artifact_sha256'].items():
        assert sha(path) == h, path
    assert json.loads((BASE/'native_audit.json').read_text())['passed']
    train = json.loads((TRAIN/'manifest.json').read_text())
    assert train['complete'] and len(train['rows']) == 5
    paths = [TRAIN/'manifest.json', TRAIN/'validation.npz', BASE/'completion.json', BASE/'statistics.json',
             BASE/'statistical_arrays.npz', BASE/'test_data.npz', BASE/'native_audit.json']
    paths += [Path(row['path']) for row in train['rows']]
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': {str(path): sha(path) for path in Path(__file__).parent.glob('*.py')},
         'input_sha256': {str(path): sha(path) for path in paths},
         'modes': MODES, 'models': 5, 'seeds': 10, 'samples': [64, 256, 1024],
         'images': 10000, 'bank_size': 8, 'banks': 1250, 'chunk': 16, 'case_count': 300,
         'key_root': KEY_ROOT, 'bootstrap_seed': 1687520, 'bootstrap_draws': 5000,
         'minimum_free_bytes': 400000000, 'output_bound_bytes': 280000000,
         'centering': 'Both controls use variance4r^2p(1-p)/T from the finite8-bit field commands used by the original native CNN. Mode0centers Gaussian noise on the original storedFP32weight; mode1centers it on the nominal programmed mean. No Gaussian clipping. Shared standard-normal variates across modes; new independent streams by model/T/seed/bank/layer. Nominal independent Bernoulli variance, not an empirically fitted device distribution.',
         'execution': 'One Gaussian perturbed kernel/head per8-imagebank/layer; shared across images and all spatial positions. Fresh bank/layer draws. DeterministicACT8 with original ranges, original deterministic biases, F64 convolution/readout/ReLU/maxpool. This is a conventional Gaussian negative control, not officialTHRML execution and not a physical latency/energy benchmark.',
         'statistics': 'All300cases/all3budgets/all5paired modelIDs/all10seeds. Five contrasts perbudget: each Gaussian centering minus original native, mode1minus mode0, each Gaussian minus FP32_F64. Parent5000model draws and native seed indices reused; fresh Gaussian bootstrap seed indices shared across modes. No image resampling; fixed previously consumed10000images, descriptive intervals only. No selection of a favorable method/budget, equivalence claim or retroactive gate pass.'}
    guard(p)
    write(OUT/'protocol.json', p)
    print('Gaussian negative-control full300case protocol frozen before test scoring', flush=True)


def run():
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    check = json.loads((OUT/'preflight.json').read_text())
    assert check['passed'] and check['protocol_sha256'] == sha(OUT/'protocol.json')
    assert not (OUT/'manifest.json').exists(), 'No duplicate negative-control production'
    assert shutil.disk_usage('.').free >= p['minimum_free_bytes']
    x = np.load(BASE/'test_data.npz')['inputs'].astype(float).reshape(1250, 8, 1, 14, 14)
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    (OUT/'cases').mkdir(exist_ok=True)
    manifest = {'complete': False, 'protocol_sha256': sha(OUT/'protocol.json'), 'rows': [], 'new_native_updates': 0}
    start = time.monotonic()
    with torch.no_grad():
        for mode in range(2):
            for mi in range(5):
                model = np.load(train[mi]['path'])
                for samples in p['samples']:
                    for seed in range(10):
                        guard(p)
                        assert shutil.disk_usage('.').free > 50000000
                        logits = np.empty((10000, 10), float)
                        clipping = np.zeros(3, np.int64)
                        for first in range(0, 1250, p['chunk']):
                            n = min(p['chunk'], 1250-first)
                            weights = [weights_for(model, mi, samples, seed, li, first, n, mode) for li in range(3)]
                            output, _, _, counts = torch_forward(x[first:first+n], model, weights)
                            logits[first*8:(first+n)*8] = output.reshape(n*8, 10)
                            clipping += counts
                        assert np.isfinite(logits).all()
                        name = f'mode{mode}_model{mi}_T{samples}_seed{seed}'
                        path = OUT/'cases'/(name+'.npz')
                        with path.open('wb') as stream:
                            np.savez_compressed(stream, logits=logits, clipped_coordinates=clipping)
                        manifest['rows'].append({'name': name, 'mode': mode, 'model': mi, 'T': samples, 'seed': seed,
                                                 'path': str(path), 'sha256': sha(path)})
                        write(OUT/'manifest.json', manifest)
                        print(json.dumps({'cases': len(manifest['rows']), 'last': name, 'elapsed_seconds': time.monotonic()-start}), flush=True)
    assert len(manifest['rows']) == 300
    guard(p)
    manifest['complete'] = True
    write(OUT/'manifest.json', manifest)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
