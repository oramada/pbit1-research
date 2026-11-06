"""Validation-only control witness, then frozen full deterministic baselines."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from results.thrml_spw.cifar_cnn_native.model import features
from results.thrml_spw.fashion_cnn_native.cohort.controls import torch_control
from results.thrml_spw.cifar_cnn_native.cohort.common import (
    ROOT, TRAIN, OUT, CONTROLS, sha, write, save, guard, control_weights, numpy_control, preparation_closed)


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=True)
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    if sys.argv[1] == 'selfcheck':
        assert not (OUT/'control_selfcheck.json').exists()
        preparation_closed()
        x = np.load(TRAIN/'validation.npz')['inputs'][:64]
        rows = []
        with torch.no_grad():
            for mi, row in enumerate(train):
                model = np.load(row['path'])
                for name in CONTROLS:
                    weights = control_weights(model, name)
                    actual = torch_control(x, model, weights, name)
                    independent = numpy_control(x, model, weights, name.startswith('ACT8'))
                    tol = (2e-5, 1e-4) if name == 'FP32_F32' else (2e-12, 2e-11)
                    np.testing.assert_allclose(actual, independent, rtol=tol[0], atol=tol[1])
                    np.testing.assert_array_equal(actual.argmax(1), independent.argmax(1))
                    rows.append({'model': mi, 'control': name, 'maximum_logit_error': float(np.max(abs(actual-independent)))})
        sources = [Path(__file__), ROOT/'model.py', Path('results/thrml_spw/fashion_cnn_native/cohort/controls.py'),
                   Path('results/thrml_spw/fashion_cnn_native/cohort/common.py')]
        write(OUT/'control_selfcheck.json', {'passed': True, 'rows': rows,
                                          'source_sha256': {str(p): sha(p) for p in sources},
                                          'scope': 'All50RGB control paths on first64validation images. NumPy spatial contractions versusTorch functional CNN; no tasklabels/test scoring.'})
        print('All50RGB control validation paths pass', flush=True)
        return
    assert sys.argv[1] == 'run'
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'controls.json').exists()
    data = CIFAR10('data/torchvision', train=False, download=False)
    x, labels = features(data.data).numpy(), np.asarray(data.targets, np.int64)
    assert x.shape == (10000, 3, 16, 16)
    save(OUT/'test_data.npz', inputs=x, labels=labels)
    arrays, rows = {}, []
    with torch.no_grad():
        for mi, row in enumerate(train):
            model = np.load(row['path'])
            for ci, name in enumerate(CONTROLS):
                weights = control_weights(model, name)
                logits = np.concatenate([torch_control(x[first:first+128], model, weights, name) for first in range(0, 10000, 128)])
                assert logits.shape == (10000, 10) and np.isfinite(logits).all()
                arrays[f'c{ci}_model{mi}'] = logits
                rows.append({'model': mi, 'control': name})
                print(json.dumps(rows[-1]), flush=True)
    save(OUT/'control_logits.npz', **arrays)
    guard(p)
    write(OUT/'controls.json', {'complete': True, 'protocol_sha256': sha(OUT/'protocol.json'), 'rows': rows,
                               'test_data_sha256': sha(OUT/'test_data.npz'), 'control_logits_sha256': sha(OUT/'control_logits.npz'),
                               'scope': 'All50full deterministic baseline paths, existing consumed CIFARtest corpus. Dequantized arithmetic, not optimized hardware.'})


if __name__ == '__main__':
    main()
