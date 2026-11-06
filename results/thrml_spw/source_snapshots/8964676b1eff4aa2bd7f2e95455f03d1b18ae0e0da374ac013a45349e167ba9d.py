"""Independent word-mean law and validation graph checks before test execution."""
import json
import sys
from pathlib import Path
import numpy as np
from results.thrml_spw.cifar_cnn_native.word_cohort.common import ROOT, TRAIN, OUT, PARENT, sha, write, save, guard, parent_closed, mean_weights
from results.thrml_spw.fashion_cnn_native.cohort.common import numpy_control


def selfcheck():
    import torch
    from results.thrml_spw.fashion_cnn_native.word_comparison.verify import mathematical_mean, torch_control
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'control_selfcheck.json').exists()
    sources = dict(json.loads((ROOT/'word_witness_evidence/protocol.json').read_text())['source_sha256'])
    extra = [*Path(__file__).parent.glob('*.py'),
             Path('results/thrml_spw/fashion_cnn_native/word_comparison/common.py'),
             Path('results/thrml_spw/fashion_cnn_native/word_comparison/verify.py')]
    sources.update({str(p): sha(p) for p in extra})
    x = np.load(TRAIN/'validation.npz')['inputs'][:64]
    rows = []
    for mi, row in enumerate(json.loads((TRAIN/'manifest.json').read_text())['rows']):
        assert sha(row['path']) == row['sha256']
        model = np.load(row['path'])
        canonical, other = mean_weights(model), mathematical_mean(model)
        assert sum(w.size for w in canonical) == 3928
        for a, b in zip(canonical, other):
            np.testing.assert_allclose(a, b, rtol=2e-14, atol=2e-15)
        a, b = numpy_control(x, model, canonical, True), torch_control(x, model, canonical)
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-11)
        np.testing.assert_array_equal(a.argmax(1), b.argmax(1))
        rows.append({'model': mi, 'weight_coordinates_checked': 3928,
                     'logit_values_checked': 640, 'maximum_error': float(abs(a-b).max())})
    for path, expected in sources.items():
        assert sha(path) == expected, path
    write(OUT/'control_selfcheck.json', {'passed': True, 'rows': rows,
          'source_sha256': sources, 'validation_sha256': sha(TRAIN/'validation.npz'),
          'training_manifest_sha256': sha(TRAIN/'manifest.json'),
          'scope': 'All5RGBmodels/64validation inputs, independent bit-probability mean law and NumPy/Torch graphs. No task metrics or stochastic draws; nonlinear mean-weight plug-in is not E[finite-sample CNN].'})
    print('FiveRGBwordmean laws/19640weight coordinates/3200validation logits checked', flush=True)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert not (OUT/'controls.json').exists()
    x = np.load(PARENT/'test_data.npz')['inputs']
    logits, weights = {}, {}
    for mi, row in enumerate(json.loads((TRAIN/'manifest.json').read_text())['rows']):
        model = np.load(row['path'])
        ww = mean_weights(model)
        logits[f'model{mi}'] = np.concatenate([numpy_control(x[first:first+128], model, ww, True) for first in range(0, 10000, 128)])
        for li, w in enumerate(ww):
            weights[f'model{mi}_layer{li}'] = w
    save(OUT/'mean_logits.npz', **logits)
    save(OUT/'mean_weights.npz', **weights)
    guard(p)
    write(OUT/'controls.json', {'complete': True, 'protocol_sha256': sha(OUT/'protocol.json'),
          'mean_logits_sha256': sha(OUT/'mean_logits.npz'), 'mean_weights_sha256': sha(OUT/'mean_weights.npz'),
          'parent_test_data_sha256': sha(PARENT/'test_data.npz'),
          'scope': 'Nominal finite-field word mean nonlinear plug-in; not E[stochastic CNN]. Independent full numerical verification still required.'})


if __name__ == '__main__':
    {'selfcheck': selfcheck, 'run': run}[sys.argv[1]]()
