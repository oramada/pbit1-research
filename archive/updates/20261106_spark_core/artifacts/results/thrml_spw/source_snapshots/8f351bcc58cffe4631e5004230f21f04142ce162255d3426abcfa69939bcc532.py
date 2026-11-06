"""Full numerical replay using NumPy rather than Torch grouped convolutions."""
import datetime
import json
import time
import numpy as np
from results.thrml_spw.fashion_cnn_native.gaussian_control.common import (
    TRAIN, BASE, OUT, sha, write, guard, weights_for, metrics)
from results.thrml_spw.fashion_cnn_native.gaussian_control.preflight import numpy_forward


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'numerical_audit.json').exists()
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['complete'] and manifest['protocol_sha256'] == sha(OUT/'protocol.json')
    expected = {(mode, mi, t, seed) for mode in range(2) for mi in range(5) for t in p['samples'] for seed in range(10)}
    assert len(manifest['rows']) == 300 and {(r['mode'], r['model'], r['T'], r['seed']) for r in manifest['rows']} == expected
    x = np.load(BASE/'test_data.npz')['inputs'].astype(float).reshape(1250, 8, 1, 14, 14)
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    rows = []
    start = time.monotonic()
    for row in manifest['rows']:
        guard(p)
        assert sha(row['path']) == row['sha256']
        saved = np.load(row['path'])
        assert saved['logits'].shape == (10000, 10) and saved['logits'].dtype == np.float64
        model = np.load(train[row['model']]['path'])
        clipping, worst = np.zeros(3, np.int64), 0.
        for first in range(0, 1250, 16):
            count = min(16, 1250-first)
            weights = [weights_for(model, row['model'], row['T'], row['seed'], li, first, count, row['mode']) for li in range(3)]
            logits, _, _, clip = numpy_forward(x[first:first+count], model, weights)
            retained = saved['logits'][first*8:(first+count)*8]
            actual = logits.reshape(count*8, 10)
            np.testing.assert_allclose(actual, retained, rtol=2e-12, atol=2e-11)
            np.testing.assert_array_equal(actual.argmax(1), retained.argmax(1))
            clipping += clip
            worst = max(worst, float(np.max(abs(actual-retained))))
        np.testing.assert_array_equal(clipping, saved['clipped_coordinates'])
        rows.append({'name': row['name'], 'maximum_logit_error': worst, 'all_predictions_exact': True})
        write(OUT/'replay_progress.json', {'rows': rows, 'protocol_sha256': sha(OUT/'protocol.json')})
        print(json.dumps({'replayed': len(rows), 'elapsed_seconds': time.monotonic()-start}), flush=True)
    guard(p)
    write(OUT/'numerical_audit.json', {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                     'manifest_sha256': sha(OUT/'manifest.json'), 'rows': rows,
                                     'logit_values_checked': 30000000, 'new_native_updates': 0,
                                     'scope': 'All300Gaussian cases/full10000inputs: separateNumPy spatial-window contractions versusTorch groupedconvolutions. Same specified Gaussian weights/RNG helper intentionally reused; not independent RNG or native state replay. No physical inference.'})


if __name__ == '__main__':
    main()
