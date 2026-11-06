"""Independent int64 reconstruction of T8192 counts, padding and sign mirrors."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path('results/thrml_spw/lfm3_1p7b/sample8192_preflight')
    p = json.loads((out / 'protocol.json').read_text())
    r = json.loads((out / 'results.json').read_text())
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert r['status'] == 'both_T8192_signed_counter_edge_cases_publicly_replayed'
    assert r['worker_ready']['ready'] and any('cuda' in v.lower() or 'gpu' in v.lower() for v in r['worker_ready']['devices'])
    assert p['counter_bound'] == 2146435072 < np.iinfo(np.int32).max
    assert len(r['cases']) == 2
    previous_states = previous_total = previous_output = None
    extrema = []
    for rec, sign in zip(r['cases'], [1, -1], strict=True):
        for key in ['input', 'output', 'raw']:
            assert sha(Path(rec[key])) == rec[key + '_sha256']
        source, raw = np.load(rec['input']), np.load(rec['raw'])
        assert set(source.files) == {'x', 'codes', 'lower', 'span'}
        np.testing.assert_array_equal(source['x'], np.full((1, 128), sign, np.float32))
        np.testing.assert_array_equal(source['codes'], np.full((128, 8), 4095, np.uint16))
        np.testing.assert_array_equal(source['lower'], np.zeros((1, 8), np.float32))
        np.testing.assert_array_equal(source['span'], np.ones((1, 8), np.float32))
        codes = np.full((1, 128, 64), 2048, np.uint16); codes[:, :, :8] = 4095
        np.testing.assert_array_equal(raw['codes'], codes)
        integers = np.full((1, 128), sign * 2047, np.int64)
        np.testing.assert_array_equal(raw['inputs'], integers)
        states = raw['states']
        assert states.shape == (8192, 1, 128, 64) and states.dtype == np.bool_
        counts = states.sum(0, dtype=np.int64)
        expected = np.einsum('bgk,gko->bgo', integers.reshape(1, 1, 128), counts)
        assert raw['total'].dtype == np.int32
        np.testing.assert_array_equal(raw['total'].astype(np.int64), expected)
        assert np.max(np.abs(expected)) <= p['counter_bound']
        assert np.min(np.abs(expected[:, :, :8])) > 2 ** 30
        output = expected[:, 0, :8].astype(np.float64) / 8192 / 2047
        np.testing.assert_allclose(output, np.load(rec['output']), rtol=2e-12, atol=2e-12)
        assert rec['weight_coordinates'] == 1024 and rec['padding_nodes'] == 7168
        assert rec['retained_evaluations'] == 67108864 and rec['warmup_evaluations'] == 8192
        assert rec['program_codes_used_directly'] and not rec['reference_weights_supplied']
        if previous_states is not None:
            np.testing.assert_array_equal(states, previous_states)
            np.testing.assert_array_equal(expected, -previous_total)
            np.testing.assert_array_equal(np.load(rec['output']), -previous_output)
        previous_states, previous_total, previous_output = states, expected, np.load(rec['output'])
        extrema.append({'condition': rec['condition'], 'minimum_counter': int(expected.min()), 'maximum_counter': int(expected.max())})
    audit = {'status': 'all_T8192_public_counts_high_magnitude_int32_readouts_and_sign_mirrors_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
        'public_binary_observations': 134217728, 'exact_integer_counter_entries': 128,
        'theoretical_max_abs_counter': p['counter_bound'], 'observed_counter_extrema': extrema,
        'scope': 'Two synthetic signed ACT12conditions nearint32limit, independentint64reconstruction andexactsame-state signsymmetry. DirectGPU THRMLworker at8192retainedsamples. No fullneuralallocation quality/BF16orhardwareperformanceclaim.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
