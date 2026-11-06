"""Independent audit of count-derived finite-controller offset compensation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha

OUT = Path('results/thrml_spw/calibrated_operator')


def raw_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p = json.loads((OUT/'protocol.json').read_text()); report = json.loads((OUT/'REPORT.json').read_text())
    run = json.loads((OUT/'manifest.json').read_text())
    assert run['config'] == p and run['result'] == report
    assert run['source_commit'] == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for path, digest in p['source_sha256'].items(): assert sha(path) == digest
    assert p['sigma_b'] == .2 and p['seeds'] == list(range(10)) and p['calibration_samples'] == [256, 1024, 4096]
    assert p['inference_samples'] == 512 and p['probe_code'] == 128 and p['controller_bits'] == 8
    for device in ['cpu', 'gpu']: assert 'OK' in (OUT/f'counts_{device}_test.log').read_text()
    artifact = Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
    old = Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')
    xp, rp = old/'stage11_inputs.npy', old/'stage11_conditional_reference.npy'
    for path, key in [(artifact, 'artifact_sha256'), (xp, 'inputs_sha256'), (rp, 'reference_sha256')]: assert sha(path) == p[key]
    x, ref = np.load(xp)[:8].astype(np.float64), np.load(rp)[:8].astype(np.float64)
    with np.load(artifact) as a:
        header = json.loads(a['header'].tobytes()); shape = tuple(header['shape']); n = int(np.prod(shape))
        bits = np.unpackbits(a['modes'], bitorder='little').reshape(-1, 2)[:n]
        modes = bits@np.array([1, 2], dtype=np.uint8); free = np.flatnonzero(modes == 0)
        code0 = a['codes'].copy(); field0 = (-4+code0.astype(np.float64)*8/255).astype(np.float32)
        lower = np.repeat(a['lower'].astype(np.float64), 128, 0); span = np.repeat(a['span'].astype(np.float64), 128, 0); bias = a['bias'].copy()
    assert shape == (3072, 768)
    nominal_q = (modes == 2).astype(np.float64); nominal_q[free] = 1/(1+np.exp(-2*field0.astype(np.float64)))
    nominal = x@(lower+span*nominal_q.reshape(shape))+bias
    np.testing.assert_allclose(np.mean((nominal-ref)**2), report['nominal_control_bias_floor'], rtol=1e-10)
    with (OUT/'metrics.csv').open() as f: rows = list(csv.DictReader(f))
    methods = ['uncalibrated', 'oracle_offset', 'measured_256', 'measured_1024', 'measured_4096']
    assert len(rows) == 50 and {(int(r['seed']), r['method']) for r in rows} == {(s, m) for s in range(10) for m in methods}
    assert len(list((OUT/'cases').glob('*.json'))) == len(list((OUT/'cases').glob('*.npz'))) == 50
    probe = float(np.float32(-4+128*8/255)); assert probe == report['probe_field']
    checked_counts = 0
    for seed in range(10):
        offset = .2*np.random.default_rng(2026102700+seed).standard_normal((4, len(free)))[0]
        probe_field = (offset+probe).astype(np.float32)
        probe_q = 1/(1+np.exp(-2*probe_field.astype(np.float64)))
        for method in methods:
            row = next(r for r in rows if int(r['seed']) == seed and r['method'] == method)
            prefix = OUT/'cases'/f'seed{seed}_{method}'
            case = json.loads(prefix.with_suffix('.json').read_text())
            for key, value in case.items(): assert ('' if value is None else str(value)) == row[key]
            assert row['offset_sha256'] == raw_sha(offset) and sha(prefix.with_suffix('.npz')) == row['archive_sha256']
            data = np.load(prefix.with_suffix('.npz')); codes, y = data['codes'], data['output'].astype(np.float64)
            assert codes.dtype == np.uint8 and codes.shape == code0.shape and y.shape == ref.shape and np.isfinite(y).all()
            ncal = int(method.split('_')[1]) if method.startswith('measured_') else 0
            assert int(row['calibration_samples']) == ncal
            if ncal:
                counts = data['counts']; assert counts.dtype == np.int32 and counts.shape == code0.shape
                assert np.all(counts >= 0) and np.all(counts <= ncal)
                # Algebraically independent form of the producer's logit formula.
                estimate = .5*(np.log(counts.astype(float)+.5)-np.log(ncal-counts.astype(float)+.5))-probe
                requested = field0.astype(np.float64)-estimate
                np.testing.assert_allclose(float(row['estimated_offset_rmse']), np.sqrt(np.mean((estimate-offset)**2)), rtol=1e-12)
                np.testing.assert_allclose(float(row['calibration_frequency_mse']), np.mean((counts/ncal-probe_q)**2), rtol=1e-12)
                np.testing.assert_allclose(float(row['calibration_predicted_frequency_mse']), np.mean(probe_q*(1-probe_q))/ncal, rtol=1e-12)
                checked_counts += 1
            else:
                assert 'counts' not in data.files and row['estimated_offset_rmse'] == ''
                requested = field0.astype(np.float64)-offset if method == 'oracle_offset' else field0.astype(np.float64)
            wanted = np.rint((np.clip(requested, -4, 4)+4)*255/8).astype(np.uint8)
            np.testing.assert_array_equal(codes, wanted)
            assert raw_sha(codes) == row['command_sha256']
            realized = ((-4+codes.astype(np.float64)*8/255).astype(np.float32).astype(np.float64)+offset).astype(np.float32)
            assert raw_sha(realized) == row['realized_field_sha256']
            q = (modes == 2).astype(np.float64); q[free] = 1/(1+np.exp(-2*realized.astype(np.float64))); q = q.reshape(shape)
            mu = x@(lower+span*q)+bias
            floor = float(np.mean((mu-ref)**2)); variance = float(np.mean(np.mean(x*x, axis=(0, 1))@(span*span*(q-q*q)))/512)
            for key, value in [('conditional_bias_floor', floor), ('offset_error_floor_vs_nominal', float(np.mean((mu-nominal)**2))),
                ('predicted_sampling_mse', variance), ('predicted_total_mse', floor+variance),
                ('observed_sampling_mse', float(np.mean((y-mu)**2))), ('observed_total_mse', float(np.mean((y-ref)**2)))]:
                np.testing.assert_allclose(float(row[key]), value, rtol=4e-7, atol=1e-12)
            assert int(row['bound_clipped_commands']) == np.count_nonzero(np.abs(requested)>4)
            assert int(row['stored_command_bytes']) == len(free)
            assert int(row['calibration_free_node_evaluations']) == ncal*len(free)
            assert int(row['inference_free_node_evaluations']) == 512*len(free)
            assert int(row['inference_binary_products']) == 8*65*512*n
        print(f'Independent calibration/compensation audit seed{seed}', flush=True)
    assert checked_counts == report['native_calibration_runs'] == 30
    assert report['total_calibration_free_node_evaluations'] == 10*(256+1024+4096)*len(free)
    assert report['total_inference_free_node_evaluations'] == 50*512*len(free)
    audit = {'passed': True, 'native_inference_runs': 50, 'native_calibration_count_runs': checked_counts,
        'all_measured_codes_reconstructed_from_counts': True, 'source_sha256': sha(__file__),
        'scope': 'Independent packed bit decoding, probe/estimator/control reconstruction, fixed offsets, every raw output/MSE and calibration/inference count. Actual THRML count observer separately checked against public-factor trajectories on CPU/CUDA. Synthetic fixed additive offsets only; not physical performance or full-model quality.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit, indent=2)+'\n'); print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
