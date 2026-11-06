"""Independent count/control/output audit of native gain-offset calibration."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha

OUT = Path('results/thrml_spw/gain_calibrated_operator')


def raw_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p = json.loads((OUT/'protocol.json').read_text()); report = json.loads((OUT/'REPORT.json').read_text())
    run = json.loads((OUT/'manifest.json').read_text())
    assert run['config'] == p and run['result'] == report
    assert run['source_commit'] == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for path, h in p['source_sha256'].items(): assert sha(path) == h
    assert p['log_gain_sd'] == [0., .2] and p['offset_sd'] == .2 and p['seeds'] == list(range(10))
    assert p['total_probe_samples'] == [256, 1024, 4096] and p['inference_samples'] == 512
    assert p['single_probe_code'] == 128 and p['two_probe_codes'] == [96, 159]
    test = json.loads((OUT/'estimator_test_record.json').read_text())
    assert test['exit_code'] == 0 and test['tests'] == 2 and test['test_source_sha256'] == sha('research/thrml_spw/test_gain_calibration_math.py')
    # The native field/count implementation is unchanged from its actual CPU/CUDA
    # public-factor trajectory tests and independently audited offset experiment.
    earlier = Path('results/thrml_spw/calibrated_operator')
    assert json.loads((earlier/'evidence_audit.json').read_text())['passed']
    old_protocol = json.loads((earlier/'protocol.json').read_text())
    assert old_protocol['source_sha256']['research/thrml_spw/calibration_count_native.py'] == sha('research/thrml_spw/calibration_count_native.py')
    assert old_protocol['source_sha256']['research/thrml_spw/controlled_replica.py'] == sha('research/thrml_spw/controlled_replica.py')
    for device in ['cpu', 'gpu']: assert 'OK' in (earlier/f'counts_{device}_test.log').read_text()
    artifact = Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
    old = Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')
    xp, rp = old/'stage11_inputs.npy', old/'stage11_conditional_reference.npy'
    for path, key in [(artifact, 'artifact_sha256'), (xp, 'inputs_sha256'), (rp, 'reference_sha256')]: assert sha(path) == p[key]
    x, ref = np.load(xp)[:8].astype(np.float64), np.load(rp)[:8].astype(np.float64)
    with np.load(artifact) as a:
        header = json.loads(a['header'].tobytes()); shape = tuple(header['shape']); n = int(np.prod(shape))
        bit_pairs = np.unpackbits(a['modes'], bitorder='little').reshape(-1, 2)[:n]
        modes = bit_pairs@np.array([1, 2], np.uint8); free = np.flatnonzero(modes == 0)
        codes0 = a['codes'].copy(); field0 = (-4+codes0.astype(float)*8/255).astype(np.float32)
        lower = np.repeat(a['lower'].astype(float), 128, 0); span = np.repeat(a['span'].astype(float), 128, 0); bias = a['bias'].copy()
    assert shape == (3072, 768)
    nominal_q = (modes == 2).astype(float); nominal_q[free] = 1/(1+np.exp(-2*field0.astype(float)))
    nominal = x@(lower+span*nominal_q.reshape(shape))+bias
    np.testing.assert_allclose(np.mean((nominal-ref)**2), report['nominal_control_bias_floor'], rtol=1e-10)
    with (OUT/'metrics.csv').open() as f: rows = list(csv.DictReader(f))
    methods = ['uncalibrated', 'oracle', 'single256', 'two256', 'single1024', 'two1024', 'single4096', 'two4096']
    assert p['methods'] == methods
    assert len(rows) == report['native_inference_runs'] == 160
    assert {(int(r['scenario']), int(r['seed']), r['method']) for r in rows} == {(sc, s, m) for sc in [0, 1] for s in range(10) for m in methods}
    assert len(list((OUT/'cases').glob('*.json'))) == len(list((OUT/'cases').glob('*.npz'))) == 160
    probes = {code: float(np.float32(-4+code*8/255)) for code in [96, 128, 159]}
    assert {str(k): v for k, v in probes.items()} == report['probe_fields']
    checked_probes = 0
    for scenario, sd in enumerate([0., .2]):
        for seed in range(10):
            delta = .2*np.random.default_rng(2026102700+seed).standard_normal((4, len(free)))[0]
            z = np.random.default_rng(2026102900+seed).standard_normal(len(free))
            gain = np.exp(sd*z-sd*sd/2)
            for method in methods:
                r = next(r for r in rows if int(r['scenario']) == scenario and int(r['seed']) == seed and r['method'] == method)
                prefix = OUT/'cases'/f'scenario{scenario}_seed{seed}_{method}'
                case = json.loads(prefix.with_suffix('.json').read_text())
                for key, value in case.items(): assert ('' if value is None else str(value)) == r[key]
                assert raw_sha(delta) == r['offset_sha256'] and raw_sha(gain) == r['gain_sha256'] and sha(prefix.with_suffix('.npz')) == r['archive_sha256']
                data = np.load(prefix.with_suffix('.npz')); codes, y = data['codes'], data['output'].astype(float)
                assert codes.dtype == np.uint8 and codes.shape == codes0.shape and y.shape == ref.shape and np.isfinite(y).all()
                budget = next((ncal for ncal in [256, 1024, 4096] if method.endswith(str(ncal))), 0)
                assert int(r['total_probe_samples']) == budget and float(r['log_gain_sd']) == sd
                if budget:
                    pcodes = [128] if method.startswith('single') else [96, 159]
                    each = budget//len(pcodes); u = {}; mse = []; prediction = []
                    assert set(data.files) == {'output', 'codes'}|{f'count_{c}' for c in pcodes}
                    for code in pcodes:
                        counts = data[f'count_{code}']; assert counts.dtype == np.int32 and counts.shape == codes0.shape and np.all((counts >= 0)&(counts <= each))
                        u[code] = .5*(np.log(counts.astype(float)+.5)-np.log(each-counts.astype(float)+.5))
                        physical = (gain*probes[code]+delta).astype(np.float32)
                        qp = 1/(1+np.exp(-2*physical.astype(float)))
                        mse.append(np.mean((counts/each-qp)**2)); prediction.append(np.mean(qp*(1-qp))/each)
                        checked_probes += 1
                    if len(pcodes) == 1:
                        a_hat = np.ones(len(free)); d_hat = u[128]-probes[128]
                        low_count = high_count = 0
                    else:
                        raw = (u[159]-u[96])/(probes[159]-probes[96])
                        a_hat = np.minimum(4., np.maximum(.25, raw))
                        d_hat = .5*(u[159]+u[96])-.5*raw*(probes[159]+probes[96])
                        low_count, high_count = np.count_nonzero(raw < .25), np.count_nonzero(raw > 4.)
                    np.testing.assert_allclose(float(r['estimated_gain_rmse']), np.sqrt(np.mean((a_hat-gain)**2)), rtol=1e-12, atol=1e-14)
                    np.testing.assert_allclose(float(r['estimated_offset_rmse']), np.sqrt(np.mean((d_hat-delta)**2)), rtol=1e-12)
                    np.testing.assert_allclose(float(r['calibration_frequency_mse']), np.mean(mse), rtol=1e-12)
                    np.testing.assert_allclose(float(r['calibration_predicted_frequency_mse']), np.mean(prediction), rtol=1e-12)
                    requested = (field0.astype(float)-d_hat)/a_hat
                    assert int(r['gain_guard_low']) == low_count and int(r['gain_guard_high']) == high_count
                else:
                    assert set(data.files) == {'output', 'codes'}
                    assert r['estimated_gain_rmse'] == r['estimated_offset_rmse'] == ''
                    requested = (field0.astype(float)-delta)/gain if method == 'oracle' else field0.astype(float)
                want = np.rint((np.clip(requested, -4, 4)+4)*255/8).astype(np.uint8)
                np.testing.assert_array_equal(codes, want)
                assert raw_sha(codes) == r['command_sha256']
                realized = (gain*(-4+codes.astype(float)*8/255).astype(np.float32).astype(float)+delta).astype(np.float32)
                assert raw_sha(realized) == r['realized_field_sha256']
                q = (modes == 2).astype(float); q[free] = 1/(1+np.exp(-2*realized.astype(float))); q = q.reshape(shape)
                mean = x@(lower+span*q)+bias; floor = float(np.mean((mean-ref)**2))
                variance = float(np.mean(np.mean(x*x, axis=(0, 1))@(span*span*(q-q*q)))/512)
                for key, value in [('conditional_bias_floor', floor), ('offset_error_floor_vs_nominal', float(np.mean((mean-nominal)**2))),
                    ('predicted_sampling_mse', variance), ('predicted_total_mse', floor+variance),
                    ('observed_sampling_mse', float(np.mean((y-mean)**2))), ('observed_total_mse', float(np.mean((y-ref)**2)))]:
                    np.testing.assert_allclose(float(r[key]), value, rtol=4e-7, atol=1e-12)
                assert int(r['probe_count']) == (1 if method.startswith('single') else 2 if method.startswith('two') else 0)
                assert int(r['bound_clipped_commands']) == np.count_nonzero(np.abs(requested)>4)
                assert int(r['calibration_free_node_evaluations']) == budget*len(free)
                assert int(r['inference_free_node_evaluations']) == 512*len(free)
                assert int(r['inference_binary_products']) == 8*65*512*n and int(r['stored_command_bytes']) == len(free)
            print(f'Independent gain-calibration audit scenario{scenario} seed{seed}', flush=True)
    assert checked_probes == report['native_calibration_runs'] == 180
    assert report['total_calibration_free_node_evaluations'] == 40*(256+1024+4096)*len(free)
    assert report['total_inference_free_node_evaluations'] == 160*512*len(free)
    audit = {'passed': True, 'native_inference_runs': 160, 'native_calibration_probe_runs': checked_probes,
        'all_measured_commands_reconstructed_from_counts': True, 'equal_total_probe_budget_checked': True,
        'source_sha256': sha(__file__), 'scope': 'Independent packed-bit decoding, all fixed device parameters/probe counts, both estimators, guarded inverse codes, output MSEs and calibration/inference counters. Synthetic static gain/offset only; no physical or full-model claim.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit, indent=2)+'\n'); print(json.dumps(audit, indent=2))


if __name__ == '__main__': main()
