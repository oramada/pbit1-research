"""Independent packed-bit, fixed-field and raw-output audit of calibration study."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha

OUT = Path('results/thrml_spw/calibration_operator')


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    report = json.loads((OUT/'REPORT.json').read_text())
    run = json.loads((OUT/'manifest.json').read_text())
    assert run['config'] == p and run['result'] == report
    assert run['source_commit'] == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for path, digest in p['source_sha256'].items():
        assert sha(path) == digest
    trajectory = json.loads((OUT/'trajectory_test_record.json').read_text())
    assert trajectory['test_source_sha256'] == sha('research/thrml_spw/test_controlled_replica.py')
    assert trajectory['operator_source_sha256'] == sha('research/thrml_spw/controlled_replica.py')
    assert len(trajectory['observed_terminal_results']) == 2
    assert all(r['exit_code'] == 0 and r['result'] == 'OK' for r in trajectory['observed_terminal_results'])
    source = Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')
    artifact = Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
    xp, rp = source/'stage11_inputs.npy', source/'stage11_conditional_reference.npy'
    for path, key in [(artifact, 'artifact_sha256'), (xp, 'inputs_sha256'), (rp, 'reference_sha256')]:
        assert sha(path) == p[key]
    assert p['sigmas'] == [0., .05, .2] and p['total_observations_per_weight'] == [128, 512]
    assert p['device_and_sampling_seeds'] == list(range(10)) and p['contexts'] == 8
    x, ref = np.load(xp)[:8].astype(np.float64), np.load(rp)[:8].astype(np.float64)
    with np.load(artifact) as a:
        header = json.loads(a['header'].tobytes())
        shape = tuple(header['shape'])
        assert shape == (3072, 768)
        n = int(np.prod(shape))
        bit_pairs = np.unpackbits(a['modes'], bitorder='little').reshape(-1, 2)[:n]
        mode = bit_pairs@np.array([1, 2], dtype=np.uint8)
        free = np.flatnonzero(mode == 0)
        base_fields = (-4+a['codes'].astype(np.float64)*8/255).astype(np.float32)
        assert len(base_fields) == len(free)
        lower = np.repeat(a['lower'].astype(np.float64), 128, axis=0)
        span = np.repeat(a['span'].astype(np.float64), 128, axis=0)
        bias = a['bias'].copy()
    nominal_q = (mode == 2).astype(np.float64)
    nominal_q[free] = 1/(1+np.exp(-2*base_fields.astype(np.float64)))
    nominal = x@(lower+span*nominal_q.reshape(shape))+bias
    np.testing.assert_allclose(np.mean((nominal-ref)**2), report['nominal_control_bias_floor'], rtol=1e-10)
    with (OUT/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    expected = {(k, si, dep, seed, obs) for k in [1, 4] for si in range(3)
        for dep in (['independent', 'shared'] if k == 4 and si else ['independent'])
        for seed in range(10) for obs in [128, 512]}
    observed = {(int(r['K']), int(r['sigma_index']), r['dependence'], int(r['device_sampling_seed']), int(r['observations'])) for r in rows}
    assert len(rows) == report['native_conditions'] == len(expected) == 160 and observed == expected
    assert len(list((OUT/'cases').glob('*.json'))) == len(list((OUT/'cases').glob('*.npz'))) == 160
    checked = 0
    floors = {}
    for k in [1, 4]:
        for si, sigma in enumerate([0., .05, .2]):
            for dep in (['independent', 'shared'] if k == 4 and si else ['independent']):
                for seed in range(10):
                    z = np.random.default_rng(2026102700+seed).standard_normal((4, len(free)))
                    selected = z[:k] if dep == 'independent' else np.repeat(z[:1], k, axis=0)
                    f = (base_fields+sigma*selected).astype(np.float32)
                    digest = hashlib.sha256(f.ravel().tobytes()).hexdigest()
                    q = np.repeat((mode == 2).astype(np.float64)[None], k, axis=0)
                    q[:, free] = 1/(1+np.exp(-2*f.astype(np.float64)))
                    q = q.reshape(k, *shape)
                    mu = x@(lower+span*np.add.reduce(q, axis=0)/k)+bias
                    floor = float(np.mean(np.square(mu-ref)))
                    offset = float(np.mean(np.square(mu-nominal)))
                    floors[k, si, dep, seed] = floor
                    # Sum the weight variances before applying input squared norms.
                    v = span**2*np.add.reduce(q-q*q, axis=0)/(k*k)
                    variance = float(np.mean(np.mean(np.square(x), axis=(0, 1))@v))
                    group = [r for r in rows if (int(r['K']), int(r['sigma_index']), r['dependence'], int(r['device_sampling_seed'])) == (k, si, dep, seed)]
                    assert len(group) == 2
                    for row in group:
                        obs = int(row['observations']); t = obs//k
                        assert row['name'] == f'K{k}_s{si}_{dep}_seed{seed}_N{obs}'
                        assert float(row['sigma_b']) == sigma
                        case = json.loads((OUT/'cases'/f"{row['name']}.json").read_text())
                        for key, value in case.items():
                            assert str(value) == row[key]
                        npz = OUT/'cases'/f"{row['name']}.npz"
                        assert sha(npz) == row['output_archive_sha256'] and digest == row['field_sha256']
                        y = np.load(npz)['output'].astype(np.float64)
                        assert y.shape == ref.shape and np.isfinite(y).all()
                        assert int(row['T']) == t and int(row['logical_nodes']) == k*n
                        assert int(row['free_node_evaluations']) == obs*len(free)
                        assert int(row['logical_weight_readouts']) == obs*n
                        assert int(row['nominal_binary_product_terms']) == 8*65*obs*n
                        for key, value in [('conditional_bias_floor', floor), ('offset_error_floor_vs_nominal', offset),
                            ('predicted_sampling_mse', variance/t), ('predicted_total_mse', floor+variance/t),
                            ('observed_sampling_mse', float(np.mean((y-mu)**2))),
                            ('observed_total_mse', float(np.mean((y-ref)**2)))]:
                            np.testing.assert_allclose(float(row[key]), value, rtol=4e-7, atol=1e-12)
                        checked += 1
                print(f'Independent raw audit K{k} sigma={sigma} {dep}', flush=True)
    # Shared offsets cannot be averaged away by replicas. This checks the exact
    # conditional expectation, independent of random finite-sample fluctuations.
    for si in range(3):
        dep = 'shared' if si else 'independent'
        for seed in range(10):
            np.testing.assert_allclose(floors[1, si, 'independent', seed], floors[4, si, dep, seed], rtol=1e-11, atol=1e-14)
    for k in [1, 4]:
        c = report['controls'][str(k)]
        assert c['logical_pbits'] == k*n and c['free_pbits'] == k*len(free)
    audit = {'passed': True, 'native_conditions': checked, 'matrix_weights': n,
        'all_fixed_field_hashes_reconstructed': True, 'shared_offset_mean_identity_checked': True,
        'source_sha256': sha(__file__),
        'scope': 'Independent packed decoding, probability means, fixed-offset fields, every raw output/MSE and resource counter. Tiny CPU/CUDA checks separately validate changed official fields against public-factor native trajectories. Synthetic device offsets, no physical-device or formal equivalence claim.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
