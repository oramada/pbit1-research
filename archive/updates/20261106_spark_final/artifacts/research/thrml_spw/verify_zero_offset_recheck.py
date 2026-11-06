"""Raw audit and separate fresh/combined statistics for the fixed seed recheck."""
import csv
import json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha

OUT = Path('results/thrml_spw/zero_offset_recheck')


def main():
    p = json.loads((OUT/'protocol.json').read_text()); report = json.loads((OUT/'REPORT.json').read_text())
    run = json.loads((OUT/'manifest.json').read_text())
    assert run['config'] == p and run['result'] == report
    assert run['source_commit'] == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert sha('research/thrml_spw/zero_offset_recheck.py') == p['source_sha256']
    assert sha('research/thrml_spw/controlled_replica.py') == p['operator_sha256']
    assert p['seeds'] == list(range(10, 50)) and p['observations'] == 512 and p['replicas'] == [1, 4]
    prior = Path('results/thrml_spw/calibration_operator')
    assert sha(prior/'metrics.csv') == p['prior_metrics_sha256'] and sha(prior/'comparisons.json') == p['prior_comparisons_sha256']
    assert json.loads((prior/'evidence_audit.json').read_text())['passed']
    artifact = Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
    old = Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')
    xp, rp = old/'stage11_inputs.npy', old/'stage11_conditional_reference.npy'
    for path, key in [(artifact, 'artifact_sha256'), (xp, 'inputs_sha256'), (rp, 'reference_sha256')]: assert sha(path) == p[key]
    x, ref = np.load(xp)[:8].astype(np.float64), np.load(rp)[:8].astype(np.float64)
    with np.load(artifact) as a:
        h = json.loads(a['header'].tobytes()); shape = tuple(h['shape']); n = int(np.prod(shape))
        bits = np.unpackbits(a['modes'], bitorder='little').reshape(-1, 2)[:n]
        mode = bits@np.array([1, 2], np.uint8); free = np.flatnonzero(mode == 0)
        field = (-4+a['codes'].astype(np.float64)*8/255).astype(np.float32)
        q = (mode == 2).astype(np.float64); q[free] = 1/(1+np.exp(-2*field.astype(np.float64))); q = q.reshape(shape)
        lower = np.repeat(a['lower'].astype(np.float64), 128, 0); span = np.repeat(a['span'].astype(np.float64), 128, 0); bias = a['bias'].copy()
    mu = x@(lower+span*q)+bias; floor = float(np.mean((mu-ref)**2))
    variance = float(np.mean(np.mean(x*x, axis=(0, 1))@(span**2*(q-q*q)))/512)
    np.testing.assert_allclose(report['conditional_bias_floor'], floor, rtol=1e-10)
    np.testing.assert_allclose(report['predicted_sampling_mse'], variance, rtol=4e-7)
    with (OUT/'metrics.csv').open() as f: rows = list(csv.DictReader(f))
    assert len(rows) == 80 and {(int(r['K']), int(r['seed'])) for r in rows} == {(k, s) for k in [1, 4] for s in range(10, 50)}
    assert len(list((OUT/'cases').glob('*.npz'))) == len(list((OUT/'cases').glob('*.json'))) == 80
    values = {1: [], 4: []}
    for r in rows:
        k, seed = int(r['K']), int(r['seed'])
        prefix = OUT/'cases'/f'K{k}_seed{seed}'
        case = json.loads(prefix.with_suffix('.json').read_text())
        for key, value in case.items(): assert str(value) == r[key]
        assert sha(prefix.with_suffix('.npz')) == r['archive_sha256']
        y = np.load(prefix.with_suffix('.npz'))['output'].astype(np.float64)
        assert y.shape == ref.shape and np.isfinite(y).all()
        mse = float(np.mean((y-ref)**2)); sampling = float(np.mean((y-mu)**2))
        np.testing.assert_allclose(float(r['observed_total_mse']), mse, rtol=1e-12)
        np.testing.assert_allclose(float(r['observed_sampling_mse']), sampling, rtol=1e-12)
        assert int(r['T']) == 512//k and int(r['free_node_evaluations']) == 512*len(free)
        values[k].append((seed, mse, sampling))
    values = {k: np.array([v[1:] for v in sorted(a)]) for k, a in values.items()}
    with (prior/'metrics.csv').open() as f: old_rows = list(csv.DictReader(f))
    original = {k: np.array([[float(r['observed_total_mse']), float(r['observed_sampling_mse'])] for r in sorted(
        [r for r in old_rows if int(r['K']) == k and int(r['sigma_index']) == 0 and int(r['observations']) == 512], key=lambda r: int(r['device_sampling_seed']))]) for k in [1, 4]}
    summaries = []
    for label, arrays in [('original10', original), ('fresh40', values), ('combined50_diagnostic', {k: np.concatenate([original[k], values[k]]) for k in [1, 4]})]:
        delta = arrays[4][:, 0]-arrays[1][:, 0]; margin = float(1.96*delta.std(ddof=1)/np.sqrt(len(delta)))
        summaries.append({'batch': label, 'seeds': len(delta), 'K1_MSE': float(arrays[1][:, 0].mean()), 'K4_MSE': float(arrays[4][:, 0].mean()),
            'K1_sampling_over_prediction': float(arrays[1][:, 1].mean()/variance), 'K4_sampling_over_prediction': float(arrays[4][:, 1].mean()/variance),
            'K4_minus_K1_mean': float(delta.mean()), 'approximate_paired_95_margin': margin,
            'interval_includes_zero': bool(abs(delta.mean()) <= margin)})
    (OUT/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    audit = {'passed': True, 'native_runs': 80, 'original_runs_preserved': 20, 'source_sha256': sha(__file__),
        'scope': 'Every fresh output/MSE/resource counter independently recomputed; fixed fresh seed set audited, original ten-seed result retained. Diagnostic follow-up and correlated combined analysis, not equivalence or physical superiority proof.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    text = ['# Zero-offset finite-seed discrepancy: fixed diagnostic follow-up\n',
        'The primary synthetic-calibration study had a small K4-minus-K1 MSE difference at zero offset whose unadjusted approximate ten-seed interval excluded zero. This triggered exactly forty new sampling seeds10..49 at512 total observations per weight. The follow-up preserves the original key schedule, eight old contexts, 2,359,296-weight operator and finite eight-bit controls. It does not stop based on significance or overwrite the original result. All80 new native outputs and counters were independently audited.\n',
        '| Batch | Seeds | K1 MSE | K4 MSE | K1 sampling / prediction | K4 sampling / prediction | K4 minus K1 | Approx. 95% margin | Includes zero |\n|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in summaries:
        text.append(f"| {r['batch']} | {r['seeds']} | {r['K1_MSE']:.7f} | {r['K4_MSE']:.7f} | {r['K1_sampling_over_prediction']:.4f} | {r['K4_sampling_over_prediction']:.4f} | {r['K4_minus_K1_mean']:+.7f} | {r['approximate_paired_95_margin']:.7f} | {r['interval_includes_zero']} |")
    text.extend(['\nThe conditional total-MSE prediction is identical for K1/T512 and K4/T128: '+f'{floor+variance:.8f}. The shared finite-control bias floor is {floor:.8f}. The new batch is separate from the triggering data; the combined50 analysis includes that triggering data and is diagnostic, not an independent confirmation. Approximate normal intervals are neither multiplicity-adjusted nor formal equivalence tests. Finite-seed discrepancies and all raw outcomes are retained regardless of whether a displayed interval covers zero.\n',
        'This investigates random estimation variation in an isolated matched-resource operator. It cannot establish full-network quality, equal hardware energy, or that Extropic uses four p-bits per learned weight. See the main calibration-error and measured-calibration reports for the distinct fixed-device-error question.\n'])
    rendered = '\n'.join(text)+'\n'
    for a, b in {'seeds10':'seeds 10','at512':'at 512','All80':'All 80','combined50':'combined 50'}.items(): rendered = rendered.replace(a,b)
    Path('results/thrml_spw/ZERO_OFFSET_RECHECK_REPORT.md').write_text(rendered)
    print(json.dumps({'audit': audit, 'summaries': summaries}, indent=2))


if __name__ == '__main__': main()
