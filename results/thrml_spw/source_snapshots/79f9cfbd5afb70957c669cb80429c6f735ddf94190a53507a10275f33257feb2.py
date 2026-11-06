"""Independent raw-logit metrics and scalar bootstrap enumeration for BF16 contrasts."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw')
OUT = ROOT/'bf16_comparison/evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(z, y):
    maximum = np.max(z, axis=1)
    loss = maximum+np.log(np.sum(np.exp(z-maximum[:, None]), axis=1))-z[np.arange(len(y)), y]
    return np.array([np.count_nonzero(z.argmax(1) == y)/len(y), loss.sum()/len(y)])


def main():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, digest in p[group].items():
            assert sha(path) == digest, path
    s = json.loads((OUT/'statistics.json').read_text())
    assert s['protocol_sha256'] == sha(OUT/'protocol.json')
    assert s['bootstrap_arrays_sha256'] == sha(OUT/'bootstrap_arrays.npz')
    saved = np.load(OUT/'bootstrap_arrays.npz')
    checked, seen, all_native, all_control, raw_values = [], set(), 0, 0, 0
    for dataset in ['fashion', 'cifar']:
        root = ROOT/(dataset+'_cnn_native')
        y = np.load(root/'cohort_evidence/test_data.npz')['labels']
        assert y.shape == (10000,)
        learned = np.load(root/'probability_cohort_evidence/statistical_arrays.npz')
        base = np.load(root/'cohort_evidence/statistical_arrays.npz')
        precision = np.load(root/'precision_evidence/statistical_arrays.npz')
        native = np.zeros((2, 3, 5, 10, 2))
        manifest = json.loads((root/'probability_cohort_evidence/manifest.json').read_text())
        expected = {(a, b, t, seed) for a in range(2) for b in range(5) for t in [64, 256, 1024] for seed in range(10)}
        assert len(manifest['rows']) == 300
        assert {(r['method'], r['model'], r['T'], r['seed']) for r in manifest['rows']} == expected
        for row in manifest['rows']:
            assert sha(row['path']) == row['sha256']
            with np.load(row['path']) as case:
                z = case['logits']
                assert z.shape == (10000, 10) and np.isfinite(z).all()
                native[row['method'], [64, 256, 1024].index(row['T']), row['model'], row['seed']] = metric(z, y)
                raw_values += z.size
            all_native += 1
        np.testing.assert_allclose(native, learned['values'], rtol=2e-12, atol=2e-13)
        controls = {}
        with np.load(root/'precision_evidence/logits.npz') as logits:
            for ci in [1, 3]:
                controls[ci] = np.array([metric(logits[f'c{ci}_model{mi}'], y) for mi in range(5)])
                np.testing.assert_allclose(controls[ci], precision['control_metrics'][ci], rtol=2e-12, atol=2e-13)
                all_control += 5
                raw_values += 5*10000*10
        for c in [c for c in s['contrasts'] if c['dataset'] == dataset]:
            method, bi, ci = c['method_index'], [64, 256, 1024].index(c['T']), c['control_index']
            key = (dataset, method, bi, ci)
            assert key not in seen
            seen.add(key)
            assert c['control'] == {1: 'BF16_weights_F64', 3: 'BF16_Torch_CPU'}[ci]
            assert c['method'] == p['methods'][method]
            left, right = native[method, bi], controls[ci]
            per_model = np.array([left[mi].sum(0)/10-right[mi] for mi in range(5)])
            point = per_model.sum(0)/5
            # Enumerate selected model units separately from producer broadcast indexing.
            boot = np.zeros((5000, 2))
            for unit in range(5):
                model_ids = base['bootstrap_models'][:, unit]
                seed_ids = learned['bootstrap_method_seeds'][method, :, unit]
                boot += left[model_ids[:, None], seed_ids].sum(1)/50-right[model_ids]/5
            np.testing.assert_allclose(boot, saved[c['array_key']], rtol=2e-9, atol=3e-12)
            np.testing.assert_allclose(point, c['difference'], rtol=2e-9, atol=3e-12)
            np.testing.assert_allclose(per_model, c['per_model_difference'], rtol=2e-9, atol=3e-12)
            np.testing.assert_allclose(left.mean((0, 1)), c['native_mean'], rtol=2e-12, atol=2e-13)
            np.testing.assert_allclose(right.mean(0), c['BF16_mean'], rtol=2e-12, atol=2e-13)
            interval = np.quantile(boot, [.025, .975], axis=0).T
            np.testing.assert_allclose(interval, c['descriptive_95_interval'], rtol=2e-9, atol=3e-12)
            checked.append(c)
        print(dataset, 'all300native cases,10BF16paths and12contrasts independently checked', flush=True)
    assert seen == {(d, m, b, c) for d in ['fashion', 'cifar'] for m in range(2) for b in range(3) for c in [1, 3]}
    assert len(s['contrasts']) == len(saved.files) == len(checked) == 24
    report = ROOT/'SINGLE_PBIT_VS_BF16_REPORT.md'
    assert not report.exists()
    lines = ['# One p-bit per weight versus BF16: direct task-quality comparison', '',
             'On the fixed Fashion-MNIST cohort, the sample-aware one-p-bit method reaches 85.0262% accuracy at T=1024, versus 85.1100% for CPU BF16 execution and 85.0660% for BF16-rounded weights with F64 arithmetic. This is a small observed task-accuracy gap, not proof of BF16 numerical precision or statistical equivalence.', '',
             'The corresponding RGB CIFAR-10 method reaches 55.0146%, versus 56.2940% CPU BF16 and 56.2660% BF16-rounded/F64. Thus comparable performance is model-dependent: this experiment does not establish universal BF16-equivalent inference.', '',
             '| Dataset | Training | T | BF16 control | P-bit accuracy (%) | BF16 accuracy (%) | P-bit minus BF16 (pp) | Descriptive 95% interval (pp) |',
             '|---|---|---:|---|---:|---:|---:|---|']
    for c in checked:
        lo, hi = c['descriptive_95_interval'][0]
        lines.append(f"|{c['dataset']}|{c['method']}|{c['T']}|{c['control']}|{100*c['native_mean'][0]:.4f}|{100*c['BF16_mean'][0]:.4f}|{100*c['difference'][0]:+.4f}|[{100*lo:+.6f}, {100*hi:+.6f}]|")
    lines += ['', 'This retrospective analysis retains both pre-existing training methods, all three budgets and both BF16 paths: 24 contrasts, five paired starting-model units, ten MC inference seeds per method/budget/model, and 5000 inherited paired-model/within-model-seed bootstrap draws. Test images are fixed and historically consumed. Intervals are descriptive and unadjusted for multiplicity. No equivalence/noninferiority margin or test was selected; an interval containing zero does not establish equivalence. A confirmatory BF16-equivalence study still needs an application-defined acceptable difference and fresh held-out evaluation.', '',
              'The learned p-bit methods have eight additional training epochs versus the frozen original BF16 controls; these contrasts describe the end-to-end procedures and do not isolate encoding from training. Expected-weight training controls the sampled training parameterization and epoch budget. Training uses a biased surrogate, while the reported inference is actual official THRML simulation with full public replay. Both small CNN baselines are limited architectures; Fashion and compact RGB CIFAR are distinct tasks.', '',
              'Each normalized signed weight uses one free p-bit; original centers/radii map its probability into the signed weight range. Biases (34/model), group metadata, 8-bit field commands, ACT8 activations and integer counters/external F64 readout remain deterministic resources. Fashion uses 2664 weight p-bits and CIFAR 3928. A p-bit state remains binary. Matching task quality does not supply BF16 numerical precision per parameter or BF16 arithmetic. Time samples and physical probability-programming precision remain costs.', '',
              'BF16_weights_F64 rounds only weights to BF16, retaining F64 arithmetic and original deterministic biases. BF16_Torch_CPU executes the existing model with BF16 tensors; internal accumulator widths are not asserted. Neither path measures probabilistic hardware latency or energy. No physical memory, area, energy, or latency advantage follows.', '',
              f'Independent verification recomputed metrics from {all_native} native case files and {all_control} BF16 model paths ({raw_values} stored logit values), checked all24 contrasts and240000bootstrap scalar values. Existing native public-replay/graph audits remain the evidence for original inference correctness; this analysis makes zero new stochastic-device evaluations and adds no independent inference/model replicas.', '',
              '[Full statistics](bf16_comparison/evidence/statistics.json) · [Analysis protocol](bf16_comparison/evidence/protocol.json) · [Verification](bf16_comparison/evidence/evidence_audit.json)', '',
              'The long-horizon research goal remains active.']
    report.write_text('\n'.join(lines)+'\n')
    audit = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source_sha256': sha(__file__), 'native_case_metrics_checked': all_native,
             'BF16_model_paths_checked': all_control, 'stored_logit_values_checked': raw_values,
             'contrast_rows_checked': 24, 'bootstrap_scalar_values_checked': 240000,
             'statistics_sha256': sha(OUT/'statistics.json'), 'new_native_updates': 0,
             'scope': 'Independent raw-logit metrics and separate bootstrap-unit enumeration, not fresh data, formal equivalence or physical hardware evidence.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    paths = [OUT/n for n in ['protocol.json', 'statistics.json', 'bootstrap_arrays.npz', 'evidence_audit.json']]+[report]
    (OUT/'completion.json').write_text(json.dumps({'passed': True, 'status': 'all24_BF16_descriptive_contrasts_verified_goal_active',
        'artifact_sha256': {str(path): sha(path) for path in paths}}, indent=2)+'\n')
    print('All24 direct BF16 contrasts and240000bootstrap scalar values independently verified')


if __name__ == '__main__':
    main()
