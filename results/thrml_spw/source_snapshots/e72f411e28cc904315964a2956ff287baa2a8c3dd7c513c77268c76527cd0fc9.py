"""Independent raw RGB preprocessing/control calculations/metrics/bootstrap."""
import datetime
import json
import pickle
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.verify import weights_independent, metric
from results.thrml_spw.cifar_cnn_native.cohort.common import ROOT, TRAIN, OUT, CONTROLS, sha, write, guard, numpy_control


def main():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    d = json.loads((OUT/'statistics.json').read_text())
    for name, expected in d['artifact_sha256'].items():
        extension = '.npz' if name in ['test_data', 'control_logits', 'statistical_arrays'] else '.json'
        assert sha(OUT/(name+extension)) == expected, name
    data = np.load(OUT/'test_data.npz')
    x, labels = data['inputs'], data['labels']
    with Path('data/torchvision/cifar-10-batches-py/test_batch').open('rb') as stream:
        raw = pickle.load(stream, encoding='bytes')
    pixels = raw[b'data'].reshape(10000, 3, 32, 32).astype(np.float32)/np.float32(255)
    expected = (pixels[:, :, 0::2, 0::2]+pixels[:, :, 0::2, 1::2]+pixels[:, :, 1::2, 0::2]+pixels[:, :, 1::2, 1::2])/np.float32(4)
    np.testing.assert_array_equal(expected, x)
    np.testing.assert_array_equal(np.asarray(raw[b'labels']), labels)
    controls, arrays = np.load(OUT/'control_logits.npz'), np.load(OUT/'statistical_arrays.npz')
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    base = np.empty((10, 5, 2))
    errors = {}
    for mi, row in enumerate(train):
        model = np.load(row['path'])
        for ci, name in enumerate(CONTROLS):
            weights = weights_independent(model, name)
            ref = np.concatenate([numpy_control(x[first:first+128], model, weights, name.startswith('ACT8')) for first in range(0, 10000, 128)])
            retained = controls[f'c{ci}_model{mi}']
            tol = (p['control_tolerances']['F32_rtol'], p['control_tolerances']['F32_atol']) if name == 'FP32_F32' else (p['control_tolerances']['F64_rtol'], p['control_tolerances']['F64_atol'])
            np.testing.assert_allclose(ref, retained, rtol=tol[0], atol=tol[1])
            np.testing.assert_array_equal(ref.argmax(1), retained.argmax(1))
            v = metric(retained, labels)
            base[ci, mi] = v
            saved = next(r for r in d['controls'] if r['model'] == mi and r['control'] == name)
            np.testing.assert_allclose(v, [saved['accuracy'], saved['mean_CE']], rtol=2e-12, atol=2e-13)
            errors[f'model{mi}_{name}'] = float(np.max(abs(ref-retained)))
            print('Verified RGB control', mi, name, flush=True)
    np.testing.assert_allclose(base, arrays['baseline_metrics'], rtol=2e-12, atol=2e-13)
    m = json.loads((OUT/'manifest.json').read_text())
    values = np.empty((3, 5, 10, 2))
    lookup = {r['name']: r for r in d['rows']}
    assert len(lookup) == len(m['rows']) == 150
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as case:
            v = metric(case['logits'], labels)
            np.testing.assert_array_equal(case['clipped_coordinates'], row['clipped_coordinates'])
        bi = p['samples'].index(row['T'])
        values[bi, row['model'], row['seed']] = v
        np.testing.assert_allclose(v, [lookup[row['name']]['accuracy'], lookup[row['name']]['mean_CE']], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(values, arrays['values'], rtol=2e-12, atol=2e-13)
    rng = np.random.default_rng(p['bootstrap_seed'])
    bm, bs = rng.integers(0, 5, (5000, 5)), rng.integers(0, 10, (5000, 5, 10))
    np.testing.assert_array_equal(bm, arrays['bootstrap_models'])
    np.testing.assert_array_equal(bs, arrays['bootstrap_seeds'])
    assert len(d['contrasts']) == 30 and len(d['summaries']) == 3
    for contrast in d['contrasts']:
        bi, ci = contrast['budget_index'], contrast['control_index']
        assert contrast['control'] == CONTROLS[ci] and contrast['T'] == p['samples'][bi]
        delta = values[bi]-base[ci, :, None, :]
        independent = np.zeros((5000, 2))
        for unit in range(5):
            independent += delta[bm[:, unit, None], bs[:, unit]].sum(1)/50
        np.testing.assert_allclose(independent, arrays[f'budget{bi}_control{ci}'], rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(delta.mean((0, 1)), contrast['difference'], rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(np.quantile(independent, [.025, .975], axis=0).T, contrast['descriptive_95_interval'], rtol=2e-10, atol=2e-12)
    for bi, summary in enumerate(d['summaries']):
        np.testing.assert_allclose(values[bi].mean((0, 1)), summary['mean'], rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(values[bi].mean(1), summary['model_means'], rtol=2e-12, atol=2e-13)
        assert summary['clipped_coordinates'] == np.sum([r['clipped_coordinates'] for r in m['rows'] if r['T'] == summary['T']], axis=0).tolist()
    for ci, name in enumerate(CONTROLS):
        np.testing.assert_allclose(base[ci].mean(0), d['control_summary'][name], rtol=2e-12, atol=2e-13)
    guard(p)
    write(OUT/'statistics_audit.json', {'passed': True, 'controls_checked': 50, 'control_logit_values_checked': 5000000,
                                      'case_metrics_checked': 150, 'bootstrap_scalar_values_checked': 300000,
                                      'input_coordinates_exact': 7680000, 'control_maximum_errors': errors,
                                      'scope': 'Raw RGB preprocessing/labels, separate quantizer and spatial contractions, all metrics and model-unit bootstrap checks; fixed consumed data, not fresh generalization.'})
    ledger = {'weight_pbits': 3928, 'activation_pbits': 0, 'weight_field_command_bits': 31424,
              'weight_range_scalars_F64': 68, 'weight_range_bytes': 544, 'bias_parameters_FP32': 34, 'bias_bytes': 136,
              'activation_range_scalars_F64': 6, 'activation_range_bytes': 48, 'parameter_payload_bytes': 4656,
              'dynamic_weight_state_bits': 3928, 'output_counters_per8image_bank': 49266, 'counter_bytes_int32': 197064,
              'production_retained_sample_updates': 329952000000, 'public_retained_sample_updates': 329952000000,
              'warmup_updates_per_full_pass': 736500000, 'preflight_retained_sample_updates_per_pass': 52792320,
              'preflight_warmup_updates_per_pass': 117840,
              'scope': 'Logical payload and actual software observer counts only; not physical memory/area/energy/latency or optimized hardware mapping.'}
    assert 3928+544+136+48 == 4656 and 49266*4 == 197064
    write(OUT/'resource_ledger.json', ledger)
    report = ROOT.parent/'CIFAR_CNN_NATIVE_REPORT.md'
    assert not report.exists()
    lines = ['# Full native compact RGB CIFAR-10 CNN', '', 'All150cases, all50deterministic controls and independent public/statistical checks complete. This controlled16x16RGB model has3928signedweights/34deterministicbiases; its moderate conventional baseline and different architecture from the old pilot remain explicit. Fixed historically consumed test images support descriptive results, not fresh generalization.', '', '| Samples/layer | Accuracy (%) | Mean CE |', '|---:|---:|---:|']
    for row in d['summaries']:
        lines.append(f"| {row['T']} | {100*row['mean'][0]:.4f} | {row['mean'][1]:.8f} |")
    lines += ['', '| Deterministic control | Accuracy (%) | Mean CE |', '|---|---:|---:|']
    for name, v in d['control_summary'].items():
        lines.append(f'| {name} | {100*v[0]:.4f} | {v[1]:.8f} |')
    lines += ['', 'All30control contrasts and descriptive95%hierarchical model/seed intervals are retained. No favorable budget/seed is selected; a nominal-mean nonlinear CNN is a plug-in, not the expectation of stochastic CNN outputs. Low-bit baselines use dequantized arithmetic, not optimized integer hardware. Public replay verifies every receipt and final logit but is not a new quality replicate.', '',
              'Native production and public replay each329952000000retained-sample updates plus736500000warmup updates. Full-budget preflight and earlier CPU witnesses/training are separately retained. Onefreepbit pernormalizedweight needs finite8bitcontrol and range metadata; integer output counters accumulate before externalF64readout/ReLU/maxpool. Logical parameter payload4656bytes and197064bytes/bankcounters do not establish physical density, energy, latency or systems advantage.', '',
              '[Frozen protocol](cifar_cnn_native/cohort_evidence/protocol.json), [full public replay](cifar_cnn_native/cohort_evidence/native_audit.json), [all statistics/contrasts](cifar_cnn_native/cohort_evidence/statistics.json), [independent statistics audit](cifar_cnn_native/cohort_evidence/statistics_audit.json). The larger research goal remains active.']
    report.write_text('\n'.join(lines)+'\n')
    paths = [OUT/n for n in ['protocol.json', 'preflight.json', 'manifest.json', 'controls.json', 'test_data.npz', 'control_logits.npz', 'native_audit.json', 'statistics.json', 'statistical_arrays.npz', 'statistics_audit.json', 'resource_ledger.json']]+[report]
    write(OUT/'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'status': 'all150compactRGBnative_cases_independently_audited_goal_active',
                                 'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All150native RGB cases/50controls/300000bootstrap values independently checked', flush=True)


if __name__ == '__main__':
    main()
