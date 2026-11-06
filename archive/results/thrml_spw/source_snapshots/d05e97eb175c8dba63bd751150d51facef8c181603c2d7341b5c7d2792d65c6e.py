"""Independent mean-control graph and complete metrics/bootstrap verification."""
import datetime
import json
import sys

import numpy as np
import torch
from torch.nn import functional as F

from results.thrml_spw.fashion_cnn_native.probability_cohort.common import (
    ROOT, OUT, BASE, WORD, METHODS, sha, write, guard, parent_closed, checkpoint)


def reference(x, exported, original):
    h = torch.tensor(x, dtype=torch.float64)
    for li, prefix in enumerate(['conv1', 'conv2', 'head']):
        source = original[prefix+'_weight'].astype(float)
        flat = source.reshape(len(source), -1)
        c = (flat.max(1)+flat.min(1))/2
        r = (flat.max(1)-flat.min(1))/2
        np.testing.assert_array_equal(c, exported[prefix+'_center'])
        np.testing.assert_array_equal(r, exported[prefix+'_radius'])
        commands = np.rint((exported[prefix+'_latent_fields']+4)*255/8).astype(np.uint8)
        np.testing.assert_array_equal(commands, exported[prefix+'_commands'])
        fields = np.asarray([[-4+float(code)*8/255 for code in row] for row in commands], np.float32).astype(float)
        mean = c[:, None]-r[:, None]+2*r[:, None]*(1/(1+np.exp(-2*fields)))
        np.testing.assert_allclose(mean.reshape(source.shape), exported[prefix+'_weight'], rtol=2e-14, atol=2e-15)
        span = float(exported['activation_span'][li])
        h = span*torch.round(255*torch.clamp(h/span, 0, 1))/255
        weight = torch.tensor(mean.reshape(source.shape), dtype=torch.float64)
        bias = torch.tensor(original[prefix+'_bias'], dtype=torch.float64)
        h = F.max_pool2d(F.relu(F.conv2d(h, weight, bias, padding=1)), 2) if li < 2 else F.linear(h.flatten(1), weight, bias)
    return h.numpy()


def metric(z, labels):
    maximum = np.max(z, axis=1)
    loss = np.log(np.sum(np.exp(z-maximum[:, None]), axis=1))+maximum-z[np.arange(len(labels)), labels]
    return np.array([np.count_nonzero(np.argmax(z, axis=1)==labels)/len(labels), np.mean(loss)])


def selfcheck():
    from results.thrml_spw.fashion_cnn_native.cohort.common import numpy_control
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'control_selfcheck.json').exists()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    x = np.load(ROOT/'training/validation.npz')['inputs'][:64]
    originals = json.loads((ROOT/'training/manifest.json').read_text())['rows']
    rows = []
    for method in range(2):
        for mi in range(5):
            model = checkpoint(mi, method)
            original = np.load(originals[mi]['path'])
            expected = reference(x, model, original)
            actual = numpy_control(x, model, [model[n+'_weight'] for n in ['conv1', 'conv2', 'head']], True)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-11)
            np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
            rows.append({'method': METHODS[method], 'model': mi, 'max_logit_error': float(np.max(abs(actual-expected)))})
    write(OUT/'control_selfcheck.json', {'passed': True, 'rows': rows, 'source_sha256': sha(__file__),
                                       'scope': 'Validation-only independent TorchF64 graph versus NumPyF64 mean controls, all10models/64inputs. No labels or test scores/native calls.'})


def verify():
    from results.thrml_spw.fashion_cnn_native.cohort.common import CONTROLS
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assert not (OUT/'completion.json').exists()
    s = json.loads((OUT/'statistics.json').read_text())
    m = json.loads((OUT/'manifest.json').read_text())
    audit = json.loads((OUT/'native_audit.json').read_text())
    paths = {'manifest': OUT/'manifest.json', 'native_audit': OUT/'native_audit.json', 'mean_logits': OUT/'mean_logits.npz',
             'statistical_arrays': OUT/'statistical_arrays.npz', 'base_arrays': BASE/'statistical_arrays.npz',
             'word_arrays': WORD/'statistical_arrays.npz', 'test_data': BASE/'test_data.npz'}
    assert s['artifact_sha256'] == {name: sha(path) for name, path in paths.items()}
    assert audit['passed'] and audit['manifest_sha256'] == sha(OUT/'manifest.json')
    expected = {(method, mi, t, seed) for method in range(2) for mi in range(5) for t in p['samples'] for seed in range(10)}
    assert len(m['rows']) == 300 and {(r['method'], r['model'], r['T'], r['seed']) for r in m['rows']} == expected
    arrays = np.load(OUT/'statistical_arrays.npz')
    base = np.load(BASE/'statistical_arrays.npz')
    word = np.load(WORD/'statistical_arrays.npz')
    data = np.load(BASE/'test_data.npz')
    x, y = data['inputs'], data['labels']
    saved = np.load(OUT/'mean_logits.npz')
    originals = json.loads((ROOT/'training/manifest.json').read_text())['rows']
    means = np.empty((2, 5, 2))
    control_errors = []
    for method in range(2):
        for mi in range(5):
            model = checkpoint(mi, method)
            original = np.load(originals[mi]['path'])
            z = np.concatenate([reference(x[first:first+128], model, original) for first in range(0, 10000, 128)])
            q = saved[f'method{method}_model{mi}']
            np.testing.assert_allclose(z, q, rtol=2e-12, atol=2e-11)
            np.testing.assert_array_equal(z.argmax(1), q.argmax(1))
            means[method, mi] = metric(q, y)
            control_errors.append(float(np.max(abs(z-q))))
            print('Verified learned-mean control', method, mi, flush=True)
    np.testing.assert_allclose(means, arrays['mean_metrics'], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(means.mean(1), s['mean_control_summary'], rtol=2e-12, atol=2e-13)
    values = np.empty((2, 3, 5, 10, 2))
    saved_rows = {r['name']: r for r in s['rows']}
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path']) as case:
            v = metric(case['logits'], y)
            assert case['receipts'].shape == (1250, 14, 32)
        bi = p['samples'].index(row['T'])
        values[row['method'], bi, row['model'], row['seed']] = v
        np.testing.assert_allclose(v, [saved_rows[row['name']]['accuracy'], saved_rows[row['name']]['mean_CE']], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(values, arrays['values'], rtol=2e-12, atol=2e-13)
    rng = np.random.default_rng(p['bootstrap_seed'])
    bs = rng.integers(0, 10, size=(2, 5000, 5, 10))
    np.testing.assert_array_equal(bs, arrays['bootstrap_method_seeds'])
    bm = base['bootstrap_models']
    for contrast in s['contrasts']:
        bi = contrast['budget_index']
        name = contrast['contrast']
        method = 1 if name.startswith('sampled') else 0
        left = values[method, bi]
        suffix = name.split('_minus_')[1]
        if suffix == 'original':
            right, right_seeds = base['values'][bi], base['bootstrap_seeds']
        elif suffix == 'expected':
            right, right_seeds = values[0, bi], bs[0]
        elif suffix == 'word':
            right, right_seeds = word['values'][bi], word['bootstrap_word_seeds']
        elif suffix == 'own_mean':
            right, right_seeds = means[method], None
        else:
            right = base['baseline_metrics'][CONTROLS.index('FP32_F64' if suffix == 'FP32' else 'ACT8_INT4_minmax')]
            right_seeds = None
        # Independent accumulation by selected model unit, rather than the
        # producer's broadcast advanced-index expression.
        dist = np.zeros((5000, 2))
        for unit in range(5):
            model_ids = bm[:, unit]
            dist += left[model_ids[:, None], bs[method, :, unit]].mean(1)/5
            if right_seeds is None:
                dist -= right[model_ids]/5
            else:
                dist -= right[model_ids[:, None], right_seeds[:, unit]].mean(1)/5
        point = left.mean((0, 1))-(right.mean(0) if right.ndim == 2 else right.mean((0, 1)))
        np.testing.assert_allclose(dist, arrays[f'{name}_budget{bi}'], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(point, contrast['difference'], rtol=2e-9, atol=3e-12)
        np.testing.assert_allclose(np.quantile(dist, [.025, .975], axis=0).T, contrast['descriptive_95_interval'], rtol=2e-9, atol=3e-12)
    assert len(s['contrasts']) == 33
    for summary in s['summaries']:
        method, bi = summary['method_index'], p['samples'].index(summary['T'])
        np.testing.assert_allclose(values[method, bi].mean((0, 1)), summary['mean'], rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(values[method, bi].mean(1), summary['model_means'], rtol=2e-12, atol=2e-13)
        assert summary['clipped_coordinates'] == np.sum([r['clipped_coordinates'] for r in m['rows'] if r['method'] == method and r['T'] == summary['T']], axis=0).tolist()
    write(OUT/'statistics_audit.json', {'passed': True, 'case_metrics_checked': 300, 'controls_checked': 10,
                                      'control_logits_checked': 1000000, 'bootstrap_values_checked': 330000,
                                      'statistics_sha256': sha(OUT/'statistics.json'), 'max_control_errors': control_errors})
    ledger = {'weight_pbits': 2664, 'activation_pbits': 0, 'deterministic_biases': 34,
              'nominal_parameter_payload_bytes': 3392, 'dynamic_weight_state_bits': 2664,
              'output_counters_per8imagebank': 37746, 'native_counter_bytes_int32': 150984,
              'all300case_raw_receipt_and_logit_bytes': 408000000,
              'production_updates': 447552000000, 'public_replay_updates': 447552000000,
              'scope': 'Fixed-range deployment commands/ranges/bias/activation metadata only, not total stored evidence including training latents, RAM/physical controller/readout/area/energy/latency. Prior training/prerequisite/replay costs separately retained.'}
    write(OUT/'resource_ledger.json', ledger)
    lines = ['# Full native probability-finetuned Fashion-MNIST CNNs', '',
             'All300native cases pass public-state causal recovery and independent controls/metrics/bootstrap checks. Five starting CNNs × two predeclared training methods × three budgets × ten inference seeds. All10000test images are historically consumed; these are descriptive results, not fresh-data confirmation. Training selection used only the already consumed validation set, not these test scores.', '',
             '| Training method | Samples/layer | Accuracy (%) | CE |', '|---|---:|---:|---:|']
    for row in s['summaries']:
        lines.append(f"|{row['method']}|{row['T']}|{100*row['mean'][0]:.4f}|{row['mean'][1]:.8f}|")
    lines += ['', 'All33contrasts/95% descriptive intervals compare both training methods, original single-p-bit inference, FP32, INT4, each learned nominal-mean control and the internal four-node reference at equal node evaluations. Model IDs are paired; the methods and original/word inference seeds are independently resampled within model. Fixed consumed images are not resampled. All methods/budgets/negative outcomes retained; the original frozen-model quality-target miss remains unchanged.', '',
              'Each learned weight retains one free p-bit plus an8-bitfieldcommand; original fixed per-output centers/radii and biases are preserved. ACT8activation spans are the predeclared F32exports. Integer output counters accumulate before externalF64readout/ReLU/maxpool. Training uses a declared biased binomial-mean STE surrogate; its results are not native training or unbiased gradient claims.', '',
              'All14,154,750,000transient integer counters and30millionfinal-logitvalues are checked through full public-state regeneration and exact semantic digests. Hashes cannot decode counter arrays. Production447,552,000,000and public replay447,552,000,000scheduled nodeupdates are separately charged; verification is not a new quality replicate. Earlier validation/preflight/training evidence remains separately retained.', '',
              'Logical parameter payload3392bytes/model and150984bytes of output counters/bank are separate from stochastic states, activation/readout buffers, software evidence/training latents and physical controller costs. No measured systems, energy, latency, area or total memory advantage follows. The four-node word is an internal reference, not an asserted Extropic architecture.', '',
              '[Frozen protocol](fashion_cnn_native/probability_cohort_evidence/protocol.json) · [All statistics](fashion_cnn_native/probability_cohort_evidence/statistics.json) · [Public replay](fashion_cnn_native/probability_cohort_evidence/native_audit.json) · [Statistics audit](fashion_cnn_native/probability_cohort_evidence/statistics_audit.json)', '',
              'The broader long-horizon research goal remains active.']
    report = OUT.parent.parent/'FASHION_CNN_PROBABILITY_FINETUNE_REPORT.md'
    report.write_text('\n'.join(lines)+'\n')
    paths = [OUT/name for name in ['protocol.json', 'manifest.json', 'controls.json', 'mean_logits.npz', 'native_audit.json',
                                  'statistics.json', 'statistical_arrays.npz', 'statistics_audit.json', 'resource_ledger.json']]+[report]
    guard(p)
    write(OUT/'completion.json', {'status': 'full_probability_finetuned_CNN_independently_audited_goal_active',
                                 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'artifact_sha256': {str(path): sha(path) for path in paths},
                                 'production_updates': 447552000000, 'public_replay_updates': 447552000000})
    print('All fine-tuned native CNN controls/statistics audited; full report complete', flush=True)


if __name__ == '__main__':
    {'selfcheck': selfcheck, 'verify': verify}[sys.argv[1]]()
