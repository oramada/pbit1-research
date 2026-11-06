"""Full fine-tuned CNN controls, production, raw replay and descriptive statistics."""
import fcntl
import json
import os
import shutil
import sys
import time

import numpy as np

from results.thrml_spw.fashion_cnn_native.probability_cohort.common import (
    ROOT, OUT, BASE, WORD, METHODS, CONTRASTS, sha, write, save, guard, parent_closed, checkpoint, metrics, keys, contrast_inputs)


def controls():
    from results.thrml_spw.fashion_cnn_native.cohort.common import numpy_control
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert not (OUT/'controls.json').exists()
    assert shutil.disk_usage('.').free >= p['minimum_free_before_controls']
    x = np.load(BASE/'test_data.npz')['inputs']
    data = {}
    for method in range(2):
        for mi in range(5):
            model = checkpoint(mi, method)
            ww = [model[prefix+'_weight'] for prefix in ['conv1', 'conv2', 'head']]
            data[f'method{method}_model{mi}'] = np.concatenate([numpy_control(x[first:first+128], model, ww, True) for first in range(0, 10000, 128)])
    save(OUT/'mean_logits.npz', **data)
    write(OUT/'controls.json', {'complete': True, 'protocol_sha256': sha(OUT/'protocol.json'),
                               'mean_logits_sha256': sha(OUT/'mean_logits.npz'), 'parent_test_data_sha256': sha(BASE/'test_data.npz'),
                               'scope': 'Ten fixed-range nominal learned-mean ACT8 F64 nonlinear controls; not E[finite-sample CNN], optimized low-bit execution or fresh test confirmation.'})


def production():
    import jax
    from importlib.metadata import version
    from results.thrml_spw.fashion_cnn_native.inference.fast_native import forward
    from results.thrml_spw.fashion_cnn_native.probability_native.adapter import encoding
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert jax.default_backend() == 'gpu' and not jax.config.x64_enabled
    assert os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0'
    c = json.loads((OUT/'controls.json').read_text())
    assert c['complete'] and c['mean_logits_sha256'] == sha(OUT/'mean_logits.npz')
    OUT.joinpath('cases').mkdir(exist_ok=True)
    with (OUT/'operation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT/'manifest.json').exists(), 'No automatic duplicate/resume of full production'
        assert shutil.disk_usage('.').free >= p['minimum_free_before_native']
        state = {'complete': False, 'independently_audited': False, 'protocol_sha256': sha(OUT/'protocol.json'),
                 'parent_test_data_sha256': sha(BASE/'test_data.npz'), 'controls_sha256': sha(OUT/'controls.json'),
                 'parent_completion_sha256': sha(ROOT/'probability_native_evidence/completion.json'),
                 'rows': [], 'native_updates': 0, 'backend': jax.default_backend(),
                 'packages': {name: version(name) for name in ['thrml', 'jax', 'jaxlib', 'numpy']}}
        write(OUT/'manifest.json', state)
        x = np.load(BASE/'test_data.npz')['inputs'].astype(float).reshape(1250, 8, 1, 14, 14)
        started = time.monotonic()
        for method in range(2):
            for mi in range(5):
                model = checkpoint(mi, method)
                enc = encoding(model)
                for t in p['samples']:
                    for seed in range(10):
                        guard(p)
                        assert shutil.disk_usage('.').free >= p['minimum_free_before_case']
                        receipts = np.empty((1250, 14, 32), np.uint8)
                        logits = np.empty((10000, 10), float)
                        clipped = np.zeros(3, np.int64)
                        for first in range(0, 1250, 4):
                            n = min(4, 1250-first)
                            kk = [keys(p, method, mi, seed, t, li, first, n) for li in range(3)]
                            raw = forward(x[first:first+n], model, enc, t, kk)
                            receipts[first:first+n] = receipt(raw)
                            logits[first*8:(first+n)*8] = raw['output_2'].reshape(n*8, 10)
                            for li, h in enumerate([x[first:first+n], raw['output_0'], raw['output_1']]):
                                clipped[li] += np.count_nonzero((h < 0) | (h > float(model['activation_span'][li])))
                        assert np.isfinite(logits).all()
                        name = f'method{method}_model{mi}_T{t}_seed{seed}'
                        path = OUT/'cases'/(name+'.npz')
                        save(path, receipts=receipts, logits=logits, clipped_coordinates=clipped)
                        u = 1250*2664*t
                        state['native_updates'] += u
                        state['rows'].append({'name': name, 'method': method, 'model': mi, 'T': t, 'seed': seed,
                                              'path': str(path), 'sha256': sha(path), 'native_updates': u,
                                              'clipped_coordinates': clipped.tolist()})
                        write(OUT/'manifest.json', state)
                        print(json.dumps({'cases': len(state['rows']), 'last': name, 'native_updates': state['native_updates'],
                                          'elapsed_seconds': time.monotonic()-started}), flush=True)
        assert len(state['rows']) == 300 and state['native_updates'] == 447552000000
        guard(p)
        state['complete'] = True
        write(OUT/'manifest.json', state)


def audit():
    import jax
    from results.thrml_spw.fashion_cnn_native.probability_native.replay import recover
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert jax.default_backend() == 'gpu' and not jax.config.x64_enabled
    assert os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0'
    m = json.loads((OUT/'manifest.json').read_text())
    assert m['complete'] and m['protocol_sha256'] == sha(OUT/'protocol.json') and m['parent_test_data_sha256'] == sha(BASE/'test_data.npz')
    with (OUT/'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT/'audit_progress.json').exists(), 'No automatic duplicate/resume of audit'
        state = {'rows': [], 'public_updates': 0, 'counter_entries_exact': 0, 'protocol_sha256': sha(OUT/'protocol.json')}
        x = np.load(BASE/'test_data.npz')['inputs'].astype(float).reshape(1250, 8, 1, 14, 14)
        originals = json.loads((ROOT/'training/manifest.json').read_text())['rows']
        for row in m['rows']:
            guard(p)
            assert sha(row['path']) == row['sha256']
            model = checkpoint(row['model'], row['method'])
            original = np.load(originals[row['model']]['path'])
            case = np.load(row['path'])
            case_receipts = case['receipts']
            case_logits = case['logits']
            case_clipped = case['clipped_coordinates']
            case.close()
            counter_entries = 0
            clipped = np.zeros(3, np.int64)
            for first in range(0, 1250, 4):
                n = min(4, 1250-first)
                kk = [keys(p, row['method'], row['model'], row['seed'], row['T'], li, first, n) for li in range(3)]
                raw = recover(x[first:first+n], model, original, row['T'], kk)
                np.testing.assert_array_equal(receipt(raw), case_receipts[first:first+n])
                np.testing.assert_array_equal(raw['output_2'].reshape(n*8, 10), case_logits[first*8:(first+n)*8])
                counter_entries += sum(raw[f'gate_{li}'].size+raw[f'column_{li}'].size for li in range(3))
                for li, h in enumerate([x[first:first+n], raw['output_0'], raw['output_1']]):
                    clipped[li] += np.count_nonzero((h < 0) | (h > float(model['activation_span'][li])))
            np.testing.assert_array_equal(clipped, row['clipped_coordinates'])
            np.testing.assert_array_equal(clipped, case_clipped)
            assert counter_entries == 47182500
            state['counter_entries_exact'] += counter_entries
            state['public_updates'] += row['native_updates']
            state['rows'].append({'name': row['name'], 'case_sha256': row['sha256'], 'counter_entries_exact': counter_entries,
                                  'all17500_array_receipts_exact': True, 'all100000_final_logits_exact': True,
                                  'public_updates': row['native_updates']})
            write(OUT/'audit_progress.json', state)
            print(json.dumps({'audited_cases': len(state['rows']), 'last': row['name'], 'public_updates': state['public_updates']}), flush=True)
        assert len(state['rows']) == 300 and state['counter_entries_exact'] == 14154750000
        assert state['public_updates'] == m['native_updates'] == 447552000000
        guard(p)
        write(OUT/'native_audit.json', {'passed': True, **state, 'manifest_sha256': sha(OUT/'manifest.json'),
                                      'scope': 'Every transient output counter/code/mask/canonical output regenerated via separate public raw-state replay and semantic digest checked; all final logits directly checked. Hashes do not decode arrays. Audit trajectories are verification, not independent quality replications.'})


def statistics():
    from results.thrml_spw.fashion_cnn_native.cohort.common import CONTROLS
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    m = json.loads((OUT/'manifest.json').read_text())
    a = json.loads((OUT/'native_audit.json').read_text())
    assert a['passed'] and a['manifest_sha256'] == sha(OUT/'manifest.json')
    labels = np.load(BASE/'test_data.npz')['labels']
    base = np.load(BASE/'statistical_arrays.npz')
    word = np.load(WORD/'statistical_arrays.npz')
    plugin = np.load(OUT/'mean_logits.npz')
    means = np.array([[metrics(plugin[f'method{method}_model{mi}'], labels) for mi in range(5)] for method in range(2)])
    values = np.empty((2, 3, 5, 10, 2))
    rows = []
    for row in m['rows']:
        assert sha(row['path']) == row['sha256']
        value = metrics(np.load(row['path'])['logits'], labels)
        values[row['method'], p['samples'].index(row['T']), row['model'], row['seed']] = value
        rows.append({**{k: row[k] for k in ['name', 'method', 'model', 'T', 'seed']}, 'accuracy': float(value[0]), 'mean_CE': float(value[1])})
    bm = base['bootstrap_models']
    rng = np.random.default_rng(p['bootstrap_seed'])
    bs = rng.integers(0, 10, size=(2, 5000, 5, 10))
    word_bs = word['bootstrap_word_seeds']
    arrays = {'values': values, 'mean_metrics': means, 'bootstrap_method_seeds': bs}
    summaries, contrasts = [], []
    for bi, t in enumerate(p['samples']):
        for method in range(2):
            summaries.append({'method': METHODS[method], 'method_index': method, 'T': t,
                              'mean': values[method, bi].mean((0, 1)).tolist(),
                              'model_means': values[method, bi].mean(1).tolist(),
                              'clipped_coordinates': np.sum([r['clipped_coordinates'] for r in m['rows'] if r['method'] == method and r['T'] == t], axis=0).tolist()})
        for name in CONTRASTS:
            left, right = contrast_inputs(name, bi, values, base['values'], word['values'], means, base['baseline_metrics'], CONTROLS)
            method = 1 if name.startswith('sampled') else 0
            draw_l = left[bm[:, :, None], bs[method]].mean((1, 2))
            suffix = name.split('_minus_')[1]
            if right.ndim == 2:
                draw_r = right[bm].mean(1)
                point = left.mean((0, 1))-right.mean(0)
            else:
                seed_indices = base['bootstrap_seeds'] if suffix == 'original' else (bs[0] if suffix == 'expected' else word_bs)
                draw_r = right[bm[:, :, None], seed_indices].mean((1, 2))
                point = left.mean((0, 1))-right.mean((0, 1))
            dist = draw_l-draw_r
            arrays[f'{name}_budget{bi}'] = dist
            contrasts.append({'contrast': name, 'budget_index': bi, 'T': t, 'difference': point.tolist(),
                              'descriptive_95_interval': np.quantile(dist, [.025, .975], axis=0).T.tolist()})
    save(OUT/'statistical_arrays.npz', **arrays)
    paths = {'manifest': OUT/'manifest.json', 'native_audit': OUT/'native_audit.json', 'mean_logits': OUT/'mean_logits.npz',
             'statistical_arrays': OUT/'statistical_arrays.npz', 'base_arrays': BASE/'statistical_arrays.npz',
             'word_arrays': WORD/'statistical_arrays.npz', 'test_data': BASE/'test_data.npz'}
    write(OUT/'statistics.json', {'artifact_sha256': {name: sha(path) for name, path in paths.items()}, 'rows': rows,
                                 'summaries': summaries, 'contrasts': contrasts, 'mean_control_summary': means.mean(1).tolist(),
                                 'metric_order': ['accuracy_fraction', 'mean_CE_nats'], 'scope': p['statistics']})


if __name__ == '__main__':
    {'controls': controls, 'production': production, 'audit': audit, 'statistics': statistics}[sys.argv[1]]()
