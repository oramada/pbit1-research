"""Native production/public replay derivative for the fixed T4096 cohort."""
import fcntl
import json
import os
import shutil
import sys
import time
import numpy as np
from results.thrml_spw.cifar_cnn_native.probability_high_budget.common import (
    ROOT, OUT, PARENT, WITNESS, BASE, METHODS, sha, write, save, guard, parent_closed, checkpoint, keys)
from results.thrml_spw.cifar_cnn_native.probability_cohort.run import backend


def controls():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert not (OUT/'controls.json').exists()
    assert shutil.disk_usage('.').free >= p['minimum_free_before_native']
    source = PARENT/'mean_logits.npz'
    shutil.copyfile(source, OUT/'mean_logits.npz')
    assert sha(source) == sha(OUT/'mean_logits.npz')
    write(OUT/'controls.json', {'complete': True, 'protocol_sha256': sha(OUT/'protocol.json'),
        'mean_logits_sha256': sha(OUT/'mean_logits.npz'), 'original_controls_sha256': sha(PARENT/'controls.json'),
        'scope': 'Byte-identical previously independently audited means for unchanged ten models. No new numerical graph or task replicate.'})


def production():
    backend()
    import jax
    from importlib.metadata import version
    from results.thrml_spw.cifar_cnn_native.probability_high_budget.operator import forward
    from results.thrml_spw.fashion_cnn_native.probability_native.adapter import encoding
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert jax.default_backend() == 'gpu' and not jax.config.x64_enabled
    assert os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0'
    pre = json.loads((WITNESS/'evidence_audit.json').read_text())
    assert pre['passed'] and sha(WITNESS/'protocol.json') == p['witness_protocol_sha256']
    c = json.loads((OUT/'controls.json').read_text())
    assert c['complete'] and c['mean_logits_sha256'] == sha(OUT/'mean_logits.npz')
    OUT.joinpath('cases').mkdir(exist_ok=True)
    with (OUT/'operation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT/'manifest.json').exists(), 'No automatic duplicate/resume of full production'
        assert shutil.disk_usage('.').free >= p['minimum_free_before_native']
        state = {'complete': False, 'independently_audited': False, 'protocol_sha256': sha(OUT/'protocol.json'),
                 'parent_test_data_sha256': sha(BASE/'test_data.npz'), 'controls_sha256': sha(OUT/'controls.json'),
                 'parent_completion_sha256': sha(PARENT/'completion.json'),
                 'rows': [], 'native_updates': 0, 'backend': jax.default_backend(),
                 'packages': {name: version(name) for name in ['thrml', 'jax', 'jaxlib', 'numpy']}}
        write(OUT/'manifest.json', state)
        x = np.load(BASE/'test_data.npz')['inputs'].astype(float).reshape(1250, 8, 3, 16, 16)
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
                            raw = forward(x[first:first+n], model, enc, [t]*3, kk)
                            receipts[first:first+n] = receipt(raw)
                            logits[first*8:(first+n)*8] = raw['output_2'].reshape(n*8, 10)
                            for li, h in enumerate([x[first:first+n], raw['output_0'], raw['output_1']]):
                                clipped[li] += np.count_nonzero((h < 0) | (h > float(model['activation_span'][li])))
                        assert np.isfinite(logits).all()
                        name = f'method{method}_model{mi}_T{t}_seed{seed}'
                        path = OUT/'cases'/(name+'.npz')
                        save(path, receipts=receipts, logits=logits, clipped_coordinates=clipped)
                        u = 1250*3928*t
                        state['native_updates'] += u
                        state['rows'].append({'name': name, 'method': method, 'model': mi, 'T': t, 'seed': seed,
                                              'path': str(path), 'sha256': sha(path), 'native_updates': u,
                                              'clipped_coordinates': clipped.tolist()})
                        write(OUT/'manifest.json', state)
                        print(json.dumps({'cases': len(state['rows']), 'last': name, 'native_updates': state['native_updates'],
                                          'elapsed_seconds': time.monotonic()-started}), flush=True)
        assert len(state['rows']) == 100 and state['native_updates'] == 2011136000000
        guard(p)
        state['complete'] = True
        write(OUT/'manifest.json', state)


def audit():
    backend()
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
        x = np.load(BASE/'test_data.npz')['inputs'].astype(float).reshape(1250, 8, 3, 16, 16)
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
            assert counter_entries == 61582500
            state['counter_entries_exact'] += counter_entries
            state['public_updates'] += row['native_updates']
            state['rows'].append({'name': row['name'], 'case_sha256': row['sha256'], 'counter_entries_exact': counter_entries,
                                  'all17500_array_receipts_exact': True, 'all100000_final_logits_exact': True,
                                  'public_updates': row['native_updates']})
            write(OUT/'audit_progress.json', state)
            print(json.dumps({'audited_cases': len(state['rows']), 'last': row['name'], 'public_updates': state['public_updates']}), flush=True)
        assert len(state['rows']) == 100 and state['counter_entries_exact'] == 6158250000
        assert state['public_updates'] == m['native_updates'] == 2011136000000
        guard(p)
        write(OUT/'native_audit.json', {'passed': True, **state, 'manifest_sha256': sha(OUT/'manifest.json'),
                                      'scope': 'Every transient output counter/code/mask/canonical output regenerated via separate public raw-state replay and semantic digest checked; all final logits directly checked. Hashes do not decode arrays. Audit trajectories are verification, not independent quality replications.'})


if __name__ == '__main__':
    {'controls': controls, 'production': production, 'audit': audit}[sys.argv[1]]()
