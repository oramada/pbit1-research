"""Full-budget validation and matched RGB word production/public regeneration."""
import json
import os
import shutil
import subprocess
import sys
import numpy as np
from results.thrml_spw.cifar_cnn_native.word_cohort.common import ROOT, TRAIN, OUT, PARENT, sha, write, save, guard, parent_closed, keys


def backend():
    import jax
    from importlib.metadata import version
    assert version('thrml') == '0.1.4' and jax.default_backend() == 'gpu'
    assert not jax.config.x64_enabled and os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0'
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)


def preflight():
    from results.thrml_spw.fashion_cnn_native.word_reference.native import encode, forward
    from results.thrml_spw.fashion_cnn_native.word_reference.witness import recover
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    backend()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert not (OUT/'preflight.json').exists()
    dest = OUT/'preflight_cases'
    dest.mkdir()
    x = np.load(TRAIN/'validation.npz')['inputs'][:16].astype(float).reshape(2, 8, 3, 16, 16)
    rows, updates, entries = [], 0, 0
    for mi, row in enumerate(json.loads((TRAIN/'manifest.json').read_text())['rows']):
        model = np.load(row['path'])
        enc = encode(model)
        for t in p['word_samples']:
            kk = [keys(0, mi, 0, t, li, 0, 2) for li in range(3)]
            a, b = forward(x, model, enc, t, kk), recover(x, model, t, kk, enc)
            assert set(a) == set(b) and len(a) == 14
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])
            np.testing.assert_array_equal(receipt(a), receipt(b))
            path = dest/f'model{mi}_T{t}.npz'
            save(path, **a)
            n = sum(a[f'gate_{li}'].size+a[f'column_{li}'].size for li in range(3))
            entries += n
            updates += 2*15712*t
            rows.append({'model': mi, 'T_word': t, 'path': str(path), 'sha256': sha(path), 'counter_entries_exact': n})
            print(json.dumps({'GPU_word_preflight_cases': len(rows)}), flush=True)
    assert len(rows) == 15 and entries == 1477980 and updates == 52792320
    guard(p)
    write(OUT/'preflight.json', {'passed': True, 'rows': rows, 'counter_entries_exact': entries,
          'native_updates': updates, 'public_updates': updates, 'warmup_updates_per_pass': 471360,
          'protocol_sha256': sha(OUT/'protocol.json')})


def run(audit=False):
    from results.thrml_spw.fashion_cnn_native.word_reference.native import encode, forward
    from results.thrml_spw.fashion_cnn_native.word_reference.witness import recover
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    backend()
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    pre = json.loads((OUT/'preflight.json').read_text())
    assert pre['passed'] and pre['protocol_sha256'] == sha(OUT/'protocol.json')
    controls = json.loads((OUT/'controls.json').read_text())
    assert controls['complete'] and controls['parent_test_data_sha256'] == sha(PARENT/'test_data.npz')
    assert controls['mean_logits_sha256'] == sha(OUT/'mean_logits.npz')
    destination = OUT/('native_audit.json' if audit else 'manifest.json')
    assert not destination.exists(), 'No implicit restart or duplicate execution.'
    assert shutil.disk_usage('.').free >= p['case_minimum_free_bytes']
    x = np.load(PARENT/'test_data.npz')['inputs'].astype(float).reshape(1250, 8, 3, 16, 16)
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    if audit:
        m = json.loads((OUT/'manifest.json').read_text())
        assert m['complete'] and len(m['rows']) == 150
        cases = m['rows']
    else:
        (OUT/'cases').mkdir()
        cases = [{'model': mi, 'T': t, 'seed': seed} for mi in range(5) for t in p['word_samples'] for seed in range(10)]
    state = {'complete': False, 'rows': [], 'scheduled_updates': 0, 'counter_entries': 0,
             'protocol_sha256': sha(OUT/'protocol.json'), 'parent_completion_sha256': sha(PARENT/'completion.json')}
    for row in cases:
        guard(p)
        assert shutil.disk_usage('.').free >= p['case_minimum_free_bytes']
        mi, t, seed = row['model'], row['T'], row['seed']
        model = np.load(train[mi]['path'])
        enc = encode(model)
        name = f'word_model{mi}_T{t}_seed{seed}'
        path = OUT/'cases'/(name+'.npz')
        if audit:
            assert sha(path) == row['sha256']
            saved = np.load(path)
        hashes = np.empty((1250, 14, 32), np.uint8)
        logits = np.empty((10000, 10), np.float64)
        clipped = np.zeros(3, np.int64)
        for first in range(0, 1250, 4):
            count = min(4, 1250-first)
            kk = [keys(1, mi, seed, t, li, first, count) for li in range(3)]
            raw = recover(x[first:first+count], model, t, kk, enc) if audit else forward(x[first:first+count], model, enc, t, kk)
            hashes[first:first+count] = receipt(raw)
            logits[first*8:(first+count)*8] = raw['output_2'].reshape(count*8, 10)
            state['counter_entries'] += sum(raw[f'gate_{li}'].size+raw[f'column_{li}'].size for li in range(3))
            for li, h in enumerate([x[first:first+count], raw['output_0'], raw['output_1']]):
                clipped[li] += np.count_nonzero((h < 0) | (h > float(model['activation_span'][li])))
        if audit:
            for k, v in [('receipts', hashes), ('logits', logits), ('clipped_coordinates', clipped)]:
                np.testing.assert_array_equal(v, saved[k])
        else:
            assert np.isfinite(logits).all()
            save(path, receipts=hashes, logits=logits, clipped_coordinates=clipped)
        state['scheduled_updates'] += 1250*15712*t
        state['rows'].append({'model': mi, 'T': t, 'seed': seed, 'matched_single_T': 4*t,
              'name': name, 'path': str(path), 'sha256': sha(path), 'clipped_coordinates': clipped.tolist(),
              'all_receipts_and_final_logits_exact': bool(audit)})
        write(OUT/'audit_progress.json' if audit else destination, state)
        print(json.dumps({'audited' if audit else 'produced': len(state['rows']), 'last': name}), flush=True)
    assert len(state['rows']) == 150 and state['counter_entries'] == 9237375000
    assert state['scheduled_updates'] == 329952000000
    state['complete'] = True
    if audit:
        state.update(passed=True, manifest_sha256=sha(OUT/'manifest.json'))
    guard(p)
    write(destination, state)


if __name__ == '__main__':
    {'preflight': preflight, 'run': run, 'audit': lambda: run(True)}[sys.argv[1]]()
