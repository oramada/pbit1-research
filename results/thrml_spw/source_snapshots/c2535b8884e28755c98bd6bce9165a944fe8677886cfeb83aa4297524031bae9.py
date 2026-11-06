"""Same-fit word controls, native production and exact public-state replay."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import numpy as np
from results.thrml_spw.cifar_cnn_native.learned_word_cohort.common import (
    ROOT, OUT, BASE, checkpoint, keys, sha, write, save, guard, closed)


def controls():
    from results.thrml_spw.cifar_cnn_native.learned_word_v2.adapter import mean_weights
    from results.thrml_spw.fashion_cnn_native.cohort.common import numpy_control
    p = json.loads((OUT/'protocol.json').read_text()); guard(p); closed()
    assert not (OUT/'controls.json').exists()
    x = np.load(BASE/'test_data.npz')['inputs']; logits = {}; weights = {}
    for method in range(2):
        for mi in range(5):
            model = checkpoint(mi, method); ww = mean_weights(model)
            logits[f'method{method}_model{mi}'] = np.concatenate([
                numpy_control(x[first:first+128], model, ww, True) for first in range(0, len(x), 128)])
            for li, w in enumerate(ww):
                weights[f'method{method}_model{mi}_layer{li}'] = w
    save(OUT/'mean_logits.npz', **logits); save(OUT/'mean_weights.npz', **weights)
    guard(p)
    write(OUT/'controls.json', {'complete': True, 'protocol_sha256': sha(OUT/'protocol.json'),
        'mean_logits_sha256': sha(OUT/'mean_logits.npz'), 'mean_weights_sha256': sha(OUT/'mean_weights.npz'),
        'scope': 'Ten finite-field nominal-word-mean ACT8 F64 nonlinear controls, not E[finite-sample CNN].'})


def run(audit=False):
    import jax
    from importlib.metadata import version
    from results.thrml_spw.fashion_cnn_native.word_reference.native import forward
    from results.thrml_spw.cifar_cnn_native.learned_word_v2.adapter import encode
    from results.thrml_spw.cifar_cnn_native.learned_word_v2.replay import recover
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    assert version('thrml') == '0.1.4' and jax.default_backend() == 'gpu' and not jax.config.x64_enabled
    assert os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0'
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)
    p = json.loads((OUT/'protocol.json').read_text()); guard(p); closed()
    c = json.loads((OUT/'controls.json').read_text())
    assert c['complete'] and c['protocol_sha256'] == sha(OUT/'protocol.json')
    assert c['mean_logits_sha256'] == sha(OUT/'mean_logits.npz') and c['mean_weights_sha256'] == sha(OUT/'mean_weights.npz')
    destination = OUT/('native_audit.json' if audit else 'manifest.json')
    assert not destination.exists() and (not audit or not (OUT/'audit_progress.json').exists()), 'No automatic resume/duplicate execution'
    x = np.load(BASE/'test_data.npz')['inputs'].astype(float).reshape(1250, 8, 3, 16, 16)
    originals = json.loads((ROOT/'training/manifest.json').read_text())['rows']
    with (OUT/('audit.lock' if audit else 'production.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if audit:
            manifest = json.loads((OUT/'manifest.json').read_text())
            assert manifest['complete'] and len(manifest['rows']) == 400
            cases = manifest['rows']
        else:
            assert not (OUT/'cases').exists(); (OUT/'cases').mkdir()
            cases = [{'method': m, 'model': mi, 'T': t, 'seed': seed}
                for m in range(2) for mi in range(5) for t in p['word_samples'] for seed in range(10)]
        state = {'complete': False, 'rows': [], 'scheduled_updates': 0, 'counter_entries': 0,
            'protocol_sha256': sha(OUT/'protocol.json'), 'independently_audited': False}
        for row in cases:
            guard(p); assert shutil.disk_usage('.').free >= p['minimum_free_before_case']
            method, mi, t, seed = row['method'], row['model'], row['T'], row['seed']
            model = checkpoint(mi, method); original = np.load(originals[mi]['path']); enc = encode(model)
            assert sum(e['fields'].size for e in enc) == 15712
            name = f'word_method{method}_model{mi}_T{t}_seed{seed}'; path = OUT/'cases'/(name+'.npz')
            if audit:
                assert sha(path) == row['sha256']
            hashes = np.empty((1250, 14, 32), np.uint8); logits = np.empty((10000, 10), np.float64)
            clipped = np.zeros(3, np.int64)
            for first in range(0, 1250, 4):
                count = min(4, 1250-first)
                kk = [keys(p, method, mi, seed, t, li, first, count) for li in range(3)]
                raw = recover(x[first:first+count], model, original, t, kk, enc) if audit else forward(x[first:first+count], model, enc, t, kk)
                hashes[first:first+count] = receipt(raw)
                logits[first*8:(first+count)*8] = raw['output_2'].reshape(count*8, 10)
                state['counter_entries'] += sum(raw[f'gate_{li}'].size+raw[f'column_{li}'].size for li in range(3))
                for li, h in enumerate([x[first:first+count], raw['output_0'], raw['output_1']]):
                    clipped[li] += np.count_nonzero((h < 0) | (h > float(model['activation_span'][li])))
            assert np.isfinite(logits).all()
            if audit:
                with np.load(path) as saved:
                    for key, value in [('receipts', hashes), ('logits', logits), ('clipped_coordinates', clipped)]:
                        np.testing.assert_array_equal(saved[key], value)
            else:
                save(path, receipts=hashes, logits=logits, clipped_coordinates=clipped)
            state['scheduled_updates'] += 1250*15712*t
            state['rows'].append({'method': method, 'model': mi, 'T': t, 'seed': seed,
                'matched_single_T': 4*t, 'name': name, 'path': str(path), 'sha256': sha(path),
                'clipped_coordinates': clipped.tolist(), 'all_receipts_and_final_logits_exact': bool(audit)})
            write(OUT/'audit_progress.json' if audit else destination, state)
            print(json.dumps({'audited' if audit else 'produced': len(state['rows']), 'last': name,
                'scheduled_updates': state['scheduled_updates']}), flush=True)
        assert len(state['rows']) == 400 and state['counter_entries'] == p['counter_entries']
        assert state['scheduled_updates'] == p['retained_updates_per_pass']
        state['complete'] = True
        if audit:
            state.update(passed=True, manifest_sha256=sha(OUT/'manifest.json'))
        guard(p); write(destination, state)


if __name__ == '__main__':
    {'controls': controls, 'production': run, 'audit': lambda: run(True)}[sys.argv[1]]()
