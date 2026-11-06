"""Freeze and run all ten same-fit positional-word validation witnesses."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from results.thrml_spw.cifar_cnn_native.probability_cohort.common import (
    ROOT, checkpoint, sha, write, save, guard, seal)

OUT = ROOT/'learned_word_witness_evidence'
HIGH = ROOT/'probability_high_budget_evidence'
SAME = ROOT/'same_fit_bf16_evidence'


def closed():
    for folder in [HIGH, SAME]:
        seal(folder/'completion.json')
        seal(folder/'review_completion.json')


def prepare():
    assert not OUT.exists()
    closed()
    parent = json.loads((SAME/'protocol.json').read_text())
    sources, inputs = dict(parent['source_sha256']), dict(parent['input_sha256'])
    files = list(Path(__file__).parent.glob('*.py'))+[
        Path('results/thrml_spw/fashion_cnn_native/word_reference/native.py'),
        Path('results/thrml_spw/fashion_cnn_native/compact/recover.py'),
        Path('results/thrml_spw/fashion_cnn_native/cohort/replay.py')]
    for path in files:
        sources[str(path)] = sha(path)
        assert (ROOT.parent/'source_snapshots'/(sha(path)+'.py')).exists()
    for folder in [HIGH, SAME]:
        for name in ['completion.json', 'review_completion.json']:
            inputs[str(folder/name)] = sha(folder/name)
            inputs.update(json.loads((folder/name).read_text())['artifact_sha256'])
    inputs[str(ROOT/'training/validation.npz')] = sha(ROOT/'training/validation.npz')
    protocol = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': sources, 'input_sha256': inputs, 'models': 5, 'methods': 2,
        'word_samples': [16, 64, 256, 1024], 'matched_single_samples': [64, 256, 1024, 4096],
        'cases': 40, 'banks': 2, 'bank_size': 8, 'validation_images': 16,
        'key_root': 1917520, 'retained_updates_per_pass': 427366400,
        'warmup_updates_per_pass': 1256960, 'counter_entries': 3941280,
        'scope': 'Validation-only internal four-free-node positional reference for the same ten learned fits. Original ranges, biases and ACT8 spans preserved. Each learned stationary mean rounded to16levels in original range; four free finite fields plus/minus4 with coefficients1,2,4,8 denominator15. No asserted Extropic weight architecture. Native observer accumulates integer output counters; independent public-state replay reconstructs original ranges and learned means from commands, then word fields/patches/counters/readout. No labels, test accuracy, systems advantage or formal BF16 equivalence. Precedes any full matched-work task execution. Scheduled evaluations are not physical flips or energy.'}
    assert 10*2*15712*sum(protocol['word_samples']) == protocol['retained_updates_per_pass']
    assert 40*2*15712 == protocol['warmup_updates_per_pass']
    OUT.mkdir(); guard(protocol); write(OUT/'protocol.json', protocol)
    print('All 40 same-fit word validation cases frozen; no labels or test scoring')


def run():
    import jax
    import jax.numpy as jnp
    from importlib.metadata import version
    from results.thrml_spw.fashion_cnn_native.word_reference.native import forward
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    from results.thrml_spw.cifar_cnn_native.learned_word.adapter import encode
    from results.thrml_spw.cifar_cnn_native.learned_word.replay import recover
    p = json.loads((OUT/'protocol.json').read_text()); guard(p); closed()
    assert not (OUT/'cases').exists() and not (OUT/'completion.json').exists()
    assert version('thrml') == '0.1.4' and jax.default_backend() == 'gpu' and not jax.config.x64_enabled
    assert os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0'
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)
    assert shutil.disk_usage('.').free > 100000000
    # Detect actual workers; a historical launch/state file is not a live worker.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            if b'.venv-thrml-gpu/bin/python' in argv[0] and b'-m' in argv:
                module = argv[argv.index(b'-m')+1]
                assert not module.startswith((b'results.thrml_spw.', b'research.thrml_spw.')), (proc.name, module)
        except (FileNotFoundError, ProcessLookupError):
            continue
    x = np.load(ROOT/'training/validation.npz')['inputs'][:16].astype(float).reshape(2, 8, 3, 16, 16)
    originals = json.loads((ROOT/'training/manifest.json').read_text())['rows']
    (OUT/'cases').mkdir(); rows = []; updates = entries = 0
    for method in range(2):
        for mi in range(5):
            model = checkpoint(mi, method); original = np.load(originals[mi]['path']); enc = encode(model)
            assert sum(e['fields'].size for e in enc) == 15712
            for t in p['word_samples']:
                guard(p)
                key = jax.random.key(p['key_root'])
                for value in [method, mi, t]:
                    key = jax.random.fold_in(key, value)
                keys = [jax.vmap(lambda bank: jax.random.fold_in(jax.random.fold_in(key, li), bank))(
                    jnp.arange(2, dtype=jnp.uint32)) for li in range(3)]
                bounds = [255*e['fields'].shape[0]*15*t for e in enc]
                assert all(bound < 2**31 for bound in bounds)
                native = forward(x, model, enc, t, keys)
                public = recover(x, model, original, t, keys, enc)
                assert set(native) == set(public) and len(native) == 14
                for name in native:
                    np.testing.assert_array_equal(native[name], public[name])
                np.testing.assert_array_equal(receipt(native), receipt(public))
                for li, bound in enumerate(bounds):
                    assert native[f'gate_{li}'].dtype == np.int32
                    assert np.min(native[f'gate_{li}']) >= 0 and np.max(native[f'gate_{li}']) <= bound
                count = sum(native[f'gate_{li}'].size+native[f'column_{li}'].size for li in range(3))
                assert count == 98532
                path = OUT/'cases'/f'method{method}_model{mi}_T{t}.npz'; save(path, **native)
                updates += 2*15712*t; entries += count
                rows.append({'method': method, 'model': mi, 'word_T': t, 'matched_single_T': 4*t,
                    'path': str(path), 'sha256': sha(path), 'counter_entries_exact': count,
                    'all14arrays_exact': True, 'counter_bounds': bounds})
                write(OUT/'progress.json', {'rows': rows, 'retained_updates_per_pass': updates,
                    'counter_entries': entries})
                print(json.dumps({'validated_cases': len(rows), 'word_T': t}), flush=True)
    assert len(rows) == 40 and entries == p['counter_entries'] and updates == p['retained_updates_per_pass']
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'rows': rows,
        'counter_entries_exact': entries, 'bank_arrays_exact': 1120,
        'native_updates': updates, 'public_updates': updates,
        'warmup_updates_per_pass': p['warmup_updates_per_pass'], 'protocol_sha256': sha(OUT/'protocol.json'),
        'labels_used': False, 'scope': p['scope']})
    paths = [OUT/'protocol.json', OUT/'evidence_audit.json']+[Path(row['path']) for row in rows]
    write(OUT/'completion.json', {'passed': True, 'status': 'same_fit_word_validation_passed_goal_active',
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'artifact_sha256': {str(path): sha(path) for path in paths}})


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
