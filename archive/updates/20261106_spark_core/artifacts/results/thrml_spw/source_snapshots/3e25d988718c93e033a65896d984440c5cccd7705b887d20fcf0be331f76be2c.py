"""Validation-only native/public RGB witness for the internal four-node reference."""
import datetime
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
from results.thrml_spw.cifar_cnn_native.cohort.common import ROOT, TRAIN, sha, write, save, guard, preparation_closed
from results.thrml_spw.fashion_cnn_native.word_reference.native import encode, forward
from results.thrml_spw.fashion_cnn_native.word_reference.witness import recover
from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt

OUT = ROOT/'word_witness_evidence'


def main():
    assert jax.default_backend() == 'cpu' and not jax.config.x64_enabled
    assert version('thrml') == '0.1.4'
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)
    preparation_closed()
    assert not OUT.exists(), 'No duplicate witness; preserve earlier evidence.'
    OUT.mkdir()
    parent = json.loads((ROOT/'cohort_evidence/protocol.json').read_text())
    sources = dict(parent['source_sha256'])
    extra = [Path(__file__),
             Path('results/thrml_spw/fashion_cnn_native/word_reference/native.py'),
             Path('results/thrml_spw/fashion_cnn_native/word_reference/witness.py')]
    sources.update({str(path): sha(path) for path in extra})
    inputs = dict(parent['input_sha256'])
    inputs.update({str(TRAIN/'validation.npz'): sha(TRAIN/'validation.npz')})
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': sources, 'input_sha256': inputs,
         'models': 5, 'word_samples': [1, 4, 16], 'banks': 2, 'bank_size': 8,
         'key_root': 1767520, 'K': 4, 'weight_nodes': 15712,
         'field_values': [-4, 4], 'coefficients': [1, 2, 4, 8], 'denominator': 15,
         'future_matched_single_samples': [64, 256, 1024],
         'future_matched_word_samples': [16, 64, 256],
         'scope': 'Validation-only bounded RGB correctness witness, all5models/first16validation inputs. Internal four-free-node positional reference; not an asserted Extropic architecture, full-budget GPU validation, task quality, or physical systems result.'}
    guard(p)
    write(OUT/'protocol.json', p)
    x = np.load(TRAIN/'validation.npz')['inputs'][:16].astype(float).reshape(2, 8, 3, 16, 16)
    rows, updates, entries, arrays = [], 0, 0, 0
    for mi, row in enumerate(json.loads((TRAIN/'manifest.json').read_text())['rows']):
        model = np.load(row['path'])
        enc = encode(model)
        assert sum(e['fields'].size for e in enc) == 15712
        for t in p['word_samples']:
            root = jax.random.key(p['key_root'])
            for v in [mi, t]:
                root = jax.random.fold_in(root, v)
            keys = [jax.vmap(lambda bank: jax.random.fold_in(jax.random.fold_in(root, li), bank))(
                    jnp.arange(2, dtype=jnp.uint32)) for li in range(3)]
            native = forward(x, model, enc, t, keys)
            public = recover(x, model, t, keys, enc)
            assert set(native) == set(public) and len(native) == 14
            for name in native:
                np.testing.assert_array_equal(native[name], public[name])
            np.testing.assert_array_equal(receipt(native), receipt(public))
            for li in range(3):
                assert native[f'gate_{li}'].dtype == np.int32
                assert native[f'column_{li}'].dtype == np.int32
            path = OUT/f'model{mi}_T{t}.npz'
            save(path, **native)
            n = sum(native[f'gate_{li}'].size+native[f'column_{li}'].size for li in range(3))
            entries += n
            arrays += 2*len(native)
            updates += 2*15712*t
            rows.append({'model': mi, 'T_word': t, 'path': str(path), 'sha256': sha(path),
                         'counter_entries_exact': n})
            write(OUT/'progress.json', {'complete': False, 'rows': rows})
            print(json.dumps({'validation_word_cases': len(rows), 'last': path.name}), flush=True)
    assert len(rows) == 15 and entries == 1477980 and arrays == 420 and updates == 3299520
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'rows': rows,
          'native_retained_sample_updates': updates, 'public_retained_sample_updates': updates,
          'warmup_updates_per_pass': 471360, 'counter_entries_exact': entries,
          'canonical_bank_arrays_exact': arrays, 'protocol_sha256': sha(OUT/'protocol.json'),
          'largest_gate_bound_here': 255*256*15*16,
          'scope': p['scope']})
    write(OUT/'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'status': 'bounded_RGB_word_witness_passed_goal_active',
          'artifact_sha256': {str(path): sha(path) for path in [OUT/'protocol.json', OUT/'evidence_audit.json']}})
    print('All15boundedRGBwordcases/1477980counters/420bankarrays exact', flush=True)


if __name__ == '__main__':
    main()
