"""CPU full-budget learned-command RGB native/public validation; no task scores."""
import datetime
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
from results.thrml_spw.cifar_cnn_native.cohort.common import ROOT, TRAIN, sha, write, save, guard
from results.thrml_spw.fashion_cnn_native.allocated_native.operator import forward
from results.thrml_spw.fashion_cnn_native.inference.native import gate
from results.thrml_spw.fashion_cnn_native.probability_native.adapter import encoding
from results.thrml_spw.fashion_cnn_native.probability_native.replay import recover
from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt

OUT = ROOT/'probability_native_witness_evidence'
FITS = ROOT/'probability_finetune_evidence'


def main():
    assert jax.default_backend() == 'cpu' and not jax.config.x64_enabled
    assert version('thrml') == '0.1.4'
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)
    for path, expected in json.loads((FITS/'completion.json').read_text())['artifact_sha256'].items():
        assert sha(path) == expected, path
    assert json.loads((FITS/'evidence_audit.json').read_text())['passed']
    assert json.loads((FITS/'preparation_root_review.json').read_text())['passed']
    assert not OUT.exists(), 'No duplicate learned-command witness.'
    OUT.mkdir()
    parent = json.loads((ROOT/'word_cohort_evidence/protocol.json').read_text())
    sources = dict(parent['source_sha256'])
    sources.update(json.loads((FITS/'protocol.json').read_text())['source_sha256'])
    extra = [Path(__file__),
             Path('results/thrml_spw/fashion_cnn_native/probability_native/adapter.py'),
             Path('results/thrml_spw/fashion_cnn_native/probability_native/replay.py')]
    sources.update({str(path): sha(path) for path in extra})
    inputs = dict(parent['input_sha256'])
    trained = json.loads((FITS/'manifest.json').read_text())
    assert trained['complete'] and len(trained['rows']) == 10 and not trained['test_scored']
    for path in [FITS/'protocol.json', FITS/'manifest.json', FITS/'completion.json', FITS/'evidence_audit.json', FITS/'preparation_root_review.json', TRAIN/'validation.npz']:
        inputs[str(path)] = sha(path)
    for row in trained['rows']:
        assert sha(row['path']) == row['sha256']
        inputs[row['path']] = row['sha256']
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': sources, 'input_sha256': inputs, 'models': 5,
         'methods': ['expected_weight', 'binomial_mean_STE_T256'], 'samples': [64, 256, 1024],
         'banks': 2, 'bank_size': 8, 'key_root': 1847520, 'weight_nodes': 3928,
         'scope': 'Validation-only CPU full-budget learned-command native witness: all10fits/first16validation inputs/T64,256,1024. Fixed original ranges/biases and learned8bitfields, no re-fitting ranges to expected weights. No labels/task metrics, GPU full-budget proof, fresh-data confirmation, training-gradient equivalence or physical systems claim.'}
    guard(p)
    write(OUT/'protocol.json', p)
    x = np.load(TRAIN/'validation.npz')['inputs'][:16].astype(float).reshape(2, 8, 3, 16, 16)
    originals = json.loads((TRAIN/'manifest.json').read_text())['rows']
    rows, updates, entries = [], 0, 0
    for row in trained['rows']:
        mi, method = row['model'], p['methods'].index(row['method'])
        exported = np.load(row['path'])
        original = np.load(originals[mi]['path'])
        enc = encoding(exported)
        assert sum(e['fields'].size for e in enc) == 3928
        for t in p['samples']:
            root = jax.random.key(p['key_root'])
            for v in [mi, method, t]:
                root = jax.random.fold_in(root, v)
            keys = [jax.vmap(lambda bank: jax.random.fold_in(jax.random.fold_in(root, li), bank))(jnp.arange(2, dtype=jnp.uint32)) for li in range(3)]
            native = forward(x, exported, enc, [t]*3, keys)
            public = recover(x, exported, original, t, keys)
            assert set(native) == set(public) and len(native) == 14
            for name in native:
                np.testing.assert_array_equal(native[name], public[name], err_msg=name)
            np.testing.assert_array_equal(receipt(native), receipt(public))
            for li, e in enumerate(enc):
                ni, no = e['fields'].shape
                bank = gate(ni, no, native[f'gate_{li}'].shape[1]).bank
                assert len(bank.free_indices) == ni*no and not len(bank.clamped_indices)
                assert native[f'gate_{li}'].dtype == native[f'column_{li}'].dtype == np.int32
            count = sum(native[f'gate_{li}'].size+native[f'column_{li}'].size for li in range(3))
            assert count == 98532
            path = OUT/f'model{mi}_{row["method"]}_T{t}.npz'
            save(path, **native)
            updates += 2*3928*t
            entries += count
            rows.append({'model': mi, 'method': row['method'], 'T': t, 'path': str(path), 'sha256': sha(path),
                         'counter_entries_exact': count, 'all28_bank_arrays_exact': True})
            write(OUT/'progress.json', {'rows': rows, 'native_updates': updates, 'public_updates': updates})
            print(json.dumps({'RGB_learned_native_cases': len(rows), 'last': path.name}), flush=True)
    assert len(rows) == 30 and updates == 105584640 and entries == 2955960
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'rows': rows,
          'native_retained_sample_updates': updates, 'public_retained_sample_updates': updates,
          'warmup_updates_per_pass': 235680, 'counter_entries_exact': entries,
          'canonical_bank_arrays_exact': 840, 'all_weight_nodes_free': True,
          'protocol_sha256': sha(OUT/'protocol.json'), 'scope': p['scope']})
    paths = [OUT/'protocol.json', OUT/'evidence_audit.json']+[Path(r['path']) for r in rows]
    write(OUT/'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'status': 'RGBlearned_commands_CPU_native_validation_passed_goal_active',
          'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All30RGBlearned full-budget CPU cases/2955960counters/840bankarrays exact', flush=True)


if __name__ == '__main__':
    main()
