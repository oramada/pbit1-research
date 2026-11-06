"""Validation-only 4096-sample native/public witness before any new task scores."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'probability_high_budget_witness_evidence'
PARENT = ROOT/'probability_cohort_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    assert not path.exists(), path
    path.write_text(json.dumps(data, indent=2)+'\n')


def guard(p):
    for group in ['source_sha256', 'input_sha256']:
        for path, digest in p[group].items():
            assert sha(path) == digest, path


def prepare():
    assert not OUT.exists()
    parent = json.loads((PARENT/'protocol.json').read_text())
    sources = dict(parent['source_sha256'])
    for path in Path(__file__).parent.glob('*.py'):
        sources[str(path)] = sha(path)
        assert (Path('results/thrml_spw/source_snapshots')/(sha(path)+'.py')).exists()
    inputs = dict(parent['input_sha256'])
    for name in ['completion.json', 'review_completion.json', 'root_review.json', 'protocol.json', 'pipeline.json']:
        inputs[str(PARENT/name)] = sha(PARENT/name)
    inputs[str(ROOT/'training/validation.npz')] = sha(ROOT/'training/validation.npz')
    for name in ['completion.json', 'review_completion.json']:
        record = json.loads((PARENT/name).read_text())
        for path, digest in record['artifact_sha256'].items():
            assert sha(path) == digest, path
    OUT.mkdir()
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': sources, 'input_sha256': inputs, 'samples': 4096,
         'key_root': 1887520, 'models': 5, 'methods': 2, 'cases': 10,
         'validation_images': 16, 'bank_size': 8, 'banks': 2,
         'retained_node_updates_per_pass': 321781760, 'warmup_node_updates_per_pass': 78560,
         'counter_entries_per_pass': 985320, 'minimum_free_bytes': 100000000,
         'scope': 'Validation inputs only, no labels or task accuracy. T4096 chosen after consumed T1024 study to test temporal variance reduction with one free p-bit per weight. Unchanged training commands/ranges/biases/ACT8 spans. New operator changes only the maximum sample bound; IntegerGateLayer retains per-layer signed32 overflow guard. Public raw-state replay independently constructs commands, patches, counts and readout. No BF16 equivalence, independent task replica or measured hardware cost claim.'}
    guard(p)
    write(OUT/'protocol.json', p)
    print('High-budget validation protocol sealed; no task scoring')


def run():
    import jax
    from importlib.metadata import version
    from results.thrml_spw.cifar_cnn_native.probability_high_budget.operator import forward
    from results.thrml_spw.fashion_cnn_native.probability_native.adapter import encoding
    from results.thrml_spw.fashion_cnn_native.probability_native.replay import recover
    from results.thrml_spw.cifar_cnn_native.probability_cohort.common import checkpoint, keys
    import shutil
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'completion.json').exists() and not (OUT/'cases').exists()
    assert version('thrml') == '0.1.4' and jax.default_backend() == 'gpu' and not jax.config.x64_enabled
    assert os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0'
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)
    assert shutil.disk_usage('.').free >= p['minimum_free_bytes']
    (OUT/'cases').mkdir()
    x = np.load(ROOT/'training/validation.npz')['inputs'][:16].astype(float).reshape(2, 8, 3, 16, 16)
    originals = json.loads((ROOT/'training/manifest.json').read_text())['rows']
    rows, updates, entries = [], 0, 0
    for method in range(2):
        for mi in range(5):
            guard(p)
            model, original = checkpoint(mi, method), np.load(originals[mi]['path'])
            enc = encoding(model)
            ni = [e['fields'].shape[0] for e in enc]
            assert sum(e['fields'].size for e in enc) == 3928
            bounds = [255*n*p['samples'] for n in ni]
            assert all(b < 2**31 for b in bounds)
            kk = [keys(p, method, mi, 0, p['samples'], layer, 0, 2) for layer in range(3)]
            actual = forward(x, model, enc, [p['samples']]*3, kk)
            public = recover(x, model, original, p['samples'], kk)
            assert set(actual) == set(public) and len(actual) == 14
            for name in actual:
                np.testing.assert_array_equal(actual[name], public[name])
            count = sum(actual[f'gate_{li}'].size+actual[f'column_{li}'].size for li in range(3))
            assert count == 98532
            for li, bound in enumerate(bounds):
                assert actual[f'gate_{li}'].dtype == np.int32
                assert np.min(actual[f'gate_{li}']) >= 0 and np.max(actual[f'gate_{li}']) <= bound
            path = OUT/'cases'/f'method{method}_model{mi}_T4096.npz'
            np.savez_compressed(path, **actual)
            updates += 2*3928*p['samples']
            entries += count
            rows.append({'method': method, 'model': mi, 'path': str(path), 'sha256': sha(path),
                         'all14arrays_exact': True, 'counter_entries_exact': count,
                         'input_dimensions': ni, 'signed32_worst_case_counter_bounds': bounds,
                         'retained_node_updates_per_pass': 2*3928*p['samples']})
            write(OUT/f'case_{len(rows)}.json', rows[-1])
            print('High-budget native/public validation cases', len(rows), flush=True)
    assert len(rows) == p['cases'] and updates == p['retained_node_updates_per_pass'] and entries == p['counter_entries_per_pass']
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'rows': rows, 'protocol_sha256': sha(OUT/'protocol.json'),
          'counter_entries_exact': entries, 'native_updates': updates, 'public_updates': updates,
          'warmup_updates_per_pass': p['warmup_node_updates_per_pass'], 'scope': p['scope']})
    paths = [OUT/'protocol.json', OUT/'evidence_audit.json']+[Path(r['path']) for r in rows]
    write(OUT/'completion.json', {'passed': True, 'status': 'all10_T4096_validation_cases_exact_goal_active',
          'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All10T4096 cases/985320counters/140arrays native-public exact; no task scores')


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
