"""All-five-checkpoint RGB native/public operator witness; no task labels."""
import datetime
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import numpy as np
import jax
from results.thrml_spw.fashion_cnn_native.cohort.common import sha, write, guard
from results.thrml_spw.fashion_cnn_native.inference.native import encode
from results.thrml_spw.fashion_cnn_native.allocated_native.operator import forward, recover, key_bank
from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt

ROOT = Path('results/thrml_spw/cifar_cnn_native')
TRAIN = ROOT/'training'
OUT = ROOT/'native_witness_evidence'


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    for path, h in json.loads((TRAIN/'completion.json').read_text())['artifact_sha256'].items():
        assert sha(path) == h, path
    assert json.loads((TRAIN/'evidence_audit.json').read_text())['passed']
    inherited = json.loads(Path('results/thrml_spw/fashion_cnn_native/allocated_native_evidence/protocol.json').read_text())
    sources = dict(inherited['source_sha256'])
    sources[str(Path(__file__))] = sha(__file__)
    inputs = {str(TRAIN/n): sha(TRAIN/n) for n in ['protocol.json', 'manifest.json', 'completion.json', 'evidence_audit.json', 'validation.npz']}
    training = json.loads((TRAIN/'manifest.json').read_text())
    for row in training['rows']:
        inputs[row['path']] = row['sha256']
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': sources,
         'input_sha256': inputs, 'models': 5, 'images': 16, 'banks': 2, 'bank_size': 8,
         'samples': [1, 4, 16], 'key_phase': 3, 'case_count': 15, 'weight_nodes': 3928,
         'counter_entries_per_bank': 49266,
         'scope': 'OfficialCPU THRML all5trained compactRGB CNN checkpoints/first16validation inputs/T1,4,16. Every native integer counter/activation/mask/output array checked against separately constructed public raw-state recovery. No labels/accuracy/loss/test images or physicalcost measurements. Small-count CPU witness does not replace full-budget GPU validation or full task/statistical evaluation.'}
    guard(p)
    write(OUT/'protocol.json', p)
    assert version('thrml') == '0.1.4' and jax.default_backend() == 'cpu' and not jax.config.x64_enabled
    assert subprocess.check_output(['git', '-C', 'vendor/thrml', 'rev-parse', 'HEAD'], text=True).strip() == '2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git', '-C', 'vendor/thrml', 'status', '--short'], text=True)
    x = np.load(TRAIN/'validation.npz')['inputs'][:16].astype(float).reshape(2, 8, 3, 16, 16)
    (OUT/'cases').mkdir(exist_ok=True)
    rows, entries, updates = [], 0, 0
    for mi, checkpoint in enumerate(training['rows']):
        model = np.load(checkpoint['path'])
        enc = encode(model)
        assert sum(e['fields'].size for e in enc) == 3928
        for bi, samples in enumerate(p['samples']):
            keys = [key_bank(3, mi, bi, 0, li, 0, 2) for li in range(3)]
            budgets = [samples]*3
            native = forward(x, model, enc, budgets, keys)
            public = recover(x, model, budgets, keys)
            for name in native:
                np.testing.assert_array_equal(native[name], public[name])
            np.testing.assert_array_equal(receipt(native), receipt(public))
            path = OUT/'cases'/f'model{mi}_T{samples}.npz'
            with path.open('wb') as stream:
                np.savez_compressed(stream, **native)
            count = sum(native[f'gate_{li}'].size+native[f'column_{li}'].size for li in range(3))
            work = 2*3928*samples
            assert count == 98532
            entries += count
            updates += work
            rows.append({'model': mi, 'samples': samples, 'path': str(path), 'sha256': sha(path),
                         'counter_entries_exact': count, 'native_updates': work, 'public_updates': work})
            write(OUT/'progress.json', {'rows': rows, 'counter_entries_exact': entries, 'native_updates': updates, 'public_updates': updates})
            print(json.dumps({'RGB_native_cases': len(rows), 'counter_entries_exact': entries}), flush=True)
    assert len(rows) == 15 and entries == 1477980 and updates == 824880
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'rows': rows, 'counter_entries_exact': entries,
                                    'all420canonical_arrays_exact': True, 'native_updates': updates, 'public_updates': updates,
                                    'warmup_node_updates_per_pass': 117840, 'scope': p['scope']})
    paths = [OUT/'protocol.json', OUT/'evidence_audit.json']+[Path(r['path']) for r in rows]
    write(OUT/'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'status': 'all15RGBnative_CPUcases_exact_fulltask_pending_goal_active',
                                 'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('RGB native/public witness passes15cases/1477980counterentries', flush=True)


if __name__ == '__main__':
    main()
