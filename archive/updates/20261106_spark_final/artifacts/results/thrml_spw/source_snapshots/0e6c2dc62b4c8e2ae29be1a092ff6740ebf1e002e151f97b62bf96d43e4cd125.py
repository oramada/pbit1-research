"""CPU native/public witness for distinct per-layer sample counts, no labels."""
import datetime
import json
from pathlib import Path
import numpy as np
import jax
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT, TRAIN, sha, write, guard
from results.thrml_spw.fashion_cnn_native.inference.native import encode
from results.thrml_spw.fashion_cnn_native.allocated_native.operator import forward, recover, key_bank
from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt

OUT = ROOT/'allocated_witness_evidence'


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    parent = ROOT/'allocated_native_evidence/protocol.json'
    p = json.loads(parent.read_text())
    sources = dict(p['source_sha256'])
    sources[str(Path(__file__))] = sha(__file__)
    inputs = dict(p['input_sha256'])
    inputs[str(parent)] = sha(parent)
    protocol = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_sha256': sources, 'input_sha256': inputs, 'schedules': [[1, 2, 4], [4, 2, 1], [4, 4, 4]],
                'scope': 'All5originalCNNs/first8validationinputs, CPU officialTHRML mixed per-layer counts. No labels/metrics/test scoring. Basic operator witness only; queuedGPU full-budget preflight and full100case test/public/statistical audits remain required.'}
    guard(protocol)
    write(OUT/'protocol.json', protocol)
    assert jax.default_backend() == 'cpu' and not jax.config.x64_enabled
    x = np.load(TRAIN/'validation.npz')['inputs'][:8].astype(float).reshape(1, 8, 1, 14, 14)
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    rows, updates, entries = [], 0, 0
    (OUT/'cases').mkdir(exist_ok=True)
    for mi in range(5):
        model = np.load(train[mi]['path'])
        enc = encode(model)
        for ci, budgets in enumerate(protocol['schedules']):
            keys = [key_bank(2, mi, ci, 0, li, 0, 1) for li in range(3)]
            native = forward(x, model, enc, budgets, keys)
            public = recover(x, model, budgets, keys)
            for name in native:
                np.testing.assert_array_equal(native[name], public[name])
            np.testing.assert_array_equal(receipt(native), receipt(public))
            path = OUT/'cases'/f'model{mi}_schedule{ci}.npz'
            with path.open('wb') as stream:
                np.savez_compressed(stream, **native)
            count = sum(native[f'gate_{li}'].size+native[f'column_{li}'].size for li in range(3))
            work = sum(n*t for n, t in zip([72, 1152, 1440], budgets))
            updates += work
            entries += count
            rows.append({'model': mi, 'budgets': budgets, 'path': str(path), 'sha256': sha(path),
                         'counter_entries_exact': count, 'native_updates': work, 'public_updates': work})
            print(json.dumps({'mixed_CPU_cases': len(rows)}), flush=True)
    assert len(rows) == 15 and entries == 566190 and updates == 114120
    guard(protocol)
    write(OUT/'evidence_audit.json', {'passed': True, 'rows': rows, 'counter_entries_exact': entries,
                                    'native_updates': updates, 'public_updates': updates, 'warmup_node_updates_per_pass': 39960,
                                    'all210canonical_arrays_exact': True, 'scope': protocol['scope']})
    paths = [OUT/'protocol.json', OUT/'evidence_audit.json']+[Path(r['path']) for r in rows]
    write(OUT/'completion.json', {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'status': 'all15mixed_sample_CPU_cases_exact_GPU_test_pending_goal_active',
                                 'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('Mixed native/public CPU witness passes15cases/566190counter entries', flush=True)


if __name__ == '__main__':
    main()
