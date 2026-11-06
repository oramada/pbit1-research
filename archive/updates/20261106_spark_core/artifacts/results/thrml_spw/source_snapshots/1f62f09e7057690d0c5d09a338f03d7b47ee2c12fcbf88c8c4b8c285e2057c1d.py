"""Thirty validation-only fixed-range learned-probability native cases."""
import json
import os
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from results.thrml_spw.fashion_cnn_native.inference.fast_native import forward
from results.thrml_spw.fashion_cnn_native.probability_native.adapter import encoding
from results.thrml_spw.fashion_cnn_native.probability_native.replay import recover
from results.thrml_spw.fashion_cnn_native.probability_native.watch import ROOT, OUT, sha, write, guard, parent_closed


def main():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    parent_closed()
    assert jax.default_backend() == 'gpu' and not jax.config.x64_enabled
    assert os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0'
    assert not (OUT/'evidence_audit.json').exists()
    trained = json.loads((ROOT/'probability_finetune_evidence/manifest.json').read_text())
    originals = json.loads((ROOT/'training/manifest.json').read_text())
    x = np.load(ROOT/'training/validation.npz')['inputs'][:16].astype(float).reshape(2, 8, 1, 14, 14)
    rows = []
    updates = 0
    entries = 0
    for row in trained['rows']:
        mi = row['model']
        method = p['methods'].index(row['method'])
        exported = np.load(row['path'])
        original = np.load(originals['rows'][mi]['path'])
        enc = encoding(exported)
        for t in p['samples']:
            guard(p)
            assert shutil.disk_usage('.').free >= p['minimum_free_bytes']
            root = jax.random.key(p['key_root'])
            for item in [mi, method, t]:
                root = jax.random.fold_in(root, item)
            keys = [jax.vmap(lambda bank: jax.random.fold_in(jax.random.fold_in(root, li), bank))(jnp.arange(2, dtype=jnp.uint32)) for li in range(3)]
            started = time.monotonic()
            native = forward(x, exported, enc, t, keys)
            native_seconds = time.monotonic()-started
            public = recover(x, exported, original, t, keys)
            assert len(native) == len(public) == 14
            for name in native:
                np.testing.assert_array_equal(native[name], public[name], err_msg=name)
            count = sum(v.size for name, v in native.items() if name.startswith(('gate_', 'column_')))
            assert count == 75492
            path = OUT/f'model{mi}_{row["method"]}_T{t}.npz'
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, **native)
            tmp.replace(path)
            u = 2*2664*t
            updates += u
            entries += count
            rows.append({'model': mi, 'method': row['method'], 'T': t, 'path': str(path), 'sha256': sha(path),
                         'counter_entries_exact': count, 'all28_bank_arrays_exact': True, 'native_updates': u,
                         'public_updates': u, 'native_host_seconds': native_seconds})
            write(OUT/'progress.json', {'rows': rows, 'native_updates': updates, 'public_updates': updates})
            print(json.dumps(rows[-1]), flush=True)
    assert len(rows) == 30 and updates == 71608320 and entries == 2264760
    guard(p)
    write(OUT/'evidence_audit.json', {'passed': True, 'rows': rows, 'native_updates': updates, 'public_updates': updates,
                                   'counter_entries_exact': entries, 'protocol_sha256': sha(OUT/'protocol.json'),
                                   'scope': 'All10learned probability models/three budgets/two8imagevalidationbanks. Native output counters and every code/mask/output checked against public raw-state recovery with independently reconstructed learned commands and original ranges. No labels/task scores/fresh generalization or hardware costs. Host timings are not physical latency.'})
    paths = [OUT/'protocol.json', OUT/'evidence_audit.json']+[Path(row['path']) for row in rows]
    write(OUT/'completion.json', {'status': 'learned_probability_native_validation_audited_goal_active',
                                'artifact_sha256': {str(path): sha(path) for path in paths},
                                'native_updates': updates, 'public_updates': updates})


if __name__ == '__main__':
    main()
