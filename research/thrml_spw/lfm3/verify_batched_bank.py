"""Independent offline int64 reconstruction of every serial/vmap counter."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path('results/thrml_spw/lfm3_1p7b/batched_bank_preflight')
    p = json.loads((out / 'protocol.json').read_text())
    r = json.loads((out / 'results.json').read_text())
    assert r['status'] == 'all_six_serial_vmap_public_cases_equal_producer'
    assert r['protocol_sha256'] == sha(out / 'protocol.json')
    for name, digest in p['sources_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in p['auxiliary_sources_sha256'].items():
        assert sha(Path(name)) == digest
    total_bits = entries = 0
    for rec in r['cases']:
        assert sha(Path(rec['path'])) == rec['sha256']
        with np.load(rec['path']) as a:
            serial, batched = a['serial'], a['batched']
            assert serial.dtype == batched.dtype == np.int32
            np.testing.assert_array_equal(serial, batched)
            states = a['public_serial']
            assert states.dtype == np.bool_ and np.array_equal(states, a['public_batched'])
            fanin, samples = int(a['fanin']), int(a['samples'])
            size = min(128, fanin); groups = fanin // size
            assert fanin == rec['fanin'] and samples == rec['samples']
            assert 2047 * size * samples < 2 ** 31
            assert serial.shape == (rec['lanes'], 2, groups, 64)
            np.testing.assert_array_equal(a['fields'], a['codes'].astype(np.float32) * np.float32(8 / 4095) - np.float32(4))
            for lane in range(rec['lanes']):
                z = states[lane].reshape(samples, groups, size, 64).astype(np.int64)
                counts = z.sum(axis=0, dtype=np.int64)
                x = a['inputs'][lane].astype(np.int64)
                if bool(a['depthwise']):
                    grouped = x.reshape(2, groups, size, 64)
                    expected = (grouped * counts[None]).sum(axis=2, dtype=np.int64)
                else:
                    grouped = x.reshape(2, groups, size)
                    expected = np.stack([grouped[:, g] @ counts[g] for g in range(groups)], axis=1)
                np.testing.assert_array_equal(serial[lane].astype(np.int64), expected)
            total_bits += states.size; entries += serial.size
    assert len(r['cases']) == 6
    audit = {'status': 'all_six_GPU_serial_vmap_trajectories_and_independent_int64_counters_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
        'public_bits_each_serial_and_batched_path': total_bits, 'integer_entries_each_path': entries,
        'scope': 'Sixcomponentcases, differentkeys/lanes/fanins, fouroperatorlayoutsviafanin1/7/128/256, T1/16/1024/8192. Bitwise serial/vmap publicstatesandint32counters withindependentint64reconstruction. No fullmodel integration, fullmodel fidelity, samplerhardwarelatency, energyorqualityclaim. Existingstudies unchanged.'}
    with (out / 'independent_audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
