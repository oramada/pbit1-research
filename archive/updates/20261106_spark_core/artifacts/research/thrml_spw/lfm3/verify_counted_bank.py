"""Independent public-count and int64 contractions for count factorization."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path('results/thrml_spw/lfm3_1p7b/counted_bank_preflight')
    p = json.loads((out / 'protocol.json').read_text()); r = json.loads((out / 'results.json').read_text())
    assert r['status'] == 'all_six_count_factorizations_equal_direct_observer_producer'
    assert sha(out / 'protocol.json') == r['protocol_sha256'] and len(r['cases']) == 6
    for name, digest in p['sources_sha256'].items(): assert sha(Path(__file__).with_name(name)) == digest
    public_bits = entries = count_entries = 0
    for rec in r['cases']:
        path = Path(rec['path']); assert sha(path) == rec['sha256']
        with np.load(path) as a:
            fanin = int(a['fanin']); samples = int(a['samples']); size = min(128, fanin); groups = fanin // size
            assert 2047 * size * samples < 2 ** 31
            assert a['counts'].dtype == a['original'].dtype == np.int32
            assert a['counts'].nbytes == rec['count_workspace_bytes']
            assert a['original'].nbytes == rec['reference_output_accumulator_bytes']
            np.testing.assert_array_equal(a['fields'], a['codes'].astype(np.float32) * np.float32(8 / 4095) - np.float32(4))
            for lane in range(rec['lanes']):
                counts = a['states'][lane].reshape(samples, groups, size, 64).sum(0, dtype=np.int64)
                np.testing.assert_array_equal(counts.reshape(-1), a['counts'][lane])
                x = a['inputs'][lane].astype(np.int64)
                if bool(a['depthwise']):
                    expected = (x.reshape(rec['positions'], groups, size, 64) * counts[None]).sum(2, dtype=np.int64)
                else:
                    x = x.reshape(rec['positions'], groups, size)
                    expected = np.stack([x[:, g] @ counts[g] for g in range(groups)], axis=1)
                for name in ['original', 'serial', 'batched_total']:
                    np.testing.assert_array_equal(expected, a[name][lane].astype(np.int64))
            np.testing.assert_array_equal(a['counts'], a['batched_counts'])
            public_bits += a['states'].size; entries += a['original'].size; count_entries += a['counts'].size
    audit = {'status': 'all_six_public_counts_and_exact_integer_observable_factorizations_verified',
        'source_sha256': sha(Path(__file__)), 'results_sha256': sha(out / 'results.json'),
        'public_bits': public_bits, 'output_counter_entries_per_path': entries, 'weight_count_entries_per_path': count_entries,
        'scope': 'Software simulation identity sum_t XZ_t=Xsum_tZ_t underintegeroverflowguard, including520positions/7680fanin. Explicitper-weightint32countworkspace; not physicaldirect-outputcounterdatapath equivalence, freeprecision/samplinggain, fullmodeladoption, modelquality, simulator-speedorhardwareenergy/latencyclaim. Existing model studies unchanged.'}
    with (out / 'independent_audit.json').open('x') as f: json.dump(audit, f, indent=2)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
