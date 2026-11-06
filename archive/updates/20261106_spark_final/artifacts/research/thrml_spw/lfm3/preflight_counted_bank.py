"""GPU exact counts/direct-observer comparison including520position shapes."""
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from research.thrml_spw.native import NativeProbabilityBank
from .batched_bank import build_bank_functions
from .counted_bank import build_count_functions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path('results/thrml_spw/lfm3_1p7b/counted_bank_preflight'); out.mkdir(exist_ok=True)
    assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in jax.devices())
    sources = ['counted_bank.py', 'preflight_counted_bank.py', 'verify_counted_bank.py', 'batched_bank.py', 'programmed_worker.py']
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in sources},
        'scope': 'Exact software observable factorization only. Int32 per-weight count workspace is explicit; no reduction in samples, physical counters, physical energy or statistical error claimed. Current fullmodel studies unchanged.'}
    with (out / 'protocol.json').open('x') as f: json.dump(protocol, f, indent=2)
    rng = np.random.default_rng(202610412); records = []
    cases = [(1, True, 1, 3, 520), (7, True, 16, 3, 520), (128, False, 16, 4, 520),
             (7680, False, 16, 2, 520), (128, False, 1024, 3, 3), (128, False, 8192, 2, 2)]
    for fanin, depthwise, samples, lanes, positions in cases:
        bank = NativeProbabilityBank(np.full((fanin, 64), .5))
        reference = build_bank_functions(bank, samples, depthwise)
        factored = build_count_functions(bank, samples, depthwise)
        codes = rng.integers(0, 4096, (lanes, fanin * 64), dtype=np.uint16)
        codes[:, 0] = 0; codes[:, -1] = 4095
        fields = codes.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
        shape = (lanes, positions, fanin, 64) if depthwise else (lanes, positions, fanin)
        x = rng.integers(-2047, 2048, shape, dtype=np.int32)
        if samples == 8192: x[:, 0] = 2047; x[:, 1] = -2047
        keys = jnp.stack([jax.random.fold_in(jax.random.fold_in(jax.random.key(202610412), 287), i) for i in range(lanes)])
        original = np.stack([np.asarray(reference['serial'](keys[i], jnp.asarray(x[i]), jnp.asarray(fields[i]))) for i in range(lanes)])
        serial_pairs = [factored['serial'](keys[i], jnp.asarray(x[i]), jnp.asarray(fields[i])) for i in range(lanes)]
        serial = np.stack([np.asarray(v[0]) for v in serial_pairs]); counts = np.stack([np.asarray(v[1]) for v in serial_pairs])
        batched_total, batched_counts = factored['batched'](keys, jnp.asarray(x), jnp.asarray(fields))
        batched_total, batched_counts = np.asarray(batched_total), np.asarray(batched_counts)
        np.testing.assert_array_equal(original, serial); np.testing.assert_array_equal(serial, batched_total)
        np.testing.assert_array_equal(counts, batched_counts)
        states = np.stack([np.asarray(reference['serial_public'](keys[i], jnp.asarray(fields[i]))) for i in range(lanes)])
        np.testing.assert_array_equal(counts, states.reshape(lanes, samples, -1).sum(1, dtype=np.int32))
        path = out / f'case{len(records)}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f, original=original, serial=serial, batched_total=batched_total,
                counts=counts, batched_counts=batched_counts, states=states, inputs=x,
                codes=codes, fields=fields, fanin=np.asarray(fanin), samples=np.asarray(samples),
                depthwise=np.asarray(depthwise))
        rec = {'path': str(path), 'sha256': sha(path), 'fanin': fanin, 'positions': positions,
            'depthwise': depthwise, 'samples': samples, 'lanes': lanes,
            'count_workspace_bytes': counts.nbytes, 'reference_output_accumulator_bytes': original.nbytes}
        records.append(rec); print(json.dumps(rec), flush=True)
    with (out / 'results.json').open('x') as f:
        json.dump({'status': 'all_six_count_factorizations_equal_direct_observer_producer',
            'protocol_sha256': sha(out / 'protocol.json'), 'cases': records}, f, indent=2)


if __name__ == '__main__':
    main()
