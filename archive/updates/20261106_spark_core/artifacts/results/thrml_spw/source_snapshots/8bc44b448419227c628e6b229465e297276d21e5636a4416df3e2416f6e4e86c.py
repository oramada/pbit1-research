"""GPU vmap fidelity check against serial and official public trajectories."""
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from research.thrml_spw.native import NativeProbabilityBank
from .batched_bank import build_bank_functions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path('results/thrml_spw/lfm3_1p7b/batched_bank_preflight')
    out.mkdir(exist_ok=True)
    assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in jax.devices())
    names = ['batched_bank.py', 'preflight_batched_bank.py', 'programmed_worker.py']
    protocol = {'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in names},
        'auxiliary_sources_sha256': {n: sha(Path(n)) for n in ['research/thrml_spw/native.py', 'research/thrml_spw/controlled_positive_native.py']},
        'scope': 'Component-only exact serial/vmap/public trajectory comparison onGPU; no change to frozen fullmodel studies, fullmodel parity or timing/quality claim.'}
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    rng = np.random.default_rng(202610410)
    cases = []
    for fanin, depthwise, samples, lanes in [(1, True, 1, 3), (1, True, 1024, 4),
            (7, True, 16, 3), (128, False, 16, 4), (256, False, 16, 3), (128, False, 8192, 2)]:
        bank = NativeProbabilityBank(np.full((fanin, 64), .5))
        functions = build_bank_functions(bank, samples, depthwise)
        codes = rng.integers(0, 4096, size=(lanes, fanin * 64), dtype=np.uint16)
        codes[:, 0] = 0; codes[:, -1] = 4095
        fields = codes.astype(np.float32) * np.float32(8 / 4095) - np.float32(4)
        shape = (lanes, 2, fanin, 64) if depthwise else (lanes, 2, fanin)
        inputs = rng.integers(-2047, 2048, size=shape, dtype=np.int32)
        if samples == 8192:
            inputs[:, 0] = 2047; inputs[:, 1] = -2047
        tiles = [0, 1, 7, 19][:lanes]
        keys = jnp.stack([jax.random.fold_in(jax.random.fold_in(jax.random.key(202610410), 287), tile) for tile in tiles])
        serial = np.stack([np.asarray(functions['serial'](keys[i], jnp.asarray(inputs[i]), jnp.asarray(fields[i]))) for i in range(lanes)])
        batched = np.asarray(functions['batched'](keys, jnp.asarray(inputs), jnp.asarray(fields)))
        np.testing.assert_array_equal(serial, batched)
        public_serial = np.stack([np.asarray(functions['serial_public'](keys[i], jnp.asarray(fields[i]))) for i in range(lanes)])
        public_batched = np.asarray(functions['batched_public'](keys, jnp.asarray(fields)))
        np.testing.assert_array_equal(public_serial, public_batched)
        path = out / f'case{len(cases)}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f, serial=serial, batched=batched, public_serial=public_serial,
                public_batched=public_batched, inputs=inputs, fields=fields, codes=codes,
                tile_ids=np.asarray(tiles), samples=np.asarray(samples), depthwise=np.asarray(depthwise),
                fanin=np.asarray(fanin))
        rec = {'fanin': fanin, 'depthwise': depthwise, 'samples': samples, 'lanes': lanes,
               'path': str(path), 'sha256': sha(path), 'public_bits_per_path': public_serial.size,
               'counter_entries_per_path': serial.size}
        cases.append(rec); print(json.dumps(rec), flush=True)
    result = {'status': 'all_six_serial_vmap_public_cases_equal_producer',
        'devices': list(map(str, jax.devices())), 'cases': cases, 'protocol_sha256': sha(out / 'protocol.json')}
    with (out / 'results.json').open('x') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
