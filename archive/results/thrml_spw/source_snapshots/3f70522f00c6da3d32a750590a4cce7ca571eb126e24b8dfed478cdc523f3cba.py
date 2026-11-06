"""CPU public-trajectory and independent int64 readout checks for K4/K8."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import numpy as np
from research.thrml_spw.native import NativeProbabilityBank
from .adapted_word_operator import execute


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = ['adapted_word_operator.py', 'preflight_adapted_word_operator.py', 'counted_bank.py']
    sources = {}
    for name in names:
        path = Path(__file__).with_name(name)
        sources[name] = sha(path)
        assert sha(Path('results/thrml_spw/source_snapshots') / (sources[name] + '.py')) == sources[name]
    rng = np.random.default_rng(202610430)
    cases = [(7, 4, 16, True), (16, 8, 16, False), (128, 4, 1024, False),
             (128, 8, 512, False), (256, 4, 8, True), (256, 8, 8, False)]
    records = []
    banks = {}
    for ordinal, (fanin, bits, samples, depthwise) in enumerate(cases):
        size = min(128, fanin)
        groups = fanin // size
        if fanin not in banks:
            banks[fanin] = NativeProbabilityBank(np.full((fanin, 64), .5))
        bank = banks[fanin]
        words = (np.arange(fanin * 64).reshape(fanin, 64) % (1 << bits)).astype(np.uint8)
        xshape = (3, fanin, 64) if depthwise else (3, fanin)
        inputs = rng.integers(-2047, 2048, size=xshape, dtype=np.int32)
        inputs[0] = 2047
        inputs[1] = -2047
        lower = np.linspace(-1, 1, groups * 64).reshape(groups, 64)
        span = np.linspace(0, 2, groups * 64).reshape(groups, 64)
        y, receipt = execute(bank, inputs, words, lower, span, bits, samples,
                             jax.random.key(202610430 + ordinal), depthwise, replay=True)
        xg = inputs.reshape(3, groups, size, 64) if depthwise else inputs.reshape(3, groups, size)
        weighted = np.zeros((3, groups, 64), np.float64)
        public_bits = 0
        for plane in receipt['planes']:
            states = plane['states'].reshape(samples, groups, size, 64)
            public_bits += states.size
            counts = np.sum(states, axis=0, dtype=np.int64)
            np.testing.assert_array_equal(counts.reshape(-1), plane['counts'].astype(np.int64))
            oracle = np.zeros((3, groups, 64), np.int64)
            # Contract each public realization independently: no count-factorized oracle.
            for z in states:
                if depthwise:
                    oracle += np.sum(xg.astype(np.int64) * z[None], axis=2, dtype=np.int64)
                else:
                    for g in range(groups):
                        oracle[:, g, :] += xg[:, g].astype(np.int64) @ z[g].astype(np.int64)
            np.testing.assert_array_equal(oracle, plane['total'].astype(np.int64))
            weighted += oracle.astype(np.float64) * ((1 << plane['bit']) / ((1 << bits) - 1))
        offset = np.sum(xg, axis=2, dtype=np.int64)
        if not depthwise:
            offset = offset[:, :, None]
        expected = np.sum(weighted / samples * span[None] + offset * lower[None], axis=1)
        np.testing.assert_array_equal(y, expected)
        assert receipt['retained_scheduled_evaluations'] == bits * fanin * 64 * samples == public_bits
        assert receipt['warmup_scheduled_evaluations'] == bits * fanin * 64
        assert receipt['max_temporary_weight_count_bytes'] == fanin * 64 * 4
        assert receipt['logical_word_pbits'] == bits * fanin * 64
        records.append({'fanin': fanin, 'bits': bits, 'samples': samples, 'depthwise': depthwise,
                        'public_bits_checked': public_bits, 'all_counts_and_integer_contractions_exact': True,
                        'weighted_affine_readout_exact': True, 'output_sha256': hashlib.sha256(y.tobytes()).hexdigest()})
        print(json.dumps(records[-1]), flush=True)
    result = {'status': 'six_CPU_native_word_cases_public_states_counts_int64_contractions_and_readout_verified',
              'sources_sha256': sources, 'devices': [str(d) for d in jax.devices()], 'cases': records,
              'public_bits_checked': sum(r['public_bits_checked'] for r in records),
              'scope': 'Official THRML CPU unary banks, both K4/K8, linear and depthwise integer inputs, multiple groups, signed extrema, and matched high budgets. Per-bit temporary counts explicitly charged. Component bank only: padded 64 channels, no frontend tiling or ACT quantization in this function, no full-model quality or hardware equivalence.'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
