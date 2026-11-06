"""Native CPU before/after sampling of exported commands, with full replay."""
import json
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule, sample_states, sample_with_observation
from research.thrml_spw.native import NativeProbabilityBank
from research.thrml_spw.controlled_positive_native import set_fields
from research.thrml_spw.lfm3.precision_worker import GroupSum


def main():
    out = Path(sys.argv[1])
    a = np.load(out / 'training_arrays.npz')
    bank = NativeProbabilityBank(np.full((128, 8), .5))
    inputs = jnp.asarray(a['input_integers'])
    initial = [jnp.zeros(bank.logical_pbits, dtype=jnp.bool_)]
    schedule = SamplingSchedule(1, 16, 1)

    def run(key, fields):
        obs = GroupSum(bank.readout, inputs, bank.shape, False, 128)
        return sample_with_observation(key, set_fields(bank, fields), schedule, initial, [], obs.init(), obs)[0]

    def public(key, fields):
        return sample_states(key, set_fields(bank, fields), schedule, initial, [], [bank.readout])[0]

    run_many = jax.jit(jax.vmap(run, in_axes=(0, None)))
    public_many = jax.jit(jax.vmap(public, in_axes=(0, None)))
    keys = jnp.stack([jax.random.fold_in(jax.random.fold_in(jax.random.key(202610314), seed), trial)
                      for seed in range(5) for trial in range(100)])
    rows = []
    for label in ['before', 'after']:
        fields = jnp.asarray(a[label + '_fields'].reshape(-1))
        total = np.asarray(run_many(keys, fields))[:, :, 0, :]
        states = np.asarray(public_many(keys, fields)).reshape(500, 16, 128, 8)
        # Independent public-state contraction, never estimator reconstruction
        # in the native observer. Both use exactly the same native trajectories.
        check = np.einsum('bi,nio->nbo', a['input_integers'].astype(np.int64), states.sum(1, dtype=np.int64))
        np.testing.assert_array_equal(total.astype(np.int64), check)
        lo, span = a['lower'].reshape(8), a['span'].reshape(8)
        outputs = (total.astype(np.float64) / 16 * span + a['input_integers'].sum(1)[None, :, None] * lo) * a['input_scale']
        np.savez_compressed(out / (label + '_native.npz'), states=states, total=total, outputs=outputs)
        mse = np.mean((outputs - a['target'][None, :, :]) ** 2, axis=(1, 2))
        rows.append({'condition': label, 'empirical_mean_MSE': float(mse.mean()),
                     'MSE_standard_error_across500_trials': float(mse.std(ddof=1) / np.sqrt(500)),
                     'per_seed_mean_MSE': mse.reshape(5, 100).mean(1).tolist(),
                     'retained_node_evaluations': 500 * 16 * 1024,
                     'warmup_node_evaluations': 500 * 1024})
    (out / 'native_results.json').write_text(json.dumps({'status': 'both_conditions_full_public_replay_exact',
        'devices': [str(d) for d in jax.devices()], 'conditions': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
