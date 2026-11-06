"""Frozen synthetic calibration-error/space-time study on a real large operator.

Uses old development activations only; no new model-quality selection or physical
device specification is inferred from these dimensionless offset scenarios.
"""
import csv
import gc
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .controlled_replica import ControlledReplicaLinear
from .packed_controls import load_controls, expected_weights
from .provenance import manifest
from .verify_transformer_control import sha

OUT = Path('results/thrml_spw/calibration_operator')
ARTIFACT = Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
OLD = Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')


def main():
    if jax.default_backend() != 'gpu':
        raise RuntimeError('This pinned large-operator study requires CUDA')
    OUT.mkdir(exist_ok=True)
    (OUT/'cases').mkdir(exist_ok=True)
    xp, rp = OLD/'stage11_inputs.npy', OLD/'stage11_conditional_reference.npy'
    protocol = {
        'schema': 'spw-synthetic-offset-operator-v1', 'sigmas': [0., .05, .2],
        'replicas': [1, 4], 'total_observations_per_weight': [128, 512],
        'device_and_sampling_seeds': list(range(10)), 'contexts': 8,
        'offset_law': 'b_real=float32(b_code + sigma_b * epsilon), epsilon~N(0,1); beta=1; no re-quantization or clipping after physical offset',
        'dependence': 'independent: epsilon differs by replica; shared: same epsilon for all replicas of a weight; distinct weights independent in both; offsets fixed across T and all sampled timesteps',
        'field_rng': 'numpy.default_rng(2026102700+device_seed).standard_normal((4, number_of_unique_free_nodes)); K1 takes first row; shared K4 broadcasts first row; independent K4 takes all rows',
        'sampling_key': 'JAX key(1030730), fold K, sigma_index, dependence_index(0=independent,1=shared), seed, total_observations; no sample-path pairing claimed',
        'sigma_zero': 'Only independent label is executed: independent/shared offsets are identical at sigma=0',
        'K1': 'Only independent label is executed: shared/independent replica offsets are identical with one replica',
        'endpoint_policy': 'Packed endpoint clamps stay exact and are not perturbed; this experiment concerns free-node unary fields only',
        'input_selection': 'First eight contexts of previously consumed old quarter-model development seed0/T512 stage11 student inputs',
        'artifact_sha256': sha(ARTIFACT), 'inputs_sha256': sha(xp), 'reference_sha256': sha(rp),
        'source_sha256': {str(Path('research/thrml_spw')/n): sha(Path('research/thrml_spw')/n) for n in ['calibration_operator.py', 'controlled_replica.py', 'replicated_native.py', 'packed_controls.py', 'native.py']},
        'decision_rule': 'No budget selection or success gate; preserve all conditions and compare observed MSE to conditional analytic prediction and paired resource counts',
        'scope': 'Dimensionless synthetic calibration scenarios, not measured Extropic hardware errors. Explicit replicas are not positional bits or an official learned-weight reference. No physical energy/latency proof.',
    }
    pp = OUT/'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol, 'Refuse changed design in existing evidence directory'
    else:
        pp.write_text(json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic()
    data = load_controls(ARTIFACT)
    x, ref = np.load(xp)[:8], np.load(rp)[:8]
    shape = tuple(data['header']['shape'])
    lower = np.repeat(data['lower'], 128, axis=0)[:shape[0]]
    span = np.repeat(data['span'], 128, axis=0)[:shape[0]]
    nominal = x.astype(np.float64)@expected_weights(data)+data['bias']
    input_energy = np.mean(x.astype(np.float64)**2, axis=(0, 1))
    controls = {}
    for k in protocol['replicas']:
        layer = ControlledReplicaLinear(ARTIFACT, k)
        controls[str(k)] = layer.controls()
        for si, sigma in enumerate(protocol['sigmas']):
            dependencies = ['independent', 'shared'] if k == 4 and sigma else ['independent']
            for dep in dependencies:
                di = int(dep == 'shared')
                for seed in protocol['device_and_sampling_seeds']:
                    rng = np.random.default_rng(2026102700+seed)
                    noise = rng.standard_normal((4, len(data['fields'])))
                    noise = noise[:k] if dep == 'independent' else np.broadcast_to(noise[0], (k, noise.shape[1]))
                    fields = (data['fields'][None]+sigma*noise).astype(np.float32).reshape(-1)
                    q = np.broadcast_to(data['probabilities'][None], (k, *shape)).reshape(k, -1).copy()
                    q[:, data['free_indices']] = np.exp(-np.logaddexp(0, -2*fields.reshape(k, -1).astype(np.float64)))
                    q = q.reshape(k, *shape)
                    mean_weight = lower+span*q.mean(0)
                    mu = x.astype(np.float64)@mean_weight+data['bias']
                    floor = float(np.mean((mu-ref)**2))
                    offset_floor = float(np.mean((mu-nominal)**2))
                    per_step_variance = float(np.mean(input_energy@(span**2*np.sum(q*(1-q), axis=0)/(k*k))))
                    field_digest = __import__('hashlib').sha256(fields.tobytes()).hexdigest()
                    fields_device = jnp.asarray(fields)
                    for obs in protocol['total_observations_per_weight']:
                        t = obs//k
                        name = f'K{k}_s{si}_{dep}_seed{seed}_N{obs}'
                        jp, npz = OUT/'cases'/f'{name}.json', OUT/'cases'/f'{name}.npz'
                        if jp.exists():
                            saved = json.loads(jp.read_text())
                            assert saved['field_sha256'] == field_digest and sha(npz) == saved['output_archive_sha256']
                            continue
                        key = jax.random.key(1030730)
                        for value in [k, si, di, seed, obs]:
                            key = jax.random.fold_in(key, value)
                        y = np.asarray(layer.compiled_with_fields(t)(key, jnp.asarray(x.reshape(-1, shape[0])), fields_device)).reshape(ref.shape)
                        np.savez_compressed(npz, output=y)
                        row = {'name': name, 'K': k, 'sigma_index': si, 'sigma_b': sigma, 'dependence': dep, 'device_sampling_seed': seed,
                            'observations': obs, 'T': t, 'logical_nodes': layer.bank.logical_pbits,
                            'free_node_evaluations': t*len(layer.bank.free_indices), 'logical_weight_readouts': t*layer.bank.logical_pbits,
                            'nominal_binary_product_terms': int(x.shape[0]*x.shape[1]*t*layer.bank.logical_pbits),
                            'field_sha256': field_digest, 'output_archive_sha256': sha(npz),
                            'conditional_bias_floor': floor, 'offset_error_floor_vs_nominal': offset_floor,
                            'predicted_sampling_mse': per_step_variance/t, 'predicted_total_mse': floor+per_step_variance/t,
                            'observed_sampling_mse': float(np.mean((y.astype(np.float64)-mu)**2)),
                            'observed_total_mse': float(np.mean((y.astype(np.float64)-ref)**2))}
                        jp.write_text(json.dumps(row, indent=2)+'\n')
                        print(f'{name}: MSE={row["observed_total_mse"]:.6g}, floor={floor:.6g}, elapsed={time.monotonic()-started:.1f}s', flush=True)
                gc.collect()
        del layer
        jax.clear_caches()
        gc.collect()
    rows = [json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*.json'))]
    assert len(rows) == 160
    with (OUT/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {'native_conditions': len(rows), 'controls': controls, 'elapsed_seconds': time.monotonic()-started,
        'nominal_control_bias_floor': float(np.mean((nominal-ref)**2)), 'contexts': 8, 'matrix_weights': int(np.prod(shape)),
        'scope': protocol['scope']+' Ten replicates vary both fixed device offsets and sampling streams; these are not ten pure Monte Carlo repeats of one physical device realization.'}
    (OUT/'REPORT.json').write_text(json.dumps(result, indent=2)+'\n')
    manifest(OUT/'manifest.json', protocol, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
