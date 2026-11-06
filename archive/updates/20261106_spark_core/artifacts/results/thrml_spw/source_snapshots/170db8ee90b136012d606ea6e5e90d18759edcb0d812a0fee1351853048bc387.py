"""Fixed 40-new-seed diagnostic after the ten-seed zero-offset discrepancy."""
import csv
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

OUT = Path('results/thrml_spw/zero_offset_recheck')


def main():
    if jax.default_backend() != 'gpu': raise RuntimeError('CUDA required')
    OUT.mkdir(exist_ok=True); (OUT/'cases').mkdir(exist_ok=True)
    prior = Path('results/thrml_spw/calibration_operator')
    artifact = Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
    source = Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')
    xp, rp = source/'stage11_inputs.npy', source/'stage11_conditional_reference.npy'
    protocol = {'seeds': list(range(10, 50)), 'replicas': [1, 4], 'observations': 512,
        'trigger': 'Original zero-offset ten-seed K4-minus-K1 MSE contrast excluded zero under its unadjusted approximate normal interval; investigate with a fixed separate batch, preserving original result.',
        'decision': 'Exactly forty new seeds, no stopping or extension based on observed significance; report fresh and combined results separately. This is a diagnostic follow-up, not a new primary comparison or formal equivalence test.',
        'prior_comparisons_sha256': sha(prior/'comparisons.json'), 'prior_metrics_sha256': sha(prior/'metrics.csv'),
        'artifact_sha256': sha(artifact), 'inputs_sha256': sha(xp), 'reference_sha256': sha(rp),
        'source_sha256': sha(__file__), 'operator_sha256': sha('research/thrml_spw/controlled_replica.py'),
        'key_schedule': 'Same as original zero-offset study: key1030730 fold K,0,0,seed,512. Only unused seeds10..49.',
        'contexts': 8, 'scope': 'Old consumed operator inputs, zero device offset, finite eight-bit controls and explicit spatial replicas; not model quality or physical device comparison.'}
    pp = OUT/'protocol.json'
    if pp.exists(): assert json.loads(pp.read_text()) == protocol
    else: pp.write_text(json.dumps(protocol, indent=2)+'\n')
    data = load_controls(artifact); x, ref = np.load(xp)[:8], np.load(rp)[:8]
    shape = tuple(data['header']['shape']); mean = x.astype(np.float64)@expected_weights(data)+data['bias']
    span = np.repeat(data['span'], 128, 0); q = data['probabilities']
    floor = float(np.mean((mean-ref)**2)); variance = float(np.mean(np.mean(x.astype(np.float64)**2, axis=(0,1))@(span**2*q*(1-q)))/512)
    rows = []; started = time.monotonic()
    for k in [1, 4]:
        layer = ControlledReplicaLinear(artifact, k); fields = jnp.asarray(np.tile(data['fields'], k))
        for seed in protocol['seeds']:
            name = f'K{k}_seed{seed}'; jp, npz = OUT/'cases'/f'{name}.json', OUT/'cases'/f'{name}.npz'
            if jp.exists():
                row = json.loads(jp.read_text()); assert sha(npz) == row['archive_sha256']; rows.append(row); continue
            key = jax.random.key(1030730)
            for v in [k, 0, 0, seed, 512]: key = jax.random.fold_in(key, v)
            y = np.asarray(layer.compiled_with_fields(512//k)(key, jnp.asarray(x.reshape(-1, shape[0])), fields)).reshape(ref.shape)
            np.savez_compressed(npz, output=y)
            row = {'K': k, 'seed': seed, 'T': 512//k, 'archive_sha256': sha(npz),
                'observed_total_mse': float(np.mean((y.astype(np.float64)-ref)**2)),
                'observed_sampling_mse': float(np.mean((y.astype(np.float64)-mean)**2)),
                'free_node_evaluations': 512*len(data['fields'])}
            jp.write_text(json.dumps(row, indent=2)+'\n'); rows.append(row)
            if seed % 10 == 9: print(f'Finished K{k} seeds through{seed}; elapsed={time.monotonic()-started:.1f}s', flush=True)
    assert len(rows) == 80
    with (OUT/'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result = {'native_runs': 80, 'expected_total_mse': floor+variance, 'conditional_bias_floor': floor, 'predicted_sampling_mse': variance,
        'elapsed_seconds': time.monotonic()-started, 'scope': protocol['scope']}
    (OUT/'REPORT.json').write_text(json.dumps(result, indent=2)+'\n'); manifest(OUT/'manifest.json', protocol, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
