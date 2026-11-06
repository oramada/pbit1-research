"""Native measured-offset compensation with finite eight-bit controls."""
import csv
import hashlib
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibration_count_native import CalibrationNativeLinear
from .calibration_control_math import estimate_command, decode_command
from .packed_controls import load_controls, expected_weights
from .provenance import manifest
from .verify_transformer_control import sha

OUT = Path('results/thrml_spw/calibrated_operator')
ARTIFACT = Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
OLD = Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    if jax.default_backend() != 'gpu':
        raise RuntimeError('This pinned study requires CUDA')
    OUT.mkdir(exist_ok=True); (OUT/'cases').mkdir(exist_ok=True)
    xp, rp = OLD/'stage11_inputs.npy', OLD/'stage11_conditional_reference.npy'
    protocol = {'schema': 'spw-measured-calibration-operator-v1', 'sigma_b': .2,
        'seeds': list(range(10)), 'calibration_samples': [256, 1024, 4096], 'inference_samples': 512,
        'contexts': 8, 'probe_code': 128, 'controller_bits': 8, 'controller_bound': 4.,
        'methods': ['uncalibrated', 'oracle_offset', 'measured_256', 'measured_1024', 'measured_4096'],
        'field_rng': 'First row of numpy.default_rng(2026102700+seed).standard_normal((4,n_free)); delta=0.2*row. Fixed for calibration and inference.',
        'calibration': 'Program code128 (finite positive field, not zero); official THRML free-state counts, smoothed p=(count+0.5)/(Ncal+1), delta_hat=0.5*logit(p)-probe_field. No true offset passed to measured estimator.',
        'correction': 'Command round-to-nearest eight-bit code of clip(nominal_field-delta_hat,-4,4); physical realized field=float32(command_field+true_delta). Endpoints remain unperturbed clamps.',
        'oracle_scope': 'Known true offset supplied only to oracle diagnostic. Oracle is not an implementable measured calibration procedure or guaranteed optimal controller.',
        'sampling_keys': 'JAX key1040730, fold seed then method index0..4 for inference; calibration key1040731, fold seed then Ncal; streams disjoint, no inference data in estimator.',
        'decision': 'No selection or success gate; report all three calibration budgets, uncalibrated and oracle diagnostics, calibration and inference costs separately.',
        'artifact_sha256': sha(ARTIFACT), 'inputs_sha256': sha(xp), 'reference_sha256': sha(rp),
        'source_sha256': {str(Path('research/thrml_spw')/n): sha(Path('research/thrml_spw')/n) for n in ['calibrated_operator.py', 'calibration_count_native.py', 'calibration_control_math.py', 'controlled_replica.py', 'replicated_native.py', 'native.py', 'packed_controls.py']},
        'scope': 'One real 2,359,296-weight native operator on previously consumed old development inputs. Synthetic fixed additive unary-field offsets, not measured hardware, gain error, drift or full-model quality. Calibration precision and cost are counted, not assumed free.'}
    pp = OUT/'protocol.json'
    if pp.exists(): assert json.loads(pp.read_text()) == protocol
    else: pp.write_text(json.dumps(protocol, indent=2)+'\n')
    data = load_controls(ARTIFACT); shape = tuple(data['header']['shape']); free = data['free_indices']
    x, ref = np.load(xp)[:8], np.load(rp)[:8]
    span = np.repeat(data['span'], 128, 0); lower = np.repeat(data['lower'], 128, 0)
    nominal = x.astype(np.float64)@expected_weights(data)+data['bias']
    energy = np.mean(x.astype(np.float64)**2, axis=(0, 1))
    layer = CalibrationNativeLinear(ARTIFACT, 1)
    probe = float(decode_command(np.array(128, np.uint8)))
    started = time.monotonic(); rows = []
    for seed in protocol['seeds']:
        noise = np.random.default_rng(2026102700+seed).standard_normal((4, len(free)))[0]
        offset = .2*noise
        probe_realized = (probe+offset).astype(np.float32)
        for mi, method in enumerate(protocol['methods']):
            ncal = int(method.split('_')[1]) if method.startswith('measured_') else 0
            counts = None; estimated = None
            name = f'seed{seed}_{method}'
            jp, npz = OUT/'cases'/f'{name}.json', OUT/'cases'/f'{name}.npz'
            if jp.exists():
                row = json.loads(jp.read_text()); assert row['offset_sha256'] == digest(offset) and row['archive_sha256'] == sha(npz)
                rows.append(row); continue
            if ncal:
                key = jax.random.fold_in(jax.random.fold_in(jax.random.key(1040731), seed), ncal)
                counts = np.asarray(layer.compiled_counts(ncal)(key, jnp.asarray(probe_realized)))
                codes, estimated = estimate_command(counts, ncal, data['fields'], probe)
            elif method == 'oracle_offset':
                codes = np.rint((np.clip(data['fields'].astype(np.float64)-offset, -4, 4)+4)*255/8).astype(np.uint8)
            else:
                codes = data['codes'].copy()
            command = decode_command(codes)
            realized = (command.astype(np.float64)+offset).astype(np.float32)
            q = data['probabilities'].reshape(-1).copy()
            q[free] = np.exp(-np.logaddexp(0, -2*realized.astype(np.float64))); q = q.reshape(shape)
            mean = x.astype(np.float64)@(lower+span*q)+data['bias']
            floor = float(np.mean((mean-ref)**2))
            variance = float(np.mean(energy@(span**2*q*(1-q)))/512)
            key = jax.random.fold_in(jax.random.fold_in(jax.random.key(1040730), seed), mi)
            y = np.asarray(layer.compiled_with_fields(512)(key, jnp.asarray(x.reshape(-1, shape[0])), jnp.asarray(realized))).reshape(ref.shape)
            saved = {'output': y, 'codes': codes}
            if counts is not None: saved['counts'] = counts
            np.savez_compressed(npz, **saved)
            requested = data['fields'].astype(np.float64)-(estimated if ncal else offset if method == 'oracle_offset' else 0)
            row = {'seed': seed, 'method': method, 'calibration_samples': ncal, 'inference_samples': 512,
                'offset_sha256': digest(offset), 'realized_field_sha256': digest(realized), 'command_sha256': digest(codes), 'archive_sha256': sha(npz),
                'conditional_bias_floor': floor, 'offset_error_floor_vs_nominal': float(np.mean((mean-nominal)**2)),
                'predicted_sampling_mse': variance, 'predicted_total_mse': floor+variance,
                'observed_sampling_mse': float(np.mean((y.astype(np.float64)-mean)**2)), 'observed_total_mse': float(np.mean((y.astype(np.float64)-ref)**2)),
                'bound_clipped_commands': int(np.count_nonzero(np.abs(requested)>4)),
                'calibration_free_node_evaluations': int(ncal*len(free)), 'inference_free_node_evaluations': int(512*len(free)),
                'inference_binary_products': int(8*65*512*np.prod(shape)), 'stored_command_bytes': int(codes.nbytes),
                'estimated_offset_rmse': float(np.sqrt(np.mean((estimated-offset)**2))) if ncal else None}
            if ncal:
                probe_q = np.exp(-np.logaddexp(0, -2*probe_realized.astype(np.float64)))
                row['calibration_frequency_mse'] = float(np.mean((counts/ncal-probe_q)**2))
                row['calibration_predicted_frequency_mse'] = float(np.mean(probe_q*(1-probe_q))/ncal)
            jp.write_text(json.dumps(row, indent=2)+'\n'); rows.append(row)
            print(f'{name}: MSE={row["observed_total_mse"]:.6g}, floor={floor:.6g}, elapsed={time.monotonic()-started:.1f}s', flush=True)
    assert len(rows) == 50
    all_keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT/'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_keys); w.writeheader(); w.writerows(rows)
    result = {'native_inference_runs': 50, 'native_calibration_runs': 30, 'logical_nodes': int(np.prod(shape)), 'free_nodes': int(len(free)),
        'probe_field': probe, 'nominal_control_bias_floor': float(np.mean((nominal-ref)**2)),
        'total_calibration_free_node_evaluations': int(sum(r['calibration_free_node_evaluations'] for r in rows)),
        'total_inference_free_node_evaluations': int(sum(r['inference_free_node_evaluations'] for r in rows)),
        'elapsed_seconds': time.monotonic()-started, 'scope': protocol['scope']}
    (OUT/'REPORT.json').write_text(json.dumps(result, indent=2)+'\n'); manifest(OUT/'manifest.json', protocol, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
