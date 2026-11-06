"""Native equal-probe-budget one/two-point calibration under gain and offset."""
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
from .gain_calibration_math import fit_two_probe, corrected_codes
from .packed_controls import load_controls, expected_weights
from .provenance import manifest
from .verify_transformer_control import sha

OUT = Path('results/thrml_spw/gain_calibrated_operator')
ARTIFACT = Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
OLD = Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    if jax.default_backend() != 'gpu': raise RuntimeError('Pinned native study requires CUDA')
    OUT.mkdir(exist_ok=True); (OUT/'cases').mkdir(exist_ok=True)
    xp, rp = OLD/'stage11_inputs.npy', OLD/'stage11_conditional_reference.npy'
    methods = ['uncalibrated', 'oracle', 'single256', 'two256', 'single1024', 'two1024', 'single4096', 'two4096']
    protocol = {'schema': 'spw-native-gain-offset-calibration-v1', 'log_gain_sd': [0., .2], 'offset_sd': .2,
        'seeds': list(range(10)), 'total_probe_samples': [256, 1024, 4096], 'inference_samples': 512, 'contexts': 8,
        'methods': methods, 'single_probe_code': 128, 'two_probe_codes': [96, 159],
        'device_model': 'b_real=float32(gain*b_command+offset); gain=exp(log_gain_sd*z-0.5*log_gain_sd^2), mean gain1; offsets fixed N(0,0.2^2). Synthetic dimensionless scenarios, not measured hardware.',
        'device_rng': 'offset=0.2*first row of default_rng(2026102700+seed).standard_normal((4,n_free)); gain z=default_rng(2026102900+seed).standard_normal(n_free)',
        'single_probe': 'All Ncal samples at code128. Jeffreys-smoothed logit offset estimate assumes gain1, same earlier measured procedure. No true device parameters passed.',
        'two_probe': 'Ncal/2 samples at each code96/159. Fit u=0.5*logit((C+0.5)/(Ncal/2+1)) to u=gain*b+offset. Estimated gain clipped to[0.25,4] as an explicit estimator guard. No true parameters passed.',
        'correction': 'Round nearest eight-bit code of clip((nominal_field-offset_hat)/gain_hat,-4,4). Exact endpoint clamps remain unperturbed; all free nodes retained.',
        'oracle': 'Known true gain and offset only for a diagnostic inverse with same eight-bit range; not measured procedure or guaranteed global optimum.',
        'keys': 'Inference key1050730 fold scenario_index,seed,method_index. Probe key1050731 fold scenario_index,seed,total_Ncal,probe_code. Fresh disjoint streams; fixed device realization paired across methods.',
        'decision': 'No budget/model selection or success gate. Report all scenarios, all methods and all finite-control/estimator clipping counts. Compare methods at equal total probe draws.',
        'artifact_sha256': sha(ARTIFACT), 'inputs_sha256': sha(xp), 'reference_sha256': sha(rp),
        'source_sha256': {str(Path('research/thrml_spw')/n): sha(Path('research/thrml_spw')/n) for n in ['gain_calibrated_operator.py', 'gain_calibration_math.py', 'calibration_count_native.py', 'calibration_control_math.py', 'controlled_replica.py', 'replicated_native.py', 'native.py', 'packed_controls.py']},
        'scope': 'One 2,359,296-weight normalized-weight operator on old consumed development activations. Official THRML counts and accumulation before nonlinearities; no new model quality, device measurements, drift, area or energy claim.'}
    pp = OUT/'protocol.json'
    if pp.exists(): assert json.loads(pp.read_text()) == protocol
    else: pp.write_text(json.dumps(protocol, indent=2)+'\n')
    data = load_controls(ARTIFACT); free = data['free_indices']; shape = tuple(data['header']['shape'])
    x, ref = np.load(xp)[:8], np.load(rp)[:8]
    lower, span = [np.repeat(data[k], 128, 0) for k in ['lower', 'span']]
    nominal = x.astype(np.float64)@expected_weights(data)+data['bias']; energy = np.mean(x.astype(np.float64)**2, axis=(0, 1))
    layer = CalibrationNativeLinear(ARTIFACT, 1)
    count_functions = {n: layer.compiled_counts(n) for n in [128, 256, 512, 1024, 2048, 4096]}
    infer = layer.compiled_with_fields(512)
    probe_fields = {code: float(decode_command(np.array(code, np.uint8))) for code in [96, 128, 159]}
    rows = []; started = time.monotonic()
    for scenario, log_sd in enumerate(protocol['log_gain_sd']):
        for seed in protocol['seeds']:
            offset = .2*np.random.default_rng(2026102700+seed).standard_normal((4, len(free)))[0]
            gnoise = np.random.default_rng(2026102900+seed).standard_normal(len(free))
            gain = np.exp(log_sd*gnoise-.5*log_sd**2)
            for mi, method in enumerate(methods):
                total = next((n for n in [256, 1024, 4096] if method.endswith(str(n))), 0)
                name = f'scenario{scenario}_seed{seed}_{method}'; jp, npz = OUT/'cases'/f'{name}.json', OUT/'cases'/f'{name}.npz'
                if jp.exists():
                    row = json.loads(jp.read_text()); assert row['gain_sha256'] == digest(gain) and row['offset_sha256'] == digest(offset) and row['archive_sha256'] == sha(npz)
                    rows.append(row); continue
                saved = {}; freq_mse = []; freq_theory = []; raw_gain = None
                if total:
                    probes = [128] if method.startswith('single') else [96, 159]
                    n_each = total//len(probes)
                    for code in probes:
                        physical = (gain*probe_fields[code]+offset).astype(np.float32)
                        key = jax.random.key(1050731)
                        for value in [scenario, seed, total, code]: key = jax.random.fold_in(key, value)
                        c = np.asarray(count_functions[n_each](key, jnp.asarray(physical))); saved[f'count_{code}'] = c
                        pq = np.exp(-np.logaddexp(0, -2*physical.astype(np.float64)))
                        freq_mse.append(float(np.mean((c/n_each-pq)**2))); freq_theory.append(float(np.mean(pq*(1-pq))/n_each))
                    if method.startswith('single'):
                        codes, offset_hat = estimate_command(saved['count_128'], total, data['fields'], probe_fields[128])
                        gain_hat = np.ones(len(free)); requested = data['fields'].astype(np.float64)-offset_hat
                    else:
                        gain_hat, offset_hat, raw_gain = fit_two_probe(saved['count_96'], saved['count_159'], n_each, probe_fields[96], probe_fields[159])
                        codes, requested = corrected_codes(data['fields'], gain_hat, offset_hat)
                elif method == 'oracle':
                    codes, requested = corrected_codes(data['fields'], gain, offset)
                else:
                    codes = data['codes'].copy(); requested = data['fields'].astype(np.float64)
                realized = (gain*decode_command(codes).astype(np.float64)+offset).astype(np.float32)
                q = data['probabilities'].reshape(-1).copy(); q[free] = np.exp(-np.logaddexp(0, -2*realized.astype(np.float64))); q = q.reshape(shape)
                mean = x.astype(np.float64)@(lower+span*q)+data['bias']
                floor = float(np.mean((mean-ref)**2)); variance = float(np.mean(energy@(span**2*q*(1-q)))/512)
                key = jax.random.key(1050730)
                for value in [scenario, seed, mi]: key = jax.random.fold_in(key, value)
                y = np.asarray(infer(key, jnp.asarray(x.reshape(-1, shape[0])), jnp.asarray(realized))).reshape(ref.shape)
                saved.update(codes=codes, output=y); np.savez_compressed(npz, **saved)
                row = {'scenario': scenario, 'log_gain_sd': log_sd, 'seed': seed, 'method': method, 'total_probe_samples': total,
                    'probe_count': len(probes) if total else 0, 'gain_sha256': digest(gain), 'offset_sha256': digest(offset),
                    'realized_field_sha256': digest(realized), 'command_sha256': digest(codes), 'archive_sha256': sha(npz),
                    'conditional_bias_floor': floor, 'offset_error_floor_vs_nominal': float(np.mean((mean-nominal)**2)),
                    'predicted_sampling_mse': variance, 'predicted_total_mse': floor+variance,
                    'observed_sampling_mse': float(np.mean((y.astype(np.float64)-mean)**2)), 'observed_total_mse': float(np.mean((y.astype(np.float64)-ref)**2)),
                    'bound_clipped_commands': int(np.count_nonzero(np.abs(requested)>4)),
                    'gain_guard_low': int(np.count_nonzero(raw_gain < .25)) if raw_gain is not None else 0,
                    'gain_guard_high': int(np.count_nonzero(raw_gain > 4.)) if raw_gain is not None else 0,
                    'estimated_gain_rmse': float(np.sqrt(np.mean((gain_hat-gain)**2))) if total else None,
                    'estimated_offset_rmse': float(np.sqrt(np.mean((offset_hat-offset)**2))) if total else None,
                    'calibration_frequency_mse': float(np.mean(freq_mse)) if total else None,
                    'calibration_predicted_frequency_mse': float(np.mean(freq_theory)) if total else None,
                    'calibration_free_node_evaluations': int(total*len(free)), 'inference_free_node_evaluations': int(512*len(free)),
                    'inference_binary_products': int(8*65*512*np.prod(shape)), 'stored_command_bytes': int(codes.nbytes)}
                jp.write_text(json.dumps(row, indent=2)+'\n'); rows.append(row)
                print(f'{name}: MSE={row["observed_total_mse"]:.6g}, floor={floor:.6g}, elapsed={time.monotonic()-started:.1f}s', flush=True)
    assert len(rows) == 160
    with (OUT/'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result = {'native_inference_runs': 160, 'native_calibration_runs': int(sum(r['probe_count'] for r in rows)),
        'logical_nodes': int(np.prod(shape)), 'free_nodes': int(len(free)), 'probe_fields': probe_fields,
        'nominal_control_bias_floor': float(np.mean((nominal-ref)**2)),
        'total_calibration_free_node_evaluations': int(sum(r['calibration_free_node_evaluations'] for r in rows)),
        'total_inference_free_node_evaluations': int(sum(r['inference_free_node_evaluations'] for r in rows)),
        'elapsed_seconds': time.monotonic()-started, 'scope': protocol['scope']}
    (OUT/'REPORT.json').write_text(json.dumps(result, indent=2)+'\n'); manifest(OUT/'manifest.json', protocol, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
