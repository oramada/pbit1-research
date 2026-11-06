"""Calibrated K1/K4 native operators at equal total probe and inference draws."""
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .replica_calibration_native import ReplicaCalibrationLinear
from .calibration_control_math import decode_command
from .gain_calibration_math import fit_two_probe, corrected_codes
from .packed_controls import load_controls, expected_weights
from .provenance import manifest
from .verify_transformer_control import sha

OUT = Path('results/thrml_spw/calibrated_matched_operator')
ARTIFACT = Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
OLD = Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    if jax.default_backend() != 'gpu': raise RuntimeError('This pinned study requires CUDA')
    OUT.mkdir(exist_ok=True); (OUT/'cases').mkdir(exist_ok=True)
    xp, rp = OLD/'stage11_inputs.npy', OLD/'stage11_conditional_reference.npy'
    methods = ['uncalibrated', 'oracle', 'cal256', 'cal1024', 'cal4096']
    protocol = {'schema': 'spw-calibrated-spatial-temporal-v1', 'replicas': [1,4], 'seeds': list(range(10)),
        'total_probe_draws_per_unique_weight': [256,1024,4096], 'total_inference_draws_per_unique_weight': 512,
        'methods': methods, 'contexts': 8, 'probe_codes': [96,159],
        'device_model': 'b_real=float32(a*b_command+d); d~N(0,0.2^2), a=exp(0.2*z-0.2^2/2). Static independent device parameters across replicas and weights. Synthetic dimensionless scenarios, not measured specifications.',
        'device_rng': 'd=.2*default_rng(2026102700+seed).standard_normal((4,n_free)); a uses default_rng(2026102900+seed) same shape, exp(.2*z-.02). K1 uses first row, K4 all rows; paired first device, not identical sample paths.',
        'calibration': 'Two finite eight-bit probes, P/(2*K) samples each on K*n_free nodes. Jeffreys smoothing and logit line fit; gain guard[.25,4], inverse commands nearest eight-bit codes in[-4,4]. No true parameters or task labels enter measured estimator.',
        'inference': 'T=512/K native samples and replica output mean; accumulation before nonlinearities. Equal total device evaluations, logical reads and nominal binary products at fixed P.',
        'oracle': 'Known true device parameters for diagnostic inverse only, with same finite controller. No measured implementation or optimality claim.',
        'keys': 'Inference key1060730 fold K,seed,method_index; probe key1060731 fold K,seed,P,probe_code. Disjoint streams; no path pairing.',
        'endpoint_policy': 'Exact clamps shared by the encoded target but represented by distinct logical nodes in each replica; no calibration perturbation of endpoints.',
        'decision': 'Execute and report all conditions; no budget selection, success gate or significance-driven extension. Pair contrasts by device seed. Do not equate equal draw counts with equal energy or equal physical parallelism.',
        'artifact_sha256': sha(ARTIFACT), 'inputs_sha256': sha(xp), 'reference_sha256': sha(rp),
        'source_sha256': {str(Path('research/thrml_spw')/n): sha(Path('research/thrml_spw')/n) for n in ['calibrated_matched_operator.py','replica_calibration_native.py','gain_calibration_math.py','calibration_control_math.py','calibration_count_native.py','controlled_replica.py','replicated_native.py','native.py','packed_controls.py']},
        'scope': 'One 2,359,296-weight normalized-weight operator on eight old development contexts. Explicit independent replicas, not positional bits or an official Extropic learned-weight reference. No new model-quality or physical-system claim.'}
    pp = OUT/'protocol.json'
    if pp.exists(): assert json.loads(pp.read_text()) == protocol
    else: pp.write_text(json.dumps(protocol,indent=2)+'\n')
    data = load_controls(ARTIFACT); shape = tuple(data['header']['shape']); free = data['free_indices']
    x, ref = np.load(xp)[:8], np.load(rp)[:8]
    lower, span = [np.repeat(data[key],128,axis=0) for key in ['lower','span']]
    nominal = x.astype(np.float64)@expected_weights(data)+data['bias']; energy = np.mean(x.astype(np.float64)**2,axis=(0,1))
    probes = {code:float(decode_command(np.array(code,np.uint8))) for code in [96,159]}
    rows=[]; controls={}; started=time.monotonic()
    for k in [1,4]:
        layer=ReplicaCalibrationLinear(ARTIFACT,k); bank=layer.bank
        assert len(bank.free_indices)==k*len(free)
        controls[str(k)]=layer.controls()
        count_functions={p//(2*k):layer.compiled_counts(p//(2*k)) for p in [256,1024,4096]}
        infer=layer.compiled_with_fields(512//k); nominal_fields=np.tile(data['fields'],k)
        for seed in range(10):
            delta=(.2*np.random.default_rng(2026102700+seed).standard_normal((4,len(free)))[:k]).reshape(-1)
            z=np.random.default_rng(2026102900+seed).standard_normal((4,len(free)))[:k]
            gain=np.exp(.2*z-.02).reshape(-1)
            for mi,method in enumerate(methods):
                p=int(method[3:]) if method.startswith('cal') else 0
                name=f'K{k}_seed{seed}_{method}'; jp,npz=OUT/'cases'/f'{name}.json',OUT/'cases'/f'{name}.npz'
                if jp.exists():
                    row=json.loads(jp.read_text());assert row['gain_sha256']==digest(gain) and row['offset_sha256']==digest(delta) and row['archive_sha256']==sha(npz)
                    rows.append(row);continue
                saved={};freq_mse=[];freq_prediction=[];raw_gain=None
                if p:
                    each=p//(2*k)
                    for code in [96,159]:
                        physical=(gain*probes[code]+delta).astype(np.float32)
                        key=jax.random.key(1060731)
                        for v in [k,seed,p,code]:key=jax.random.fold_in(key,v)
                        c=np.asarray(count_functions[each](key,jnp.asarray(physical)));saved[f'count_{code}']=c
                        qp=np.exp(-np.logaddexp(0,-2*physical.astype(np.float64)))
                        freq_mse.append(float(np.mean((c/each-qp)**2)));freq_prediction.append(float(np.mean(qp*(1-qp))/each))
                    gain_hat,delta_hat,raw_gain=fit_two_probe(saved['count_96'],saved['count_159'],each,probes[96],probes[159])
                    codes,requested=corrected_codes(nominal_fields,gain_hat,delta_hat)
                elif method=='oracle':codes,requested=corrected_codes(nominal_fields,gain,delta)
                else:codes=np.tile(data['codes'],k);requested=nominal_fields.astype(np.float64)
                realized=(gain*decode_command(codes).astype(np.float64)+delta).astype(np.float32)
                q=np.broadcast_to(data['probabilities'][None],(k,*shape)).reshape(k,-1).copy()
                q[:,free]=np.exp(-np.logaddexp(0,-2*realized.reshape(k,-1).astype(np.float64)));q=q.reshape(k,*shape)
                mean=x.astype(np.float64)@(lower+span*q.mean(0))+data['bias']
                floor=float(np.mean((mean-ref)**2));variance=float(np.mean(energy@(span**2*np.sum(q*(1-q),axis=0)/(k*k)))/(512//k))
                key=jax.random.key(1060730)
                for v in [k,seed,mi]:key=jax.random.fold_in(key,v)
                y=np.asarray(infer(key,jnp.asarray(x.reshape(-1,shape[0])),jnp.asarray(realized))).reshape(ref.shape)
                saved.update(codes=codes,output=y);np.savez_compressed(npz,**saved)
                row={'K':k,'seed':seed,'method':method,'P':p,'probe_samples_per_device_per_setting':p//(2*k),'inference_samples_per_device':512//k,
                    'gain_sha256':digest(gain),'offset_sha256':digest(delta),'realized_field_sha256':digest(realized),'command_sha256':digest(codes),'archive_sha256':sha(npz),
                    'conditional_bias_floor':floor,'offset_error_floor_vs_nominal':float(np.mean((mean-nominal)**2)),
                    'predicted_sampling_mse':variance,'predicted_total_mse':floor+variance,
                    'observed_sampling_mse':float(np.mean((y.astype(np.float64)-mean)**2)),'observed_total_mse':float(np.mean((y.astype(np.float64)-ref)**2)),
                    'gain_guard_low':int(np.count_nonzero(raw_gain<.25)) if p else 0,'gain_guard_high':int(np.count_nonzero(raw_gain>4.)) if p else 0,
                    'bound_clipped_commands':int(np.count_nonzero(np.abs(requested)>4)),
                    'estimated_gain_rmse':float(np.sqrt(np.mean((gain_hat-gain)**2))) if p else None,
                    'estimated_offset_rmse':float(np.sqrt(np.mean((delta_hat-delta)**2))) if p else None,
                    'calibration_frequency_mse':float(np.mean(freq_mse)) if p else None,'calibration_predicted_frequency_mse':float(np.mean(freq_prediction)) if p else None,
                    'calibration_free_node_evaluations':int(p*len(free)),'inference_free_node_evaluations':int(512*len(free)),
                    'inference_logical_reads':int(512*np.prod(shape)),'inference_binary_products':int(8*65*512*np.prod(shape)),
                    'logical_nodes':bank.logical_pbits,'free_nodes':len(bank.free_indices),'stored_command_bytes':int(codes.nbytes),
                    'base_float32_field_bytes':int(bank.biases.nbytes),'calibration_count_bytes':int(sum(a.nbytes for key,a in saved.items() if key.startswith('count_'))),
                    'replica_output_accumulator_bytes':int(k*x.shape[0]*x.shape[1]*shape[1]*4)}
                jp.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
                print(f'{name}: MSE={row["observed_total_mse"]:.6g}, floor={floor:.6g}, elapsed={time.monotonic()-started:.1f}s',flush=True)
        del layer
        jax.clear_caches();gc.collect()
    assert len(rows)==100
    for method in methods:
        group=[r for r in rows if r['method']==method]
        for counter in ['calibration_free_node_evaluations','inference_free_node_evaluations','inference_logical_reads','inference_binary_products']:
            assert len({r[counter] for r in group})==1
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'native_inference_runs':100,'native_calibration_probe_runs':120,'unique_weights':int(np.prod(shape)),'unique_free_weights':int(len(free)),
        'controls':controls,'nominal_control_bias_floor':float(np.mean((nominal-ref)**2)),
        'total_calibration_free_node_evaluations':int(sum(r['calibration_free_node_evaluations'] for r in rows)),
        'total_inference_free_node_evaluations':int(sum(r['inference_free_node_evaluations'] for r in rows)),
        'elapsed_seconds':time.monotonic()-started,'scope':protocol['scope']}
    (OUT/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');manifest(OUT/'manifest.json',protocol,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
