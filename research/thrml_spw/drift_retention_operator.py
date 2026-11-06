"""Native clocked drift during calibration and inference, with retained/refreshed codes."""
import csv
import hashlib
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .drift_clock import (SEGMENT_SAMPLES,INFERENCE_SAMPLES,CALIBRATION_SAMPLES,AGES,
    QUERY_SEGMENTS,CALIBRATION_SEGMENTS,PROBE_SEGMENTS,TOTAL_SEGMENTS,
    query_start,calibration_start,brownian_path,offset_at)
from .replica_calibration_native import ReplicaCalibrationLinear
from .calibration_control_math import decode_command
from .gain_calibration_math import fit_two_probe,corrected_codes
from .packed_controls import load_controls,expected_weights
from .provenance import manifest
from .verify_transformer_control import sha

OUT=Path('results/thrml_spw/drift_retention_operator')
ARTIFACT=Path('results/thrml_spw/packed_quarter_controls/stage11.npz')
OLD=Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('Pinned native study requires CUDA')
    OUT.mkdir(exist_ok=True)
    for folder in ['calibrations','queries']:(OUT/folder).mkdir(exist_ok=True)
    xp,rp=OLD/'stage11_inputs.npy',OLD/'stage11_conditional_reference.npy'
    protocol={'schema':'spw-clocked-drift-retention-v1','seeds':list(range(10)),
        'drift_sd_per_512_sweep_interval':[0.,.01,.04],'ages_after_initial_calibration':list(AGES),
        'segment_samples':SEGMENT_SAMPLES,'calibration_samples':CALIBRATION_SAMPLES,'inference_samples':INFERENCE_SAMPLES,
        'clock':'One interval is512 modeled Gibbs-sweep ticks. Parameters constant for64-sweep segments; Brownian field offset increments have SD=sigma/sqrt(8) after every segment, including both calibration probes and inference. Idle gaps also advance drift, without emitting unobserved bit states.',
        'query_start_segments':{str(age):query_start(age) for age in AGES},
        'fresh_calibration_start_segments':{str(age):calibration_start(age) for age in AGES},
        'path_segments':TOTAL_SEGMENTS,'probe_codes':[96,159],
        'policies':'Retain initial measured codes, or refresh over the4096 sweeps immediately before the same query clock. At age0 the common initial calibration is the fresh calibration; emit one shared baseline, not a duplicate run.',
        'counterfactuals':'Policies and query ages share the same time-indexed exogenous device path but are distinct virtual schedules. Refreshed probes occupy a gap where retained policy idles. No claim to steady-state throughput or cost-free refresh; programming/fitting time and idle transition energy are not modeled.',
        'device_model':'b_real=float32(a*b_command+d(t)), static a=exp(.2*z-.02), initial d~N(0,.2^2); exact endpoint clamps unperturbed. Synthetic dimensionless errors, not measured hardware.',
        'device_rng':'Initial d=.2*first row of default_rng(2026102700+seed).standard_normal((4,n_free)); a from first row of default_rng(2026102900+seed), same shape, exp(.2*z-.02).',
        'drift_rng':'default_rng(2026103000+seed) standard_normal((328,n_free)); zero initial row then float64 cumsum axis0. d(s)=d0+sigma/sqrt(8)*path[s]. Same standardized path used across sigma scenarios.',
        'calibration':'Each probe spans32 consecutive64-sweep native count segments, totals2048 samples. Fit smoothed-logit line as before (gain guard[.25,4]); counts now pool changing probabilities and are not a stationary-binomial sample. Estimator receives no true parameters, clock path or task labels.',
        'keys':'Probe key1070731 fold seed,sigma_index,start_segment,probe_code,absolute_segment; inference key1070730 fold seed,sigma_index,age,policy_index(0=retain,1=refresh),absolute_segment.',
        'analysis':'Conditional query mean uses average realized probabilities over8 segments. Conditional variance uses average q*(1-q)/512, not qbar*(1-qbar)/512. Random path-to-path mean error remains in the conditional-mean error term; no unconditional1/T claim.',
        'decision':'All fixed ages, policies, drift rates and seeds retained; no selected recalibration interval or success gate. Results are retention probes, not a hardware recalibration frequency in seconds.',
        'artifact_sha256':sha(ARTIFACT),'inputs_sha256':sha(xp),'reference_sha256':sha(rp),
        'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['drift_retention_operator.py','drift_clock.py','replica_calibration_native.py','calibration_count_native.py','controlled_replica.py','replicated_native.py','gain_calibration_math.py','calibration_control_math.py','native.py','packed_controls.py']},
        'scope':'One2,359,296-weight single-p-bit normalized-weight operator, same eight old development contexts. Official THRML per-segment bit updates, no replacement Bernoulli sampler. No new full-model quality or physical energy/latency claim.'}
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
    data=load_controls(ARTIFACT);free=data['free_indices'];shape=tuple(data['header']['shape']);nf=len(free)
    x,ref=np.load(xp)[:8],np.load(rp)[:8]
    lower,span=[np.repeat(data[k],128,axis=0) for k in ['lower','span']]
    nominal=x.astype(float)@expected_weights(data)+data['bias'];energy=np.mean(x.astype(float)**2,axis=(0,1))
    layer=ReplicaCalibrationLinear(ARTIFACT,1);count_fn=layer.compiled_counts(SEGMENT_SAMPLES);infer_fn=layer.compiled_with_fields(SEGMENT_SAMPLES)
    probes={c:float(decode_command(np.array(c,np.uint8))) for c in [96,159]}
    calibration_rows=[];query_rows=[];started=time.monotonic()
    for seed in range(10):
        initial=.2*np.random.default_rng(2026102700+seed).standard_normal((4,nf))[0]
        z=np.random.default_rng(2026102900+seed).standard_normal((4,nf))[0];gain=np.exp(.2*z-.02)
        path=brownian_path(seed,nf)
        print(f'Generated shared drift path seed{seed}; {path.nbytes:,} transient bytes',flush=True)
        for si,sigma in enumerate(protocol['drift_sd_per_512_sweep_interval']):
            event_codes={};event_paths={}
            for start in [0,64,256]:
                name=f'seed{seed}_sigma{si}_start{start}';jp,npz=OUT/'calibrations'/f'{name}.json',OUT/'calibrations'/f'{name}.npz'
                ending=offset_at(initial,path,start+CALIBRATION_SEGMENTS,sigma)
                if jp.exists():
                    rec=json.loads(jp.read_text());assert sha(npz)==rec['archive_sha256'] and digest(ending)==rec['end_offset_sha256']
                    codes=np.load(npz)['codes']
                else:
                    saved={};field_records=[]
                    for probe_index,code in enumerate([96,159]):
                        counts=np.zeros(nf,dtype=np.int32)
                        first=start+probe_index*PROBE_SEGMENTS
                        for segment in range(first,first+PROBE_SEGMENTS):
                            delta=offset_at(initial,path,segment,sigma)
                            physical=(gain*probes[code]+delta).astype(np.float32)
                            key=jax.random.key(1070731)
                            for v in [seed,si,start,code,segment]:key=jax.random.fold_in(key,v)
                            counts+=np.asarray(count_fn(key,jnp.asarray(physical)))
                            field_records.append({'segment':segment,'code':code,'sha256':digest(physical)})
                        saved[f'count_{code}']=counts
                    a_hat,d_hat,raw=fit_two_probe(saved['count_96'],saved['count_159'],2048,probes[96],probes[159])
                    codes,requested=corrected_codes(data['fields'],a_hat,d_hat);saved['codes']=codes
                    np.savez_compressed(npz,**saved)
                    rec={'seed':seed,'sigma_index':si,'sigma':sigma,'start_segment':start,'end_segment':start+64,
                        'archive_sha256':sha(npz),'command_sha256':digest(codes),'end_offset_sha256':digest(ending),'gain_sha256':digest(gain),
                        'segment_fields':field_records,'estimated_gain_rmse':float(np.sqrt(np.mean((a_hat-gain)**2))),
                        'estimated_offset_rmse_at_end':float(np.sqrt(np.mean((d_hat-ending)**2))),
                        'gain_guard_low':int(np.count_nonzero(raw<.25)),'gain_guard_high':int(np.count_nonzero(raw>4)),
                        'bound_clipped_commands':int(np.count_nonzero(np.abs(requested)>4)),
                        'native_count_segments':64,'native_probe_windows':2,'calibration_free_node_evaluations':int(4096*nf)}
                    jp.write_text(json.dumps(rec,indent=2)+'\n')
                    print(f'Calibration {name}: endpoint offset RMSE={rec["estimated_offset_rmse_at_end"]:.5g}',flush=True)
                calibration_rows.append(rec);event_codes[start]=codes;event_paths[start]=npz
            for age in AGES:
                qs=query_start(age)
                for pi,policy in enumerate(['retain'] if age==0 else ['retain','refresh']):
                    start=0 if policy=='retain' else calibration_start(age)
                    codes=event_codes[start];source=event_paths[start]
                    name=f'seed{seed}_sigma{si}_age{age}_{policy}';jp,npz=OUT/'queries'/f'{name}.json',OUT/'queries'/f'{name}.npz'
                    if jp.exists():
                        rec=json.loads(jp.read_text());assert sha(npz)==rec['archive_sha256'] and rec['calibration_source_sha256']==sha(source)
                        query_rows.append(rec);continue
                    command=decode_command(codes).astype(float);qsum=np.zeros(nf);vsum=np.zeros(nf);ysum=np.zeros(ref.shape,np.float32);field_records=[]
                    for segment in range(qs,qs+QUERY_SEGMENTS):
                        delta=offset_at(initial,path,segment,sigma);physical=(gain*command+delta).astype(np.float32)
                        key=jax.random.key(1070730)
                        for v in [seed,si,age,pi,segment]:key=jax.random.fold_in(key,v)
                        yy=np.asarray(infer_fn(key,jnp.asarray(x.reshape(-1,shape[0])),jnp.asarray(physical))).reshape(ref.shape);ysum+=yy
                        q=np.exp(-np.logaddexp(0,-2*physical.astype(float)));qsum+=q;vsum+=q*(1-q)
                        field_records.append({'segment':segment,'sha256':digest(physical)})
                    y=ysum/QUERY_SEGMENTS;qbar=data['probabilities'].reshape(-1).copy();qbar[free]=qsum/QUERY_SEGMENTS
                    v=np.zeros(int(np.prod(shape)));v[free]=vsum/QUERY_SEGMENTS
                    mean=x.astype(float)@(lower+span*qbar.reshape(shape))+data['bias']
                    mean_error=float(np.mean((mean-ref)**2));variance=float(np.mean(energy@(span**2*v.reshape(shape)))/512)
                    np.savez_compressed(npz,output=y)
                    active_calibrations=1+(policy=='refresh')
                    rec={'seed':seed,'sigma_index':si,'sigma':sigma,'age':age,'policy':policy,'query_start_segment':qs,'query_end_segment':qs+8,
                        'calibration_source':str(source),'calibration_source_sha256':sha(source),'archive_sha256':sha(npz),
                        'segment_fields':field_records,'start_offset_sha256':digest(offset_at(initial,path,qs,sigma)),
                        'end_offset_sha256':digest(offset_at(initial,path,qs+8,sigma)),
                        'conditional_mean_error_mse':mean_error,'predicted_sampling_mse':variance,'predicted_total_mse':mean_error+variance,
                        'observed_total_mse':float(np.mean((y.astype(float)-ref)**2)),
                        'observed_sampling_mse':float(np.mean((y.astype(float)-mean)**2)),
                        'error_vs_nominal_programmed_mean':float(np.mean((mean-nominal)**2)),
                        'native_linear_segments':8,'inference_free_node_evaluations':int(512*nf),
                        'counterfactual_calibration_events':int(active_calibrations),'counterfactual_calibration_free_evaluations':int(active_calibrations*4096*nf),
                        'counterfactual_active_sweeps':int(active_calibrations*4096+512),
                        'counterfactual_idle_sweeps':int((qs-active_calibrations*64)*64),
                        'query_end_clock_sweeps':int((qs+8)*64)}
                    jp.write_text(json.dumps(rec,indent=2)+'\n');query_rows.append(rec)
                    print(f'Query {name}: MSE={rec["observed_total_mse"]:.6g}, mean-error={mean_error:.6g}, elapsed={time.monotonic()-started:.1f}s',flush=True)
        del path
    assert len(calibration_rows)==90 and len(query_rows)==150
    for filename,rows in [('calibration_metrics.csv',calibration_rows),('query_metrics.csv',query_rows)]:
        scalars=[{k:v for k,v in r.items() if k!='segment_fields'} for r in rows]
        with (OUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(scalars[0]));w.writeheader();w.writerows(scalars)
    result={'native_query_windows':150,'native_linear_segments':1200,'native_calibration_events':90,'native_probe_windows':180,'native_count_segments':5760,
        'unique_weights':int(np.prod(shape)),'free_nodes':nf,'nominal_control_bias_floor':float(np.mean((nominal-ref)**2)),
        'executed_calibration_free_node_evaluations':int(sum(r['calibration_free_node_evaluations'] for r in calibration_rows)),
        'executed_query_free_node_evaluations':int(sum(r['inference_free_node_evaluations'] for r in query_rows)),
        'elapsed_seconds':time.monotonic()-started,'scope':protocol['scope']}
    (OUT/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');manifest(OUT/'manifest.json',protocol,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
