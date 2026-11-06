"""Independent raw audit of calibrated equal-resource temporal/spatial banks."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha

OUT=Path('results/thrml_spw/calibrated_matched_operator')


def raw_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p=json.loads((OUT/'protocol.json').read_text());report=json.loads((OUT/'REPORT.json').read_text());run=json.loads((OUT/'manifest.json').read_text())
    assert run['config']==p and run['result']==report and run['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for path,h in p['source_sha256'].items():assert sha(path)==h
    assert p['replicas']==[1,4] and p['seeds']==list(range(10)) and p['total_probe_draws_per_unique_weight']==[256,1024,4096]
    assert p['total_inference_draws_per_unique_weight']==512 and p['probe_codes']==[96,159]
    tests=json.loads((OUT/'trajectory_test_record.json').read_text());assert tests['passed']
    assert sha('research/thrml_spw/test_replica_calibration_native.py')==tests['test_source_sha256']
    assert sha('research/thrml_spw/replica_calibration_native.py')==tests['native_source_sha256']
    for dev in ['cpu','gpu']:assert 'OK' in (OUT/f'{dev}_trajectory_test.log').read_text()
    artifact=Path('results/thrml_spw/packed_quarter_controls/stage11.npz');old=Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')
    xp,rp=old/'stage11_inputs.npy',old/'stage11_conditional_reference.npy'
    for path,key in [(artifact,'artifact_sha256'),(xp,'inputs_sha256'),(rp,'reference_sha256')]:assert sha(path)==p[key]
    x,ref=np.load(xp)[:8].astype(float),np.load(rp)[:8].astype(float)
    with np.load(artifact) as a:
        h=json.loads(a['header'].tobytes());shape=tuple(h['shape']);n=int(np.prod(shape));assert shape==(3072,768)
        bits=np.unpackbits(a['modes'],bitorder='little').reshape(-1,2)[:n];modes=bits@np.array([1,2],np.uint8);free=np.flatnonzero(modes==0)
        base_codes=a['codes'].copy();base_field=(-4+base_codes.astype(float)*8/255).astype(np.float32)
        lower=np.repeat(a['lower'].astype(float),128,axis=0);span=np.repeat(a['span'].astype(float),128,axis=0);bias=a['bias'].copy()
    q0=(modes==2).astype(float);q0[free]=1/(1+np.exp(-2*base_field.astype(float)));nominal=x@(lower+span*q0.reshape(shape))+bias
    np.testing.assert_allclose(np.mean((nominal-ref)**2),report['nominal_control_bias_floor'],rtol=1e-10)
    with (OUT/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    methods=['uncalibrated','oracle','cal256','cal1024','cal4096'];assert p['methods']==methods
    assert len(rows)==report['native_inference_runs']==100
    assert {(int(r['K']),int(r['seed']),r['method']) for r in rows}=={(k,s,m) for k in [1,4] for s in range(10) for m in methods}
    assert len(list((OUT/'cases').glob('*.json')))==len(list((OUT/'cases').glob('*.npz')))==100
    probes={c:float(np.float32(-4+c*8/255)) for c in [96,159]};checked_probes=0
    for k in [1,4]:
        ctrl=report['controls'][str(k)];assert ctrl['logical_pbits']==k*n and ctrl['free_pbits']==k*len(free)
        for seed in range(10):
            delta=(.2*np.random.default_rng(2026102700+seed).standard_normal((4,len(free)))[:k]).reshape(-1)
            z=np.random.default_rng(2026102900+seed).standard_normal((4,len(free)))[:k]
            gain=np.exp(.2*z-.02).reshape(-1);b0=np.tile(base_field,k)
            for method in methods:
                row=next(r for r in rows if int(r['K'])==k and int(r['seed'])==seed and r['method']==method)
                prefix=OUT/'cases'/f'K{k}_seed{seed}_{method}';case=json.loads(prefix.with_suffix('.json').read_text())
                for key,value in case.items():assert ('' if value is None else str(value))==row[key]
                assert raw_sha(delta)==row['offset_sha256'] and raw_sha(gain)==row['gain_sha256'] and sha(prefix.with_suffix('.npz'))==row['archive_sha256']
                archive=np.load(prefix.with_suffix('.npz'));codes=archive['codes'];y=archive['output'].astype(float)
                assert codes.dtype==np.uint8 and codes.shape==(k*len(free),) and y.shape==ref.shape and np.isfinite(y).all()
                budget=int(method[3:]) if method.startswith('cal') else 0;assert int(row['P'])==budget
                if budget:
                    each=budget//(2*k);u={};frequency_mse=[];prediction=[]
                    assert set(archive.files)=={'codes','output','count_96','count_159'}
                    for c in [96,159]:
                        counts=archive[f'count_{c}'];assert counts.dtype==np.int32 and counts.shape==codes.shape and np.all((counts>=0)&(counts<=each))
                        u[c]=.5*(np.log(counts.astype(float)+.5)-np.log(each-counts.astype(float)+.5))
                        physical=(gain*probes[c]+delta).astype(np.float32);qp=1/(1+np.exp(-2*physical.astype(float)))
                        frequency_mse.append(np.mean((counts/each-qp)**2));prediction.append(np.mean(qp*(1-qp))/each);checked_probes+=1
                    raw=(u[159]-u[96])/(probes[159]-probes[96]);a_hat=np.minimum(4.,np.maximum(.25,raw));d_hat=(u[159]+u[96])/2-raw*(probes[159]+probes[96])/2
                    requested=(b0.astype(float)-d_hat)/a_hat
                    np.testing.assert_allclose(float(row['estimated_gain_rmse']),np.sqrt(np.mean((a_hat-gain)**2)),rtol=1e-12)
                    np.testing.assert_allclose(float(row['estimated_offset_rmse']),np.sqrt(np.mean((d_hat-delta)**2)),rtol=1e-12)
                    np.testing.assert_allclose(float(row['calibration_frequency_mse']),np.mean(frequency_mse),rtol=1e-12)
                    np.testing.assert_allclose(float(row['calibration_predicted_frequency_mse']),np.mean(prediction),rtol=1e-12)
                    assert int(row['gain_guard_low'])==np.count_nonzero(raw<.25) and int(row['gain_guard_high'])==np.count_nonzero(raw>4.)
                else:
                    assert set(archive.files)=={'codes','output'}
                    assert row['estimated_gain_rmse']==row['estimated_offset_rmse']==''
                    requested=(b0.astype(float)-delta)/gain if method=='oracle' else b0.astype(float)
                desired=np.rint((np.clip(requested,-4,4)+4)*255/8).astype(np.uint8);np.testing.assert_array_equal(codes,desired)
                assert raw_sha(codes)==row['command_sha256']
                realized=(gain*(-4+codes.astype(float)*8/255).astype(np.float32).astype(float)+delta).astype(np.float32)
                assert raw_sha(realized)==row['realized_field_sha256']
                q=np.repeat((modes==2).astype(float)[None],k,axis=0);q[:,free]=1/(1+np.exp(-2*realized.astype(float).reshape(k,-1)));q=q.reshape(k,*shape)
                mean=x@(lower+span*np.add.reduce(q,axis=0)/k)+bias;floor=float(np.mean((mean-ref)**2))
                weight_variance=span**2*np.add.reduce(q-q*q,axis=0)/(k*k)
                variance=float(np.mean(np.mean(x*x,axis=(0,1))@weight_variance)/(512//k))
                for key,value in [('conditional_bias_floor',floor),('offset_error_floor_vs_nominal',float(np.mean((mean-nominal)**2))),
                    ('predicted_sampling_mse',variance),('predicted_total_mse',floor+variance),
                    ('observed_sampling_mse',float(np.mean((y-mean)**2))),('observed_total_mse',float(np.mean((y-ref)**2)))]:
                    np.testing.assert_allclose(float(row[key]),value,rtol=4e-7,atol=1e-12)
                assert int(row['bound_clipped_commands'])==np.count_nonzero(np.abs(requested)>4)
                assert int(row['probe_samples_per_device_per_setting'])==budget//(2*k) and int(row['inference_samples_per_device'])==512//k
                assert int(row['calibration_free_node_evaluations'])==budget*len(free) and int(row['inference_free_node_evaluations'])==512*len(free)
                assert int(row['inference_logical_reads'])==512*n and int(row['inference_binary_products'])==8*65*512*n
                assert int(row['logical_nodes'])==k*n and int(row['free_nodes'])==k*len(free)
                assert int(row['stored_command_bytes'])==k*len(free) and int(row['base_float32_field_bytes'])==4*k*len(free)
                assert int(row['calibration_count_bytes'])==(8*k*len(free) if budget else 0)
                assert int(row['replica_output_accumulator_bytes'])==4*k*8*65*shape[1]
            print(f'Independent calibrated-matched audit K{k} seed{seed}',flush=True)
    assert checked_probes==report['native_calibration_probe_runs']==120
    assert report['total_calibration_free_node_evaluations']==20*(256+1024+4096)*len(free)
    assert report['total_inference_free_node_evaluations']==100*512*len(free)
    audit={'passed':True,'native_inference_runs':100,'native_calibration_probe_runs':120,
        'all_counts_codes_outputs_and_resource_counters_verified':True,'equal_total_calibration_and_inference_draws_verified':True,
        'source_sha256':sha(__file__),'scope':'Independent packed-bit/parameter decoding, fitted controls and every output/conditional error, logical node and storage/depth counter. Explicit replicas with synthetic static errors, not physical performance or an official learned-weight comparator.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
