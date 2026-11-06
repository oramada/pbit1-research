"""Independent streaming-path audit of clocked drift, counts and query errors."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha

OUT=Path('results/thrml_spw/drift_retention_operator')


def raw_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def table(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    p=json.loads((OUT/'protocol.json').read_text());report=json.loads((OUT/'REPORT.json').read_text());run=json.loads((OUT/'manifest.json').read_text())
    assert run['config']==p and run['result']==report and run['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for path,h in p['source_sha256'].items():assert sha(path)==h
    assert p['seeds']==list(range(10)) and p['drift_sd_per_512_sweep_interval']==[0.,.01,.04]
    assert p['ages_after_initial_calibration']==[0,8,32] and p['segment_samples']==64 and p['path_segments']==328
    assert p['calibration_samples']==4096 and p['inference_samples']==512 and p['probe_codes']==[96,159]
    assert p['query_start_segments']=={'0':64,'8':128,'32':320} and p['fresh_calibration_start_segments']=={'0':0,'8':64,'32':256}
    tests=json.loads((OUT/'trajectory_test_record.json').read_text());assert tests['passed']
    assert tests['test_source_sha256']==sha('research/thrml_spw/test_drift_clock.py') and tests['clock_source_sha256']==sha('research/thrml_spw/drift_clock.py')
    for dev in ['cpu','gpu']:assert 'OK' in (OUT/f'{dev}_trajectory_test.log').read_text()
    artifact=Path('results/thrml_spw/packed_quarter_controls/stage11.npz');old=Path('results/thrml_spw/transformer_quarter/runs/validation_T512_seed0')
    xp,rp=old/'stage11_inputs.npy',old/'stage11_conditional_reference.npy'
    for path,key in [(artifact,'artifact_sha256'),(xp,'inputs_sha256'),(rp,'reference_sha256')]:assert sha(path)==p[key]
    x,ref=np.load(xp)[:8].astype(float),np.load(rp)[:8].astype(float)
    with np.load(artifact) as a:
        header=json.loads(a['header'].tobytes());shape=tuple(header['shape']);n=int(np.prod(shape));assert shape==(3072,768)
        bits=np.unpackbits(a['modes'],bitorder='little').reshape(-1,2)[:n];mode=bits@np.array([1,2],np.uint8);free=np.flatnonzero(mode==0);nf=len(free)
        b0=(-4+a['codes'].astype(float)*8/255).astype(np.float32)
        lower=np.repeat(a['lower'].astype(float),128,axis=0);span=np.repeat(a['span'].astype(float),128,axis=0);bias=a['bias'].copy()
    q0=(mode==2).astype(float);q0[free]=1/(1+np.exp(-2*b0.astype(float)));nominal=x@(lower+span*q0.reshape(shape))+bias
    np.testing.assert_allclose(np.mean((nominal-ref)**2),report['nominal_control_bias_floor'],rtol=1e-10)
    cal_table=table(OUT/'calibration_metrics.csv');query_table=table(OUT/'query_metrics.csv')
    assert len(cal_table)==90 and len(query_table)==150
    assert {(int(r['seed']),int(r['sigma_index']),int(r['start_segment'])) for r in cal_table}=={(s,si,t) for s in range(10) for si in range(3) for t in [0,64,256]}
    expected_queries={(s,si,age,policy) for s in range(10) for si in range(3) for age in [0,8,32] for policy in (['retain'] if age==0 else ['retain','refresh'])}
    assert {(int(r['seed']),int(r['sigma_index']),int(r['age']),r['policy']) for r in query_table}==expected_queries
    assert len(list((OUT/'calibrations').glob('*.json')))==len(list((OUT/'calibrations').glob('*.npz')))==90
    assert len(list((OUT/'queries').glob('*.json')))==len(list((OUT/'queries').glob('*.npz')))==150
    probes={code:float(np.float32(-4+code*8/255)) for code in [96,159]}
    total_fields=0
    for seed in range(10):
        initial=.2*np.random.default_rng(2026102700+seed).standard_normal((4,nf))[0]
        z=np.random.default_rng(2026102900+seed).standard_normal((4,nf))[0];gain=np.exp(.2*z-.02)
        calibrations=[];queries=[]
        for si,sigma in enumerate([0.,.01,.04]):
            for start in [0,64,256]:
                prefix=OUT/'calibrations'/f'seed{seed}_sigma{si}_start{start}';rec=json.loads(prefix.with_suffix('.json').read_text())
                row=next(r for r in cal_table if int(r['seed'])==seed and int(r['sigma_index'])==si and int(r['start_segment'])==start)
                for key,value in rec.items():
                    if key!='segment_fields':assert str(value)==row[key]
                assert sha(prefix.with_suffix('.npz'))==rec['archive_sha256'] and rec['gain_sha256']==raw_sha(gain)
                a=np.load(prefix.with_suffix('.npz'));assert set(a.files)=={'count_96','count_159','codes'}
                u={}
                for code in [96,159]:
                    c=a[f'count_{code}'];assert c.dtype==np.int32 and c.shape==(nf,) and np.all((c>=0)&(c<=2048))
                    u[code]=.5*(np.log(c.astype(float)+.5)-np.log(2048-c.astype(float)+.5))
                raw=(u[159]-u[96])/(probes[159]-probes[96]);ah=np.minimum(4.,np.maximum(.25,raw));dh=(u[159]+u[96])/2-raw*(probes[159]+probes[96])/2
                requested=(b0.astype(float)-dh)/ah;codes=np.rint((np.clip(requested,-4,4)+4)*255/8).astype(np.uint8)
                np.testing.assert_array_equal(codes,a['codes']);assert rec['command_sha256']==raw_sha(codes)
                np.testing.assert_allclose(rec['estimated_gain_rmse'],np.sqrt(np.mean((ah-gain)**2)),rtol=1e-12)
                assert rec['gain_guard_low']==np.count_nonzero(raw<.25) and rec['gain_guard_high']==np.count_nonzero(raw>4.)
                assert rec['bound_clipped_commands']==np.count_nonzero(np.abs(requested)>4)
                assert rec['native_count_segments']==64 and rec['native_probe_windows']==2 and rec['calibration_free_node_evaluations']==4096*nf
                assert rec['start_segment']==start and rec['end_segment']==start+64 and rec['sigma']==sigma
                assert [(f['segment'],f['code']) for f in rec['segment_fields']]==[(s,96 if s<start+32 else 159) for s in range(start,start+64)]
                calibrations.append({'si':si,'sigma':sigma,'start':start,'rec':rec,'offset_hat':dh,'codes':codes,'path':prefix.with_suffix('.npz')})
            for age in [0,8,32]:
                for policy in (['retain'] if age==0 else ['retain','refresh']):
                    prefix=OUT/'queries'/f'seed{seed}_sigma{si}_age{age}_{policy}';rec=json.loads(prefix.with_suffix('.json').read_text())
                    row=next(r for r in query_table if int(r['seed'])==seed and int(r['sigma_index'])==si and int(r['age'])==age and r['policy']==policy)
                    for key,value in rec.items():
                        if key!='segment_fields':assert str(value)==row[key]
                    assert sha(prefix.with_suffix('.npz'))==rec['archive_sha256']
                    start=0 if policy=='retain' else 8*age;source=next(c for c in calibrations if c['si']==si and c['start']==start)
                    assert rec['calibration_source']==str(source['path']) and rec['calibration_source_sha256']==sha(source['path'])
                    qs=64+age*8;assert rec['query_start_segment']==qs and rec['query_end_segment']==qs+8
                    assert [f['segment'] for f in rec['segment_fields']]==list(range(qs,qs+8))
                    query={'si':si,'sigma':sigma,'qs':qs,'rec':rec,'output_path':prefix.with_suffix('.npz'),
                        'command':(-4+source['codes'].astype(float)*8/255).astype(np.float32).astype(float),
                        'qsum':np.zeros(nf),'vsum':np.zeros(nf)}
                    queries.append(query)
        # Regenerate one vector increment at a time, independently of the driver's
        # large cumsum buffer. Exact path equivalence has a separate unit test.
        rng=np.random.default_rng(2026103000+seed);state=np.zeros(nf)
        for segment in range(329):
            for c in calibrations:
                rec=c['rec'];start=c['start']
                if start<=segment<start+64:
                    entry=rec['segment_fields'][segment-start];code=entry['code']
                    delta=initial+(c['sigma']/np.sqrt(8))*state
                    physical=(gain*probes[code]+delta).astype(np.float32)
                    assert entry['sha256']==raw_sha(physical);total_fields+=1
                if segment==start+64:
                    delta=initial+(c['sigma']/np.sqrt(8))*state
                    assert rec['end_offset_sha256']==raw_sha(delta)
                    np.testing.assert_allclose(rec['estimated_offset_rmse_at_end'],np.sqrt(np.mean((c['offset_hat']-delta)**2)),rtol=1e-12)
            for q in queries:
                qs=q['qs'];rec=q['rec']
                if segment==qs or segment==qs+8:
                    delta=initial+(q['sigma']/np.sqrt(8))*state
                    assert rec['start_offset_sha256' if segment==qs else 'end_offset_sha256']==raw_sha(delta)
                if qs<=segment<qs+8:
                    delta=initial+(q['sigma']/np.sqrt(8))*state;physical=(gain*q['command']+delta).astype(np.float32)
                    assert rec['segment_fields'][segment-qs]['sha256']==raw_sha(physical);total_fields+=1
                    prob=1/(1+np.exp(-2*physical.astype(float)));q['qsum']+=prob;q['vsum']+=prob-prob*prob
            if segment<328:state+=rng.standard_normal(nf)
        for q in queries:
            rec=q['rec'];avg=(mode==2).astype(float);avg[free]=q['qsum']/8;vv=np.zeros(n);vv[free]=q['vsum']/8
            mean=x@(lower+span*avg.reshape(shape))+bias;me=float(np.mean((mean-ref)**2))
            var=float(np.mean(np.mean(x*x,axis=(0,1))@(span*span*vv.reshape(shape)))/512)
            yy=np.load(q['output_path'])['output'].astype(float);assert yy.shape==ref.shape and np.isfinite(yy).all()
            for key,value in [('conditional_mean_error_mse',me),('predicted_sampling_mse',var),('predicted_total_mse',me+var),
                ('observed_total_mse',float(np.mean((yy-ref)**2))),('observed_sampling_mse',float(np.mean((yy-mean)**2))),
                ('error_vs_nominal_programmed_mean',float(np.mean((mean-nominal)**2)))]:
                np.testing.assert_allclose(rec[key],value,rtol=4e-7,atol=1e-12)
            events=1+(rec['policy']=='refresh');assert rec['native_linear_segments']==8 and rec['inference_free_node_evaluations']==512*nf
            assert rec['counterfactual_calibration_events']==events and rec['counterfactual_calibration_free_evaluations']==events*4096*nf
            assert rec['counterfactual_active_sweeps']==events*4096+512
            assert rec['counterfactual_idle_sweeps']==(q['qs']-events*64)*64>=0
            assert rec['query_end_clock_sweeps']==(q['qs']+8)*64==rec['counterfactual_active_sweeps']+rec['counterfactual_idle_sweeps']
        zero=[q['rec']['conditional_mean_error_mse'] for q in queries if q['si']==0 and q['rec']['policy']=='retain']
        np.testing.assert_allclose(zero,zero[0],rtol=1e-12)
        print(f'Independent streamed drift-path/count/query audit seed{seed}',flush=True)
    assert total_fields==6960
    assert report['native_query_windows']==150 and report['native_linear_segments']==1200
    assert report['native_calibration_events']==90 and report['native_probe_windows']==180 and report['native_count_segments']==5760
    assert report['executed_calibration_free_node_evaluations']==90*4096*nf
    assert report['executed_query_free_node_evaluations']==150*512*nf
    audit={'passed':True,'native_query_windows':150,'native_calibration_events':90,'native_probe_windows':180,
        'independently_reconstructed_segment_field_hashes':total_fields,'streaming_Brownian_path_reconstruction':True,
        'clock_and_calibration_idle_costs_verified':True,'source_sha256':sha(__file__),
        'scope':'Independent streaming drift path, every active field/hash, counts-to-code fit, query time-averaged means/variance/errors and counterfactual clock/cost identity. Segmented native trajectories separately checked on CPU/CUDA. Synthetic drift, not hardware seconds, throughput or full-model quality.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
