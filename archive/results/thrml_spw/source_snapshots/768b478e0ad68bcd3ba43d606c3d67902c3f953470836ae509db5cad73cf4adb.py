"""Independent saved-count fit, clock/resource and complete task-score audit."""
import csv,json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha
from .verify_calibrated_mnist import independently_score
OUT=Path('results/thrml_spw/clocked_mnist');DATA=Path('data/thrml_spw/positive_mnist')


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    p=json.loads((OUT/'protocol.json').read_text());result=json.loads((OUT/'REPORT.json').read_text());m=json.loads((OUT/'manifest.json').read_text())
    assert m['config']==p and m['result']==result and m['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for name,h in p['source_sha256'].items():assert sha(name)==h
    assert p['model_seeds']==p['device_path_sampling_seeds']==list(range(5));assert p['drift_SD_per_1024_sweeps']==[0.,.01,.04]
    assert p['policies']==['retain_initial','refresh_every32_batches'] and p['samples_per_layer']==1024 and p['segment_sweeps']==64
    assert p['encoding']=={'name':'free_B4_b8','bound':4.,'bits':8,'clamp_endpoints':False}
    test=json.loads((OUT/'trajectory_test_record.json').read_text());assert test['passed'] and test['source_sha256']==sha('research/thrml_spw/test_clocked_network_native.py')
    for backend in ['cpu','gpu']:assert 'Ran 2 tests' in (OUT/f'{backend}_trajectory_test.log').read_text() and 'OK' in (OUT/f'{backend}_trajectory_test.log').read_text()
    assert sha(DATA/'evaluation_data.npz')==p['input_sha256'];d=np.load(DATA/'evaluation_data.npz');ids=np.random.default_rng(2026110301).permutation(10000);x,y=d['test_inputs'][ids],d['test_labels'][ids]
    ident=np.load(OUT/'evaluation_identity.npz');np.testing.assert_array_equal(ident['labels'],y);np.testing.assert_array_equal(ident['example_indices'],ids)
    native=rows(OUT/'native_metrics.csv');windows=rows(OUT/'window_metrics.csv');baseline=rows(OUT/'baseline_metrics.csv')
    assert len(native)==150 and len(windows)==1500 and len(baseline)==5
    expected={(m,d,r,c) for m in range(5) for d in range(5) for r in [0.,.01,.04] for c in p['policies']}
    assert {(int(r['model']),int(r['device']),float(r['rate']),r['policy']) for r in native}==expected
    assert len(list((OUT/'runs').glob('*.npz')))==len(list((OUT/'runs').glob('*.json')))==150
    assert len(list((OUT/'calibration').glob('*.npz')))==len(list((OUT/'calibration').glob('*.json')))==75
    # Partition: calibration followed by32(serial two-layer) batches, last25.
    occupied=[]
    for k in range(10):occupied.extend(range(1088*k,1088*k+64))
    for b in range(313):occupied.extend(range(32*b+64*(b//32+1),32*b+64*(b//32+1)+32))
    assert sorted(occupied)==list(range(10656)) and 10656*64==681984
    totalcal=totalinfer=0;guardlow=guardhigh=0
    for model in range(5):
        assert sha(DATA/f'seed{model}.npz')==p['model_sha256'][str(model)]
        e=np.load(DATA/f'seed{model}.npz');ww=[e[f'weight_{i}'] for i in range(2)];bb=[e[f'bias_{i}'] for i in range(2)]
        reference=np.load(OUT/'baselines'/f'model{model}.npz')['FP32'];computed=np.maximum(x@ww[0]+bb[0],0)@ww[1]+bb[1];np.testing.assert_allclose(reference,computed,atol=2e-5)
        np.testing.assert_allclose(reference,e['reference_test_logits'][ids],atol=1e-4)
        independently_score(reference,y,reference,next(r for r in baseline if int(r['model'])==model))
        flat=np.concatenate([a.reshape(-1) for a in ww]).astype(float);eps=1/(1+np.exp(8));q=np.minimum(1-eps,np.maximum(eps,flat));assert np.all((q>0)&(q<1)) and len(q)==2368
        fields=((np.log(q)-np.log1p(-q))/2).astype(np.float32)
        codes=np.rint((np.clip(fields.astype(float),-4,4)+4)*255/8).astype(np.uint8);nominal=(-4+codes.astype(float)*8/255).astype(np.float32)
        probe=(-4+np.rint((np.array([-1.,1.])+4)*255/8)*8/255).astype(np.float32).astype(float)
        for device in range(5):
            noise=np.random.default_rng(2026110400+100*model+device).standard_normal((2,2368));gain=np.exp(.2*noise[1]-.02);initial=.2*noise[0]
            for rate in [0.,.01,.04]:
                tag=f'model{model}_device{device}_rate{rate:g}';cp=OUT/'calibration'/f'{tag}.npz';a=np.load(cp);cm=json.loads(cp.with_suffix('.json').read_text());assert sha(cp)==cm['archive_sha256']
                np.testing.assert_array_equal(a['gain'],gain);np.testing.assert_array_equal(a['initial_offset'],initial);np.testing.assert_array_equal(a['nominal_codes'],codes)
                cc=a['segment_counts'];assert cc.dtype==np.uint8 and cc.shape==(10,64,2368) and np.all(cc<=64)
                low,high=cc[:,:32].sum(1,dtype=np.int32),cc[:,32:].sum(1,dtype=np.int32)
                u=[.5*(np.log(c.astype(float)+.5)-np.log(2048-c.astype(float)+.5)) for c in [low,high]]
                raw=(u[1]-u[0])/(probe[1]-probe[0]);gh=np.clip(raw,.25,4);dh=(u[1]+u[0])/2-raw*(probe[1]+probe[0])/2
                np.testing.assert_allclose(a['gain_estimates'],gh,atol=2e-14,rtol=1e-13);np.testing.assert_allclose(a['offset_estimates'],dh,atol=2e-14,rtol=1e-13)
                corrected=np.rint((np.clip((nominal.astype(float)-dh)/gh,-4,4)+4)*255/8).astype(np.uint8);np.testing.assert_array_equal(a['corrected_codes'],corrected)
                assert corrected.dtype==np.uint8 and corrected.shape==(10,2368)
                for w in range(10):
                    nl,nh=int(np.count_nonzero(raw[w]<.25)),int(np.count_nonzero(raw[w]>4));assert cm['gain_guards'][w]=={'low':nl,'high':nh};guardlow+=nl;guardhigh+=nh
                assert cm['calibration_events']==10 and cm['probe_windows']==20 and cm['native_segment_calls']==640
                assert cm['free_node_evaluations']==10*4096*2368;totalcal+=cm['free_node_evaluations']
                for policy in p['policies']:
                    name=f'{tag}_{policy}';rn=OUT/'runs'/f'{name}.npz';rec=json.loads(rn.with_suffix('.json').read_text());assert sha(rn)==rec['archive_sha256'] and sha(cp)==rec['calibration_archive_sha256']
                    row=next(r for r in native if int(r['model'])==model and int(r['device'])==device and float(r['rate'])==rate and r['policy']==policy)
                    for k,v in rec.items():
                        if k not in ['layer_field_hashes','windows']:assert str(v)==row[k]
                    aout=np.load(rn);pre,logits=aout['preactivation0'],aout['logits'];assert pre.shape==(10000,32) and logits.shape==(10000,10) and pre.dtype==logits.dtype==np.float32 and np.isfinite(pre).all() and np.isfinite(logits).all()
                    independently_score(logits,y,reference,row)
                    for w in range(10):
                        low,high=w*1024,min((w+1)*1024,10000);wr=next(r for r in windows if int(r['model'])==model and int(r['device'])==device and float(r['rate'])==rate and r['policy']==policy and int(r['window'])==w)
                        for k,v in rec['windows'][w].items():assert str(v)==wr[k]
                        assert int(wr['examples'])==high-low;independently_score(logits[low:high],y[low:high],reference[low:high],wr)
                    assert rec['inference_free_evaluations']==rec['logical_weight_readouts']==313*1024*2368;totalinfer+=rec['inference_free_evaluations']
                    assert rec['binary_products_with_padding']==10016*1024*2368 and rec['binary_products_without_padding']==10000*1024*2368
                    refresh=policy=='refresh_every32_batches';cal=4096*(10 if refresh else 1);idle=0 if refresh else 36864
                    assert rec['candidate_calibration_free_evaluations']==cal*2368 and rec['active_calibration_sweeps']==cal and rec['idle_sweeps']==idle
                    assert rec['inference_serial_sweeps']==641024 and rec['elapsed_sweep_clock']==641024+cal+idle==681984
            print(f'Independent full-score/calibration-fit/clock ledger model{model} device{device}',flush=True)
    assert totalcal==result['production_calibration_free_evaluations']==7274496000 and totalinfer==result['production_inference_free_evaluations']==113845862400
    assert result['native_calibration_events']==750 and result['native_probe_windows']==1500 and result['calibration_segment_calls']==48000 and result['inference_layer_segment_calls']==1502400
    replay=json.loads((OUT/'public_path_audit.json').read_text());assert replay['passed'] and replay['native_full_test_runs']==150 and replay['calibration_events']==750 and replay['examples_per_run']==10000 and replay['native_field_hashes_regenerated']==1550400
    result={'passed':True,'native_full_test_runs':150,'examples_per_run':10000,'calibration_events':750,'probe_windows':1500,'gain_guard_low':guardlow,'gain_guard_high':guardhigh,
        'all_raw_scores_calibration_fits_codes_clock_and_resource_counts_recomputed':True,'source_sha256':sha(__file__),
        'scope':'Entire all-free finite-endpoint networks, fixed synthetic drift paths and matched serial clocks. Every example and every native count additionally independently public-factor replayed. No physical availability or optimum-refresh claim.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
