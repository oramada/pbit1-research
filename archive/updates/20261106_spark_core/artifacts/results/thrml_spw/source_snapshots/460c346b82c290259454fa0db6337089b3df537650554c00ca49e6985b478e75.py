"""Independent controls, endpoint modes, full scores and active-count audit."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha
from .verify_calibrated_mnist import independently_score

OUT=Path('results/thrml_spw/finite_endpoint_mnist');DATA=Path('data/thrml_spw/positive_mnist')


def raw_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def main():
    p=json.loads((OUT/'protocol.json').read_text());result=json.loads((OUT/'REPORT.json').read_text());m=json.loads((OUT/'manifest.json').read_text())
    assert m['config']==p and m['result']==result and m['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for path,h in p['source_sha256'].items():assert sha(path)==h
    expected_enc=[('clamped_B4_b8',4.,8,True),('free_B4_b8',4.,8,False),('free_B8_b8',8.,8,False),('free_B8_b10',8.,10,False)]
    assert [(e['name'],e['bound'],e['bits'],e['clamp_endpoints']) for e in p['encodings']]==expected_enc
    assert p['model_seeds']==list(range(5)) and p['device_sampling_seeds']==list(range(10)) and p['samples_per_layer']==[256,1024]
    assert p['conditions']==['ideal','uncalibrated','cal4096','oracle'] and p['calibration_samples_per_free_device']==4096
    tests=json.loads((OUT/'trajectory_test_record.json').read_text());assert tests['passed']
    assert tests['test_source_sha256']==sha('research/thrml_spw/test_finite_endpoint_control.py') and tests['control_source_sha256']==sha('research/thrml_spw/finite_endpoint_control.py')
    for dev in ['cpu','gpu']:assert 'OK' in (OUT/f'{dev}_trajectory_test.log').read_text()
    assert sha(DATA/'evaluation_data.npz')==p['input_sha256'];d=np.load(DATA/'evaluation_data.npz');x,y=d['test_inputs'],d['test_labels']
    ids=np.load(OUT/'evaluation_identity.npz');np.testing.assert_array_equal(ids['labels'],y);np.testing.assert_array_equal(ids['example_indices'],np.arange(10000))
    base=read(OUT/'baseline_metrics.csv');means=read(OUT/'mean_network_metrics.csv');rows=read(OUT/'native_metrics.csv')
    assert len(base)==15 and len(means)==800 and len(rows)==1600
    assert {(int(r['model']),int(r['device']),r['encoding'],r['condition'],int(r['T'])) for r in rows}=={(m,h,e,c,t) for m in range(5) for h in range(10) for e,_,_,_ in expected_enc for c in p['conditions'] for t in [256,1024]}
    assert len(list((OUT/'runs').glob('*.json')))==len(list((OUT/'runs').glob('*.npz')))==1600
    assert len(list((OUT/'devices').glob('*.json')))==len(list((OUT/'devices').glob('*_means.npz')))==200
    total_cal=total_native=0
    for model in range(5):
        path=DATA/f'seed{model}.npz';assert sha(path)==p['model_exports_sha256'][str(model)]
        ex=np.load(path);w=[ex[f'weight_{i}'] for i in range(2)];bias=[ex[f'bias_{i}'] for i in range(2)]
        original=np.concatenate([a.reshape(-1) for a in w]).astype(float)
        savedbase=np.load(OUT/'baselines'/f'model{model}.npz');reference=savedbase['FP32']
        recomputed=np.maximum(x@w[0]+bias[0],0)@w[1]+bias[1];np.testing.assert_allclose(reference,recomputed,atol=2e-5)
        np.testing.assert_allclose(reference,ex['reference_test_logits'],atol=1e-4)
        expectedbase={'FP32':recomputed}
        for bits in [4,8]:
            qw=[]
            for a in w:
                scale=float(a.max())/((1<<bits)-1);qw.append((np.rint(a/scale).clip(0,(1<<bits)-1)*scale).astype(np.float32) if scale else a.copy())
            expectedbase[f'UINT{bits}']=np.maximum(x@qw[0]+bias[0],0)@qw[1]+bias[1]
        assert set(savedbase.files)==set(expectedbase)
        for c,a in expectedbase.items():
            np.testing.assert_allclose(savedbase[c],a,atol=2e-5)
            row=next(r for r in base if int(r['model'])==model and r['condition']==c);independently_score(savedbase[c],y,reference,row)
        for en,bound,bits,clamp in expected_enc:
            epsilon=1/(1+np.exp(2*bound));encoded=original.copy() if clamp else np.minimum(1-epsilon,np.maximum(epsilon,original))
            active=np.flatnonzero((encoded>0)&(encoded<1));nf=len(active);levels=(1<<bits)-1;dtype=np.uint8 if bits<=8 else np.uint16
            if not clamp:assert nf==2368
            b=((np.log(encoded[active])-np.log1p(-encoded[active]))/2).astype(np.float32)
            codes0=np.rint((np.clip(b.astype(float),-bound,bound)+bound)*levels/(2*bound)).astype(dtype)
            b0=(-bound+codes0.astype(float)*(2*bound/levels)).astype(np.float32)
            probes=[int(np.rint((v+bound)*levels/(2*bound))) for v in [-1.,1.]]
            pf=[float(np.float32(-bound+c*(2*bound/levels))) for c in probes]
            ctrl=result['model_controls'][str(model)][en]
            assert ctrl['logical_nodes']==2368 and ctrl['free_nodes']==nf and ctrl['clamped_nodes']==2368-nf
            assert ctrl['logical_command_payload_bits']==nf*bits and ctrl['actual_command_array_bytes']==nf*np.dtype(dtype).itemsize
            assert ctrl['endpoint_mode_bits_if_packed']==(4736 if clamp else 0) and ctrl['probe_codes']==probes and ctrl['probe_fields']==pf
            assert ctrl['actual_base_field_bytes']==4*nf and ctrl['neural_bias_values']==42
            def mean_network(fields):
                q=encoded.copy();q[active]=1/(1+np.exp(-2*fields.astype(float)))
                return np.maximum(x@q[:2048].reshape(64,32)+bias[0],0)@q[2048:].reshape(32,10)+bias[1]
            for device in range(10):
                prefix=OUT/'devices'/f'model{model}_device{device}_{en}';meta=json.loads(prefix.with_suffix('.json').read_text());assert sha(prefix.with_suffix('.npz'))==meta['archive_sha256']
                a=np.load(prefix.with_suffix('.npz'));noise=np.random.default_rng(2026110200+100*model+device).standard_normal((2,2368))
                gain=np.exp(.2*noise[1]-.02)[active];delta=(.2*noise[0])[active]
                np.testing.assert_array_equal(a['active_indices'],active);np.testing.assert_array_equal(a['gain'],gain);np.testing.assert_array_equal(a['offset'],delta);np.testing.assert_array_equal(a['nominal_codes'],codes0)
                assert meta['gain_sha256']==raw_sha(gain) and meta['offset_sha256']==raw_sha(delta)
                u=[]
                for pi in [0,1]:
                    counts=a[f'count_{pi}'];assert counts.dtype==np.int32 and counts.shape==(nf,) and np.all((counts>=0)&(counts<=2048))
                    u.append(.5*(np.log(counts.astype(float)+.5)-np.log(2048-counts.astype(float)+.5)))
                raw=(u[1]-u[0])/(pf[1]-pf[0]);ah=np.minimum(4.,np.maximum(.25,raw));dh=(u[1]+u[0])/2-raw*(pf[1]+pf[0])/2
                requested=(b0.astype(float)-dh)/ah;cal=np.rint((np.clip(requested,-bound,bound)+bound)*levels/(2*bound)).astype(dtype)
                oracle=np.rint((np.clip((b0.astype(float)-delta)/gain,-bound,bound)+bound)*levels/(2*bound)).astype(dtype)
                assert a['codes_cal4096'].dtype==a['codes_oracle'].dtype==dtype
                np.testing.assert_array_equal(a['codes_cal4096'],cal);np.testing.assert_array_equal(a['codes_oracle'],oracle)
                np.testing.assert_allclose(meta['gain_rmse'],np.sqrt(np.mean((ah-gain)**2)),rtol=1e-12);np.testing.assert_allclose(meta['offset_rmse'],np.sqrt(np.mean((dh-delta)**2)),rtol=1e-12)
                assert meta['gain_guard_low']==np.count_nonzero(raw<.25) and meta['gain_guard_high']==np.count_nonzero(raw>4)
                assert meta['bound_clipped_corrected_commands']==np.count_nonzero(np.abs(requested)>bound)
                assert meta['native_probe_runs']==2 and meta['calibration_free_evaluations']==4096*nf;total_cal+=4096*nf
                allmeans=np.load(OUT/'devices'/f'model{model}_device{device}_{en}_means.npz');assert set(allmeans.files)==set(p['conditions'])
                for condition in p['conditions']:
                    cmd=codes0 if condition in ['ideal','uncalibrated'] else cal if condition=='cal4096' else oracle
                    fields=b0.copy() if condition=='ideal' else (gain*(-bound+cmd.astype(float)*(2*bound/levels)).astype(np.float32).astype(float)+delta).astype(np.float32)
                    np.testing.assert_array_equal(a[f'fields_{condition}'],fields);assert meta['field_hashes'][condition]==raw_sha(fields)
                    expected=mean_network(fields);np.testing.assert_allclose(allmeans[condition],expected,atol=2e-8,rtol=1e-11)
                    mr=next(r for r in means if int(r['model'])==model and int(r['device'])==device and r['encoding']==en and r['condition']==condition)
                    independently_score(allmeans[condition],y,reference,mr)
                    for t in [256,1024]:
                        name=f'model{model}_device{device}_{en}_{condition}_T{t}';rec=json.loads((OUT/'runs'/f'{name}.json').read_text());rr=next(r for r in rows if int(r['model'])==model and int(r['device'])==device and r['encoding']==en and r['condition']==condition and int(r['T'])==t)
                        for key,value in rec.items():assert str(value)==rr[key]
                        npz=OUT/'runs'/f'{name}.npz';assert sha(npz)==rec['archive_sha256'] and rec['device_archive_sha256']==sha(prefix.with_suffix('.npz')) and rec['field_sha256']==raw_sha(fields)
                        outputs=np.load(npz);pre,logits=outputs['preactivation0'],outputs['logits'];assert pre.shape==(10000,32) and logits.shape==(10000,10) and pre.dtype==logits.dtype==np.float32 and np.isfinite(pre).all() and np.isfinite(logits).all()
                        independently_score(logits,y,reference,rr)
                        np.testing.assert_allclose(float(rr['accuracy_difference_vs_FP32']),float(rr['accuracy'])-float(np.mean(reference.argmax(1)==y)),atol=1e-15)
                        assert int(rr['free_node_evaluations'])==313*t*nf and int(rr['logical_weight_readouts'])==313*t*2368
                        assert int(rr['binary_products_with_padding'])==10016*t*2368 and int(rr['binary_products_excluding_padding'])==10000*t*2368;total_native+=313*t*nf
            print(f'Independent endpoint/field/control/full-score audit model{model} {en}',flush=True)
    assert result['native_full_test_runs']==1600 and result['native_probe_runs']==400
    assert result['executed_calibration_free_evaluations']==total_cal and result['executed_inference_free_evaluations']==total_native
    replay=json.loads((OUT/'public_path_audit.json').read_text());assert replay['passed'] and replay['native_runs_audited']==1600 and replay['examples_per_run']==10000
    audit={'passed':True,'native_full_test_runs':1600,'native_probe_runs':400,'examples_per_run':10000,
        'all_free_variants_have_zero_clamp_nodes':True,'physical_device_coordinate_pairing_verified':True,
        'all_fields_codes_means_scores_and_resource_counts_recomputed':True,'source_sha256':sha(__file__),
        'scope':'Independent finite-control/endpoint transformation, full-coordinate device pairing, calibration counts-to-codes and all full-test score/counts; every example additionally replayed through public THRML frequencies. Synthetic static conditions, no physical availability or superiority claim.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
