"""Independent full-test score, calibration and field audit for direct-p=w networks."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha

OUT=Path('results/thrml_spw/calibrated_mnist');DATA=Path('data/thrml_spw/positive_mnist')
CONDITIONS=['ideal8','uncalibrated','cal1024','cal4096','oracle']


def raw_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def table(p):
    with p.open() as f:return list(csv.DictReader(f))


def independently_score(logits,labels,reference,row):
    values=logits.astype(float);mx=values.max(1)
    ce=float(np.mean(mx+np.log(np.exp(values-mx[:,None]).sum(1))-values[np.arange(len(labels)),labels]))
    accuracy=float(np.mean(values.argmax(1)==labels));rmse=float(np.sqrt(np.mean((values-reference)**2)))
    np.testing.assert_allclose([float(row['accuracy']),float(row['cross_entropy']),float(row['logit_rmse'])],[accuracy,ce,rmse],rtol=1e-11,atol=1e-12)


def main():
    p=json.loads((OUT/'protocol.json').read_text());report=json.loads((OUT/'REPORT.json').read_text());m=json.loads((OUT/'manifest.json').read_text())
    assert m['config']==p and m['result']==report and m['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    for path,h in p['source_sha256'].items():assert sha(path)==h
    assert p['model_seeds']==list(range(5)) and p['device_sampling_seeds']==list(range(10)) and p['samples_per_layer']==[64,256,1024]
    assert p['conditions']==CONDITIONS and p['calibration_samples_per_free_device']==[1024,4096] and p['batch_size']==32
    assert sha(DATA/'evaluation_data.npz')==p['input_sha256']
    tests=json.loads((OUT/'trajectory_test_record.json').read_text());assert tests['passed']
    assert sha('research/thrml_spw/test_controlled_positive_native.py')==tests['test_source_sha256'] and sha('research/thrml_spw/controlled_positive_native.py')==tests['native_source_sha256']
    for dev in ['cpu','gpu']:assert 'OK' in (OUT/f'{dev}_trajectory_test.log').read_text()
    data=np.load(DATA/'evaluation_data.npz');x=data['test_inputs'];y=data['test_labels'];assert x.shape==(10000,64)
    identity=np.load(OUT/'evaluation_identity.npz');np.testing.assert_array_equal(identity['example_indices'],np.arange(10000));np.testing.assert_array_equal(identity['labels'],y)
    baseline=table(OUT/'baseline_metrics.csv');meanrows=table(OUT/'mean_network_metrics.csv');rows=table(OUT/'native_metrics.csv')
    assert len(baseline)==20 and len(meanrows)==250 and len(rows)==750
    assert {(int(r['model']),int(r['device']),r['condition'],int(r['T'])) for r in rows}=={(model,device,c,t) for model in range(5) for device in range(10) for c in CONDITIONS for t in [64,256,1024]}
    assert len(list((OUT/'runs').glob('*.json')))==len(list((OUT/'runs').glob('*.npz')))==750
    assert len(list((OUT/'devices').glob('*.json')))==50 and len(list((OUT/'devices').glob('*_means.npz')))==50
    total_cal=total_infer=0
    for model in range(5):
        export=DATA/f'seed{model}.npz';assert sha(export)==p['model_exports_sha256'][str(model)]
        a=np.load(export);w=[a[f'weight_{i}'] for i in range(2)];bias=[a[f'bias_{i}'] for i in range(2)]
        reference=np.maximum(x@w[0]+bias[0],0)@w[1]+bias[1]
        np.testing.assert_allclose(reference,a['reference_test_logits'],atol=1e-4)
        flat=np.concatenate([v.reshape(-1) for v in w]).astype(float);free=np.flatnonzero((flat>0)&(flat<1));nf=len(free)
        original=((np.log(flat[free])-np.log1p(-flat[free]))/2).astype(np.float32)
        codes0=np.rint((np.clip(original.astype(float),-4,4)+4)*255/8).astype(np.uint8);b0=(-4+codes0.astype(float)*8/255).astype(np.float32)
        ctrl=report['model_controls'][str(model)];assert ctrl['logical_weight_nodes']==2368 and ctrl['free_nodes']==nf and ctrl['clamped_nodes']==2368-nf
        assert ctrl['free_by_layer']==[int(np.count_nonzero((v>0)&(v<1))) for v in w]
        assert ctrl['one_method_command_bytes']==nf and ctrl['actual_nominal_float32_field_bytes']==4*nf and ctrl['affine_range_metadata_values']==0
        assert ctrl['nominal_range_clipped_fields']==np.count_nonzero(np.abs(original)>4)
        def expectation(fields):
            q=flat.copy();q[free]=1/(1+np.exp(-2*fields.astype(float)))
            q0=q[:w[0].size].reshape(w[0].shape);q1=q[w[0].size:].reshape(w[1].shape)
            return np.maximum(x@q0+bias[0],0)@q1+bias[1]
        saved_base=np.load(OUT/'baselines'/f'model{model}.npz')
        independent_reference=reference
        np.testing.assert_allclose(saved_base['FP32'],independent_reference,rtol=1e-10,atol=2e-5)
        # RMSE and paired deltas are defined relative to the stored FP32 result,
        # whose model replay was checked above; do not replace their denominator
        # with a potentially differently rounded BLAS result from another env.
        reference=saved_base['FP32']
        expected={'FP32':independent_reference,'programmed_mean_ideal8':expectation(b0)}
        for bits in [4,8]:
            qw=[]
            for v in w:
                scale=float(v.max())/((1<<bits)-1);qw.append((np.rint(v/scale).clip(0,(1<<bits)-1)*scale).astype(np.float32) if scale else v.copy())
            expected[f'UINT{bits}_per_layer_scale']=np.maximum(x@qw[0]+bias[0],0)@qw[1]+bias[1]
        assert set(saved_base.files)==set(expected)
        for condition,logits in expected.items():
            np.testing.assert_allclose(saved_base[condition],logits,rtol=1e-10,atol=2e-5)
            r=next(r for r in baseline if int(r['model'])==model and r['condition']==condition)
            independently_score(saved_base[condition],y,reference,r)
        for device in range(10):
            prefix=OUT/'devices'/f'model{model}_device{device}';meta=json.loads(prefix.with_suffix('.json').read_text())
            assert sha(prefix.with_suffix('.npz'))==meta['archive_sha256'];bundle=np.load(prefix.with_suffix('.npz'))
            noise=np.random.default_rng(2026110100+100*model+device).standard_normal((2,nf));delta=.2*noise[0];gain=np.exp(.2*noise[1]-.02)
            assert meta['gain_sha256']==raw_sha(gain) and meta['offset_sha256']==raw_sha(delta)
            np.testing.assert_array_equal(bundle['gain'],gain);np.testing.assert_array_equal(bundle['offset'],delta);np.testing.assert_array_equal(bundle['nominal_codes'],codes0)
            commands={'ideal8':codes0,'uncalibrated':codes0}
            lo,hi=[float(np.float32(-4+c*8/255)) for c in [96,159]]
            for budget in [1024,4096]:
                each=budget//2;u={}
                for code in [96,159]:
                    counts=bundle[f'count_P{budget}_code{code}'];assert counts.dtype==np.int32 and counts.shape==(nf,) and np.all((counts>=0)&(counts<=each))
                    u[code]=.5*(np.log(counts.astype(float)+.5)-np.log(each-counts.astype(float)+.5))
                raw=(u[159]-u[96])/(hi-lo);ah=np.minimum(4.,np.maximum(.25,raw));dh=(u[159]+u[96])/2-raw*(hi+lo)/2
                requested=(b0.astype(float)-dh)/ah;code=np.rint((np.clip(requested,-4,4)+4)*255/8).astype(np.uint8)
                np.testing.assert_array_equal(bundle[f'codes_cal{budget}'],code);commands[f'cal{budget}']=code
                c=meta['calibration'][str(budget)]
                np.testing.assert_allclose(c['gain_rmse'],np.sqrt(np.mean((ah-gain)**2)),rtol=1e-12);np.testing.assert_allclose(c['offset_rmse'],np.sqrt(np.mean((dh-delta)**2)),rtol=1e-12)
                assert c['gain_guard_low']==np.count_nonzero(raw<.25) and c['gain_guard_high']==np.count_nonzero(raw>4.) and c['bound_clipped_commands']==np.count_nonzero(np.abs(requested)>4)
                assert c['native_probe_runs']==2 and c['free_evaluations']==budget*nf
            oracle=np.rint((np.clip((b0.astype(float)-delta)/gain,-4,4)+4)*255/8).astype(np.uint8)
            np.testing.assert_array_equal(bundle['codes_oracle'],oracle);commands['oracle']=oracle
            assert meta['native_probe_runs']==4 and meta['calibration_free_evaluations']==5120*nf;total_cal+=5120*nf
            means=np.load(OUT/'devices'/f'model{model}_device{device}_means.npz');assert set(means.files)==set(CONDITIONS)
            for condition in CONDITIONS:
                f=b0.copy() if condition=='ideal8' else (gain*(-4+commands[condition].astype(float)*8/255).astype(np.float32).astype(float)+delta).astype(np.float32)
                np.testing.assert_array_equal(bundle[f'fields_{condition}'],f);assert meta['field_hashes'][condition]==raw_sha(f)
                predicted=expectation(f);np.testing.assert_allclose(means[condition],predicted,rtol=1e-11,atol=2e-8)
                mr=next(r for r in meanrows if int(r['model'])==model and int(r['device'])==device and r['condition']==condition)
                independently_score(means[condition],y,reference,mr)
                for t in [64,256,1024]:
                    name=f'model{model}_device{device}_{condition}_T{t}';record=json.loads((OUT/'runs'/f'{name}.json').read_text())
                    row=next(r for r in rows if int(r['model'])==model and int(r['device'])==device and r['condition']==condition and int(r['T'])==t)
                    for key,value in record.items():assert str(value)==row[key]
                    path=OUT/'runs'/f'{name}.npz';assert sha(path)==record['archive_sha256'] and record['field_sha256']==raw_sha(f) and record['device_archive_sha256']==sha(prefix.with_suffix('.npz'))
                    result=np.load(path);pre,logits=result['preactivation0'],result['logits']
                    assert pre.shape==(10000,32) and logits.shape==(10000,10) and pre.dtype==logits.dtype==np.float32 and np.isfinite(pre).all() and np.isfinite(logits).all()
                    independently_score(logits,y,reference,row)
                    np.testing.assert_allclose(float(row['accuracy_difference_vs_FP32']),float(row['accuracy'])-float(np.mean(reference.argmax(1)==y)),atol=1e-15)
                    assert int(row['free_node_evaluations'])==313*t*nf and int(row['logical_weight_readouts'])==313*t*2368
                    assert int(row['nominal_binary_products_excluding_padding'])==10000*t*2368 and int(row['nominal_binary_products_including_padding'])==10016*t*2368
                    total_infer+=313*t*nf
            print(f'Independent full-network raw/calibration/score audit model{model} device{device}',flush=True)
    assert report['native_full_test_runs']==750 and report['native_calibration_probe_runs']==200
    assert report['executed_calibration_free_evaluations']==total_cal and report['executed_inference_free_evaluations']==total_infer
    replay=json.loads((OUT/'public_path_audit.json').read_text());assert replay['passed'] and replay['native_runs_audited']==750
    audit={'passed':True,'native_full_test_runs':750,'trained_models':5,'joint_device_sampling_realizations_per_model':10,
        'test_examples_per_run':10000,'native_calibration_probe_runs':200,'independent_public_path_runs':750,
        'source_sha256':sha(__file__),'scope':'Independent data/model/field/count/code/baseline lineage, every raw score and resource count, plus separate public-factor layer path replay. Reused standard test set, static synthetic device errors; no clocked drift or physical-system claim.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
