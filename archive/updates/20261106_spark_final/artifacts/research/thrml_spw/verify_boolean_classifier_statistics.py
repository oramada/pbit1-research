"""Recompute every metric, baseline and frozen bootstrap contrast independently."""
import json
import numpy as np
from .boolean_classifier_common import OUT,sha,guard
from .fine_range_common import atomic_json

def metric(z,y):
    z=np.asarray(z,float);top=np.max(z,axis=1);loss=top+np.log(np.exp(z-top[:,None]).sum(axis=1))-z[np.arange(y.size),y]
    return np.array([np.count_nonzero(np.argmax(z,axis=1)==y)/y.size,loss.mean()])
def qprob(x):
    e=1/(1+np.exp(8));v=np.clip(np.asarray(x,float),e,1-e);code=np.rint((np.log(v/(1-v))/2+4)*255/8);f=(-4+code*8/255).astype(np.float32).astype(float)
    return 1/(1+np.exp(-2*f))
def main():
    p=json.loads((OUT/'protocol.json').read_text());a=guard(p);s=json.loads((OUT/'statistics.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert s['producer_manifest_sha256']==sha(OUT/'manifest.json');assert s['native_audit_sha256']==sha(OUT/'native_audit.json')
    assert sha(OUT/'statistical_arrays.npz')==s['statistical_arrays_sha256'] and sha(OUT/'baseline_logits.npz')==s['baseline_logits_sha256']
    raw=np.load(OUT/'statistical_arrays.npz');base_outputs=np.load(OUT/'baseline_logits.npz');d=np.load(a['data_path']);x=d['test_inputs'].astype(float);y=d['test_labels'];value=np.empty((2,3,5,10,2));case_rows={r['name']:r for r in s['rows']}
    for r in m['rows']:
        assert sha(r['path'])==r['sha256'];v=metric(np.load(r['path'])['logits'],y);c=case_rows[r['name']];np.testing.assert_allclose(v,[c['accuracy'],c['mean_cross_entropy']],rtol=2e-13,atol=2e-14)
        value[['R1','R3'].index(r['regime']),p['samples'][r['regime']].index(r['T']),r['model'],r['seed']]=v
    np.testing.assert_allclose(value,raw['values'],rtol=2e-13,atol=2e-14)
    names=['FP32_weights_F64_readout','stored_FP32_reference','INT2','INT4','INT8','programmed_mean_R1','programmed_mean_R3'];bm=np.empty((7,5,2));baseline_rows={(r['condition'],r['model']):r for r in s['baseline_rows']}
    for mi in range(5):
        model=np.load(a['models'][mi]['model_path']);control=np.load(a['models'][mi]['controls_path']);w=[model[f'weight_{i}'].astype(float) for i in [0,1]];b=[model[f'bias_{i}'].astype(float) for i in [0,1]];pw=[qprob(v) for v in w]
        for bi,name in enumerate(names):
            if bi==1:expected=model['reference_test_logits']
            elif bi==6:
                low=a['input_lower'];span=a['input_span'];xp=low+span*qprob(np.clip((x-low)/span,0,1));h=np.maximum(xp@pw[0]+b[0],0);lo=float(control['hidden_lower']);sp=float(control['hidden_span']);expected=(lo+sp*qprob(np.clip((h-lo)/sp,0,1)))@pw[1]+b[1]
            else:
                ww=w if bi==0 else pw if bi==5 else [np.round(v*(2**int(name[3:])-1))/(2**int(name[3:])-1) for v in w]
                expected=np.maximum(x@ww[0]+b[0],0)@ww[1]+b[1]
            z=base_outputs[f'c{bi}_model{mi}'];np.testing.assert_allclose(expected,z,rtol=2e-12,atol=2e-11);np.testing.assert_array_equal(expected.argmax(1),z.argmax(1));bm[bi,mi]=metric(z,y);r=baseline_rows[name,mi];np.testing.assert_allclose(bm[bi,mi],[r['accuracy'],r['mean_cross_entropy']],rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(bm,raw['baseline_metrics'],rtol=2e-13,atol=2e-14)
    for bi,name in enumerate(names):np.testing.assert_allclose(bm[bi].mean(0),s['baseline_summary'][name],rtol=2e-13,atol=2e-14)
    rng=np.random.default_rng(1357520);models=rng.integers(0,5,(5000,5));seeds=rng.integers(0,10,(5000,5,10));np.testing.assert_array_equal(models,raw['bootstrap_models']);np.testing.assert_array_equal(seeds,raw['bootstrap_seeds'])
    for r in s['summaries']:
        ri=['R1','R3'].index(r['regime']);ti=p['samples'][r['regime']].index(r['T']);np.testing.assert_allclose(value[ri,ti].mean((0,1)),[r['mean_accuracy'],r['mean_cross_entropy']],rtol=2e-13,atol=2e-14);np.testing.assert_allclose(value[ri,ti].mean(1),r['model_mean_metrics'],rtol=2e-13,atol=2e-14)
    for c in s['contrasts']:
        ti=c['budget_index'];name=c['contrast'];delta=value[1,ti]-value[0,ti] if name=='R3_minus_R1' else value[0 if name=='R1_minus_FP32' else 1,ti]-bm[0,:,None,:]
        # Index/gather separately for each resampled model position.
        distribution=sum(delta[models[:,i,None],seeds[:,i,:]].sum(1) for i in range(5))/50
        np.testing.assert_allclose(distribution,raw[f'{name}_budget{ti}'],rtol=2e-11,atol=2e-13);np.testing.assert_allclose(delta.mean((0,1)),c['difference'],rtol=2e-11,atol=2e-13);np.testing.assert_allclose(np.quantile(distribution,[.025,.975],axis=0).T,c['descriptive_95_interval'],rtol=2e-11,atol=2e-13)
    guard(p);atomic_json(OUT/'statistics_audit.json',{'passed':True,'full_case_metrics_checked':300,'deterministic_model_baselines_checked':35,'bootstrap_values_checked':90000,'all_summaries_and_contrasts_checked':True,'statistics_sha256':sha(OUT/'statistics.json'),'verifier_sha256':sha(__file__),'scope':'Independent formula/count reductions and hierarchical gather validate all reported values; consumed test images remain fixed, no fresh-generalization or physical-system inference.'});print('Independent classifier statistics audit passed',flush=True)
if __name__=='__main__':main()
