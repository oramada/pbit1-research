"""GPU count/replay and actual two-layer validation-only causal preflight."""
import json,os,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .boolean_classifier_common import PREP,sha,fields,encode,keys
from .boolean_classifier_native import CountLayer,readout_boolean,readout_weight
from .verify_boolean_classifier_study import sampler,packed_and_counts,independent_encode
from .fine_range_common import atomic_json,atomic_npz
OUT=Path('results/thrml_spw/boolean_classifier_gpu_preflight')
def main():
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled
    assert os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0';OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    prep=json.loads(PREP.read_text());x=np.load(prep['data_path'])['validation_inputs'][:128].astype(float).reshape(16,8,64);model=np.load(prep['models'][0]['controls_path']);low=prep['input_lower'];span=prep['input_span'];hlow=float(model['hidden_lower']);hspan=float(model['hidden_span'])
    source={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['boolean_classifier_common.py','boolean_classifier_native.py','verify_boolean_classifier_study.py','boolean_classifier_gpu_preflight.py','native.py','controlled_positive_native.py']}
    atomic_json(OUT/'protocol.json',{'sources':source,'preparation_sha256':sha(PREP),'validation_inputs':128,'backend':jax.default_backend(),'samples':{'R1':1568,'R3':1184},'XLA_FLAGS':os.environ.get('XLA_FLAGS'),'scope':'Preflight uses validation inputs only, no labels/task scoring. Full two-layer causal readout, same keyed observer/public states at intended chunk16; no production cases.'})
    updates=0;rows=[]
    for ri,regime in enumerate(['R1','R3']):
        t=1568 if regime=='R1' else 1184;h=x.copy();expected=x.copy();outputs={};start=time.monotonic()
        for layer,(ni,no) in enumerate([(64,32),(32,10)]):
            wf=model[f'weight_field_{layer}'].ravel();bias=model[f'bias_{layer}'];layerbank=CountLayer(ni,no,8,ri==1)
            if ri:
                lowi=low if layer==0 else hlow;spani=span if layer==0 else hspan;ac=encode(np.clip((h-lowi)/spani,0,1));np.testing.assert_array_equal(ac,independent_encode(np.clip((expected-lowi)/spani,0,1)))
                f=np.concatenate([np.broadcast_to(wf,(16,ni*no)),fields(ac).reshape(16,-1)],axis=1)
            else:f=np.broadcast_to(wf,(16,ni*no)).copy()
            k=keys(ri,0,0,t,layer,0,16);result=layerbank.kernel(t)(k,jnp.asarray(f));states=np.asarray(sampler(f.shape[1],t)(k,jnp.asarray(f)));w=states[:,:,:ni*no].reshape(16,t,ni,no)
            if ri:
                aa,cc=map(np.asarray,result);wa=packed_and_counts(w,states[:,:,ni*no:].reshape(16,t,8,ni));wc=w.sum((1,2),dtype=np.uint32)
                np.testing.assert_array_equal(aa,wa);np.testing.assert_array_equal(cc,wc)
                y=readout_boolean(aa,cc[:,None,:],t,lowi,spani,bias);ey=lowi*wc[:,None,:].astype(float)/t+spani*wa.astype(float)/t+bias
                outputs[f'and_{layer}']=aa;outputs[f'column_{layer}']=cc
            else:
                counts=np.asarray(result).reshape(16,ni,no);wc=w.sum(1,dtype=np.uint32);np.testing.assert_array_equal(counts,wc)
                y=readout_weight(counts,t,h,bias);ey=expected@(wc.astype(float)/t)+bias;outputs[f'weight_{layer}']=counts
            np.testing.assert_array_equal(y,ey);h=np.maximum(y,0) if layer==0 else y;expected=np.maximum(ey,0) if layer==0 else ey
            updates+=16*t*f.shape[1]
        outputs['logits']=h;atomic_npz(OUT/(regime+'.npz'),**outputs);rows.append({'regime':regime,'T':t,'elapsed_seconds':time.monotonic()-start,'validation_causal_logits_exact':True});print(json.dumps(rows[-1]),flush=True)
    for s,v in source.items():assert sha(s)==v
    atomic_json(OUT/'evidence_audit.json',{'passed':True,'rows':rows,'observer_updates':updates,'independent_public_updates':updates,'source_sha256':source,'new_test_scores':False,'scope':'Actual two-layer validation128 inputs at maximum prospective budgets and16 chains. Full cohort remains unrun.'})
if __name__=='__main__':main()
