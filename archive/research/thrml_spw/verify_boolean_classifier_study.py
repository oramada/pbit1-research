"""Independent public raw-state and complete causal classifier replay."""
import json,fcntl
from functools import lru_cache
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from thrml import Block,BlockGibbsSpec,SpinNode,FactorSamplingProgram,SamplingSchedule,sample_states
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .boolean_classifier_common import OUT,CHUNK,sha,guard,keys
from .fine_range_common import atomic_json
@lru_cache(None)
def sampler(n,t):
    block=Block([SpinNode() for _ in range(n)]);spec=BlockGibbsSpec([block],[],{SpinNode:jax.ShapeDtypeStruct((),jnp.bool_)})
    def run(key,field):
        program=FactorSamplingProgram(spec,[SpinGibbsConditional()],[SpinEBMFactor([block],field)],[])
        return sample_states(key,program,SamplingSchedule(1,t,1),[jnp.zeros(n,jnp.bool_)],[],[block])[0]
    return jax.jit(jax.vmap(run))
def independent_encode(value):
    e=1/(1+np.exp(8));q=np.clip(np.asarray(value,float),e,1-e)
    return np.rint((np.log(q/(1-q))/2+4)*255/8).astype(np.uint8)
def independent_fields(code):return (-4+np.asarray(code,float)*8/255).astype(np.float32)
def packed_and_counts(w,a):
    # Independent exact bit intersections across time; no observer or float dot products.
    wp=np.packbits(w.transpose(0,2,3,1),axis=-1,bitorder='little');ap=np.packbits(a.transpose(0,3,2,1),axis=-1,bitorder='little')
    lookup=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).sum(1,dtype=np.uint32)
    out=np.zeros((w.shape[0],a.shape[2],w.shape[3]),np.uint32)
    for i in range(w.shape[2]):out+=lookup[ap[:,i,:,None,:]&wp[:,i,None,:,:]].sum(-1,dtype=np.uint32)
    return out

def main():
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled
    p=json.loads((OUT/'protocol.json').read_text());a=guard(p);m=json.loads((OUT/'manifest.json').read_text());assert m['complete'] and m['protocol_sha256']==sha(OUT/'protocol.json')
    assert len(m['rows'])==300
    with (OUT/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ap=OUT/'native_audit_progress.json'
        audit=json.loads(ap.read_text()) if ap.exists() else {'rows':[],'native_updates':0,'counter_entries_exact':0,'protocol_sha256':sha(OUT/'protocol.json')}
        assert audit['protocol_sha256']==sha(OUT/'protocol.json');done={r['name']:r for r in audit['rows']}
        x=np.load(a['data_path'])['test_inputs'].astype(float).reshape(1250,8,64);low=a['input_lower'];span=a['input_span'];xn=(x-low)/span;xc=independent_encode(np.clip(xn,0,1))
        for row in m['rows']:
            assert sha(row['path'])==row['sha256']
            if row['name'] in done:assert done[row['name']]['raw_sha256']==row['sha256'];continue
            guard(p);model=np.load(a['models'][row['model']]['controls_path']);raw=np.load(row['path']);boolean=row['regime']=='R3';t=row['T'];clip=0;elements=0
            for first in range(0,1250,CHUNK):
                n=min(CHUNK,1250-first);h=x[first:first+n];hlo=float(model['hidden_lower']);hspan=float(model['hidden_span'])
                for layer,(ni,no) in enumerate([(64,32),(32,10)]):
                    wfield=independent_fields(independent_encode(model[f'weight_{layer}']))
                    np.testing.assert_array_equal(wfield,model[f'weight_field_{layer}'])
                    if boolean:
                        codes=xc[first:first+n] if layer==0 else independent_encode(np.clip((h-hlo)/hspan,0,1))
                        if layer==1:np.testing.assert_array_equal(codes,raw['hidden_codes'][first:first+n])
                        field=np.concatenate([np.broadcast_to(wfield.ravel(),(n,ni*no)),independent_fields(codes).reshape(n,-1)],axis=1)
                    else:field=np.broadcast_to(wfield.ravel(),(n,ni*no)).copy()
                    states=np.asarray(sampler(field.shape[1],t)(keys(int(boolean),row['model'],row['seed'],t,layer,first,n),jnp.asarray(field)))
                    w=states[:,:,:ni*no].reshape(n,t,ni,no);b=model[f'bias_{layer}'].astype(float)
                    if boolean:
                        act=states[:,:,ni*no:].reshape(n,t,8,ni);aa=packed_and_counts(w,act);cc=w.sum((1,2),dtype=np.uint32)
                        assert raw[f'and_{layer}'].dtype==raw[f'column_{layer}'].dtype==np.dtype('uint32')
                        np.testing.assert_array_equal(aa,raw[f'and_{layer}'][first:first+n]);np.testing.assert_array_equal(cc,raw[f'column_{layer}'][first:first+n]);elements+=aa.size+cc.size
                        y=(low if layer==0 else hlo)*cc[:,None,:].astype(float)/t+(span if layer==0 else hspan)*aa.astype(float)/t+b
                    else:
                        wc=w.sum(1,dtype=np.uint32);assert raw[f'weight_{layer}'].dtype==np.dtype('uint16');np.testing.assert_array_equal(wc,raw[f'weight_{layer}'][first:first+n]);elements+=wc.size
                        y=np.matmul(h,wc.astype(float)/t)+b
                        direct=np.einsum('cbi,cio->cbo',h,wc.astype(float)/t,optimize=False)+b
                        np.testing.assert_allclose(y,direct,rtol=2e-12,atol=2e-11)
                    h=np.maximum(y,0) if layer==0 else y
                    if boolean and layer==0:clip+=int(np.count_nonzero((h<hlo)|(h>hlo+hspan)))
                np.testing.assert_array_equal(h,raw['logits'].reshape(1250,8,10)[first:first+n])
            assert clip==row['hidden_clipped_coordinates'];assert row['input_outside_requested_range']==np.count_nonzero((xn<0)|(xn>1))
            updates=1250*t*(3136 if boolean else 2368);assert updates==row['native_updates']
            audit['rows'].append({'name':row['name'],'raw_sha256':row['sha256'],'native_updates':updates,'counter_entries_exact':elements,'all_final_logits_exact':True,'all_second_layer_codes_exact':True});audit['native_updates']+=updates;audit['counter_entries_exact']+=elements
            atomic_json(ap,audit);print(json.dumps({'audited_cases':len(audit['rows']),'last':row['name'],'native_updates':audit['native_updates']}),flush=True)
        assert audit['native_updates']==m['native_updates']==609168000000 and len(audit['rows'])==300;guard(p)
        atomic_json(OUT/'native_audit.json',{'passed':True,**audit,'producer_manifest_sha256':sha(OUT/'manifest.json'),'scope':'Every public raw-state count, derived hidden control and final causal logit reproduced for all300 full10000-image cases. Test data consumed; no hardware measurement.'})
if __name__=='__main__':main()
