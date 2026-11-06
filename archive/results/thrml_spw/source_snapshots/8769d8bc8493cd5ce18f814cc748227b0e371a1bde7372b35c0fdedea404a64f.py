"""All-five-model validation-only native/public causal CNN preflight."""
from pathlib import Path
import json,hashlib,datetime,subprocess
import numpy as np
import jax
import jax.numpy as jnp
from scipy.signal import correlate2d
from research.thrml_spw.verify_boolean_classifier_study import sampler
from results.thrml_spw.fashion_cnn_native.inference.native import encode,forward

ROOT=Path('results/thrml_spw/fashion_cnn_native');OUT=ROOT/'preflight';TRAIN=ROOT/'training'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)

def reference_conv(x,w,b):
    result=np.empty((x.shape[0],x.shape[1],w.shape[1],x.shape[3],x.shape[4]))
    for ch in range(x.shape[0]):
        for image in range(x.shape[1]):
            for out in range(w.shape[1]):
                yy=np.full(x.shape[-2:],b[out],float)
                for ci in range(x.shape[2]):yy+=correlate2d(x[ch,image,ci],w[ch,out,ci],mode='same',boundary='fill',fillvalue=0)
                result[ch,image,out]=yy
    return result

def independent_pool(x):
    out=np.empty((*x.shape[:3],x.shape[3]//2,x.shape[4]//2))
    for rr in range(out.shape[-2]):
        for cc in range(out.shape[-1]):out[:,:,:,rr,cc]=np.maximum(x[:,:,:,2*rr:2*rr+2,2*cc:2*cc+2],0).max((-2,-1))
    return out

def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'No implicit duplicate preflight'
    m=json.loads((TRAIN/'manifest.json').read_text());audit=json.loads((TRAIN/'evidence_audit.json').read_text());assert m['complete'] and audit['passed'] and not m['test_scored']
    paths=list((ROOT/'inference').glob('*.py'))+[Path(q) for q in ['results/thrml_spw/integer_activation_native/gated_layer.py','results/thrml_spw/convolution_native/operator.py','research/thrml_spw/native.py','research/thrml_spw/controlled_positive_native.py','research/thrml_spw/verify_boolean_classifier_study.py','vendor/thrml/thrml/block_sampling.py','vendor/thrml/thrml/observers.py']]
    sources={str(p):sha(p) for p in paths};p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,'training_manifest_sha256':sha(TRAIN/'manifest.json'),'training_audit_sha256':sha(TRAIN/'evidence_audit.json'),'validation_sha256':sha(TRAIN/'validation.npz'),'models':[0,1,2,3,4],'validation_indices':list(range(8)),'chains':2,'images_per_bank':4,'samples':[1,7,64],'key_root':1517520,'vendor_commit':subprocess.check_output(['git','-C','vendor/thrml','rev-parse','HEAD'],text=True).strip(),'scope':'Bounded full-causal CNN validation witness, all5 trained checkpoints; no task labels/scoring or full test study. Signed per-output range weights become normalized probabilities. No activation p-bits; mask counters preserve exact padding; F64 external range readout/ReLU/maxpool. Same weight trajectory shared across4images/positions per bank.'}
    assert p['vendor_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9';write(OUT/'protocol.json',p)
    x=np.load(TRAIN/'validation.npz')['inputs'][:8].astype(float).reshape(2,4,1,14,14);rows=[];updates=0;entries=0;output_coordinates=0;max_error=0.
    for mi,modelrow in enumerate(m['rows']):
        assert mi==modelrow['seed'] and sha(modelrow['path'])==modelrow['sha256'];model=np.load(modelrow['path']);enc=encode(model)
        for t in p['samples']:
            key=jax.random.fold_in(jax.random.fold_in(jax.random.key(p['key_root']),mi),t);keys=[jax.random.split(jax.random.fold_in(key,li),2) for li in range(3)]
            raw=forward(x,model,enc,t,keys);h=x;case_entries=0;caseerr=0.
            for li,(prefix,e) in enumerate(zip(['conv1','conv2','head'],enc)):
                weights=model[prefix+'_weight'].astype(float);no=weights.shape[0];ni=weights.size//no
                # Independent per-channel mapping and finite field calculation.
                c=np.array([(ww.max()+ww.min())/2 for ww in weights]);r=np.array([(ww.max()-ww.min())/2 for ww in weights]);q=np.stack([((weights[o].ravel()-c[o])/r[o]+1)/2 if r[o] else np.full(ni,.5) for o in range(no)])
                q=np.clip(q,1/(1+np.exp(8)),1-1/(1+np.exp(8)));code=np.rint((np.log(q/(1-q))/2+4)*255/8).astype(np.uint8);fields=(-4+code.astype(float)*8/255).astype(np.float32).T
                np.testing.assert_array_equal(code,e['commands']);np.testing.assert_array_equal(fields,e['fields'])
                states=np.asarray(sampler(ni*no,t)(keys[li],jnp.asarray(np.broadcast_to(fields.ravel(),(2,ni*no)).copy())));wc=states.sum(1,dtype=np.uint32).reshape(2,ni,no);np.testing.assert_array_equal(wc.sum(1,dtype=np.uint32),raw[f'column_{li}'])
                if li<2:
                    ac=raw[f'codes_{li}'];mask=raw[f'mask_{li}'];gate=np.einsum('cpi,cio->cpo',ac.astype(np.uint32),wc,dtype=np.uint32,optimize=False);valid=np.einsum('cpi,cio->cpo',mask.astype(np.uint32),wc,dtype=np.uint32,optimize=False);np.testing.assert_array_equal(np.concatenate([gate,valid],axis=1),raw[f'gate_{li}'])
                    span=float(model['activation_span'][li]);xq=span*np.rint(255*np.clip(h/span,0,1))/255
                    # Independent spatial convolution checks ordering/masks, not just im2col identity.
                    effective=((c-r)[None,None,:]+2*r[None,None,:]*wc.astype(float)/t).transpose(0,2,1).reshape(2,*weights.shape)
                    y=independent_pool(reference_conv(xq,effective,model[prefix+'_bias']))
                else:
                    flat=h.reshape(2,4,ni);span=float(model['activation_span'][li]);a=np.rint(255*np.clip(flat/span,0,1)).astype(np.uint8);np.testing.assert_array_equal(a,raw[f'codes_{li}']);expected=np.einsum('cbi,cio->cbo',a.astype(np.uint32),wc,dtype=np.uint32,optimize=False);np.testing.assert_array_equal(expected,raw[f'gate_{li}']);effective=(c-r)[None,None,:]+2*r[None,None,:]*wc.astype(float)/t;y=np.einsum('cbi,cio->cbo',span*a.astype(float)/255,effective,optimize=False)+model[prefix+'_bias']
                np.testing.assert_allclose(y,raw[f'output_{li}'],rtol=2e-12,atol=2e-11);caseerr=max(caseerr,float(np.max(abs(y-raw[f'output_{li}']))));output_coordinates+=y.size;case_entries+=raw[f'gate_{li}'].size+raw[f'column_{li}'].size
                # Use exact canonical native readout for the next quantization boundary;
                # the independent spatial result was checked above, not silently substituted.
                h=raw[f'output_{li}']
            path=OUT/f'model{mi}_T{t}.npz';np.savez_compressed(path,**raw);u=2*t*2664;updates+=u;entries+=case_entries;max_error=max(max_error,caseerr);rows.append({'model':mi,'T':t,'path':str(path),'sha256':sha(path),'native_updates':u,'public_replay_updates':u,'counter_entries_exact':case_entries,'max_spatial_output_abs_error':caseerr});print(json.dumps(rows[-1]),flush=True)
    for file,h in sources.items():assert sha(file)==h
    assert updates==1918080 and len(rows)==15
    write(OUT/'evidence_audit.json',{'passed':True,'rows':rows,'native_updates':updates,'independent_public_replay_updates':updates,'counter_entries_exact':entries,'spatial_output_coordinates_checked':output_coordinates,'max_output_abs_error':max_error,'protocol_sha256':sha(OUT/'protocol.json'),'scope':p['scope'],'causal_boundary':'All integer gates checked from separate public states. Alternative spatial contraction checked at each layer; subsequent quantization uses canonical native F64 readout to avoid treating alternate-summation ulps as a distinct operator. This is not an independent alternate floating-boundary network implementation.'})
if __name__=='__main__':main()
