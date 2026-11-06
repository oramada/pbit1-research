"""Independent public-state causal witness for the positional CNN comparator."""
from pathlib import Path
import hashlib,json,datetime,subprocess
import numpy as np
import jax
import jax.numpy as jnp
from results.thrml_spw.fashion_cnn_native.word_reference.native import encode,forward
from results.thrml_spw.fashion_cnn_native.compact.recover import public_draw,pooling
from results.thrml_spw.fashion_cnn_native.cohort.replay import patches
from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
ROOT=Path('results/thrml_spw/fashion_cnn_native');OUT=ROOT/'word_witness'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def recover(x,model,t,keys,enc):
    h=x.astype(float);chains,batch=h.shape[:2];saved={}
    for li,prefix in enumerate(['conv1','conv2','head']):
        w=model[prefix+'_weight'].astype(float);no=len(w);ni=w.size//no;c=np.array([(max(row.ravel())+min(row.ravel()))/2 for row in w]);r=np.array([(max(row.ravel())-min(row.ravel()))/2 for row in w]);codes=[]
        for out in range(no):
            q=(w[out].ravel()-c[out])/r[out]/2+.5 if r[out] else np.full(ni,.5);codes.append(np.round(np.minimum(1,np.maximum(0,q))*15).astype(np.uint8))
        codes=np.asarray(codes).T;f=np.empty((ni,no,4),np.float32)
        for bit in range(4):f[:,:,bit]=np.where((codes>>bit)&1,4.,-4.)
        np.testing.assert_array_equal(codes,enc[li]['codes']);np.testing.assert_array_equal(f,enc[li]['fields']);np.testing.assert_array_equal(c,enc[li]['center']);np.testing.assert_array_equal(r,enc[li]['radius'])
        raw=np.asarray(public_draw(ni*no*4,t)(keys[li],jnp.asarray(np.broadcast_to(f.ravel(),(chains,ni*no*4)).copy())));bc=raw.sum(1,dtype=np.uint32).reshape(chains,ni,no,4);wc=sum(bc[:,:,:,bit]*np.uint32(2**bit) for bit in range(4));col=wc.sum(1,dtype=np.uint32).astype(np.int32);saved[f'column_{li}']=col;span=float(model['activation_span'][li])
        if li<2:
            hh,ww=h.shape[-2:];parts=[patches(h[ch],span) for ch in range(chains)];code=np.stack([v[0] for v in parts]);mask=np.stack([v[1] for v in parts]);g=np.einsum('cpi,cio->cpo',code.astype(np.uint32),wc,dtype=np.uint32,optimize=False);m=np.einsum('cpi,cio->cpo',mask.astype(np.uint32),wc,dtype=np.uint32,optimize=False);saved[f'gate_{li}']=np.concatenate([g,m],axis=1).astype(np.int32);saved[f'codes_{li}']=code;saved[f'mask_{li}']=mask;ys=[]
            for ch in range(chains):
                sx=np.sum(mask[ch]*(0.+span*code[ch].astype(float)/255),axis=1);sample=0.*m[ch].astype(float)/(15*t)+(span/255)*g[ch].astype(float)/(15*t);y=sx[:,None]*(c-r)+2*r*sample+model[prefix+'_bias'].astype(float);ys.append(y.reshape(batch,hh,ww,no).transpose(0,3,1,2))
            h=pooling(np.stack(ys))
        else:
            h=h.reshape(chains,batch,ni);a=np.rint(255*np.clip(h/span,0,1)).astype(np.uint8);g=np.einsum('cbi,cio->cbo',a.astype(np.uint32),wc,dtype=np.uint32,optimize=False).astype(np.int32);saved[f'codes_{li}']=a;saved[f'gate_{li}']=g;qx=0.+span*a.astype(float)/255;sample=0.*col[:,None,:].astype(float)/(15*t)+(span/255)*g.astype(float)/(15*t);h=qx.sum(-1)[:,:,None]*(c-r)+2*r*sample+model[prefix+'_bias'].astype(float)
        saved[f'output_{li}']=h
    return saved
def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled;OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'No duplicate witness';train=json.loads((ROOT/'training/manifest.json').read_text());assert train['complete'] and json.loads((ROOT/'training/evidence_audit.json').read_text())['passed'];sources=dict(json.loads((ROOT/'cohort_evidence/protocol.json').read_text())['source_sha256']);sources.update({str(p):sha(p) for p in (ROOT/'word_reference').glob('*.py')});p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,'training_manifest_sha256':sha(ROOT/'training/manifest.json'),'validation_sha256':sha(ROOT/'training/validation.npz'),'K':4,'field_magnitudes':4.,'word_coefficients':[1,2,4,8],'normalizing_denominator':15,'word_samples':[1,4,16,64,256],'future_matched_single_samples':[64,256,1024],'future_matched_word_samples':[16,64,256],'key_root':1567520,'models':5,'chains':2,'validation_inputs_per_bank':4,'scope':'Bounded validation-only internal4nodepositional reference; all bits free atfinite±4fields. Same center/radius/ACT8ranges/trainedCNNs as1node. Not asserted Extropic hardware architecture, test quality or physical energy/latency/area result.'};assert subprocess.check_output(['git','-C','vendor/thrml','rev-parse','HEAD'],text=True).strip()=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9';(OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');x=np.load(ROOT/'training/validation.npz')['inputs'][:8].astype(float).reshape(2,4,1,14,14);rows=[];updates=0;count=0
    for mi,row in enumerate(train['rows']):
        assert sha(row['path'])==row['sha256'];model=np.load(row['path']);enc=encode(model)
        for t in p['word_samples']:
            key=jax.random.fold_in(jax.random.fold_in(jax.random.key(p['key_root']),mi),t);keys=[jax.random.split(jax.random.fold_in(key,li),2) for li in range(3)];raw=forward(x,model,enc,t,keys);reference=recover(x,model,t,keys,enc);np.testing.assert_array_equal(receipt(raw),receipt(reference))
            for name in raw:np.testing.assert_array_equal(raw[name],reference[name])
            file=OUT/f'model{mi}_T{t}.npz';np.savez_compressed(file,**raw);u=2*t*4*2664;updates+=u;entries=sum(raw[f'gate_{li}'].size+raw[f'column_{li}'].size for li in range(3));count+=entries;rows.append({'model':mi,'T_word':t,'native_updates':u,'public_updates':u,'counter_entries_exact':entries,'path':str(file),'sha256':sha(file)});print(json.dumps(rows[-1]),flush=True)
    assert len(rows)==25 and updates==36336960 and count==944500
    for file,h in sources.items():assert sha(file)==h,file
    result={'passed':True,'rows':rows,'native_updates':updates,'public_updates':updates,'counter_entries_exact':count,'bank_arrays_byte_exact':700,'largest_gated_counter_bound':255*144*15*256,'all_nodes_free':True,'activation_pbits':0,'protocol_sha256':sha(OUT/'protocol.json'),'scope':p['scope']};(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
