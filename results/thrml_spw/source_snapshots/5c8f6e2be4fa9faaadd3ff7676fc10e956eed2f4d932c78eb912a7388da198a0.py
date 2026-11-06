"""Independent public raw states and direct nested-loop convolution reference."""
from pathlib import Path
import hashlib,json
import numpy as np
import jax
import jax.numpy as jnp
from thrml import Block,BlockGibbsSpec,SpinNode,FactorSamplingProgram,SamplingSchedule,sample_states
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
OUT=Path('results/thrml_spw/convolution_native')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['complete'] and sha(OUT/'protocol.json')==m['protocol_sha256'] and sha(OUT/'inputs.npz')==m['inputs_sha256']
    for f,h in p['source_sha256'].items():assert sha(f)==h
    data=np.load(OUT/'inputs.npz');x=data['x'];w=data['weight'];b=data['bias'];lower=p['activation_lower'];span=p['activation_span']
    center=np.array([(np.max(ww)+np.min(ww))/2 for ww in w]);radius=np.array([(np.max(ww)-np.min(ww))/2 for ww in w]);np.testing.assert_array_equal(center,data['center']);np.testing.assert_array_equal(radius,data['radius'])
    probability=((w-center[:,None,None,None])/radius[:,None,None,None]+1)/2
    command=np.rint((np.log(np.clip(probability,1/(1+np.exp(8)),1-1/(1+np.exp(8)))/(1-np.clip(probability,1/(1+np.exp(8)),1-1/(1+np.exp(8)))))/2+4)*255/8).astype(np.uint8)
    np.testing.assert_array_equal(command,data['commands']);fields=(-4+command.astype(float)*8/255).astype(np.float32).reshape(3,-1).T;np.testing.assert_array_equal(fields,data['fields'])
    block=Block([SpinNode() for _ in range(54)]);spec=BlockGibbsSpec([block],[],{SpinNode:jax.ShapeDtypeStruct((),jnp.bool_)})
    program=FactorSamplingProgram(spec,[SpinGibbsConditional()],[SpinEBMFactor([block],jnp.asarray(fields.ravel()))],[])
    entries=0;raw_bits=0;rows=[];maxerr=0.
    for row in m['rows']:
        assert sha(row['path'])==row['sha256'];d=np.load(row['path']);t=row['T'];geom=p['geometries'][row['geometry']];stride,pad,dil=geom['stride'],geom['padding'],geom['dilation']
        root=jax.random.fold_in(jax.random.fold_in(jax.random.key(p['key_root']),row['geometry']),t);keys=jax.random.split(root,2)
        def sample(key):return sample_states(key,program,SamplingSchedule(1,t,1),[jnp.zeros(54,jnp.bool_)],[],[block])[0]
        states=np.asarray(jax.jit(jax.vmap(sample))(keys));counts=states.sum(axis=1,dtype=np.uint32).reshape(2,18,3)
        np.testing.assert_array_equal(counts.sum(1,dtype=np.uint32),d['column'])
        oh=(5+2*pad-dil*2-1)//stride+1;ow=(6+2*pad-dil*2-1)//stride+1
        gate=np.zeros((2,2*2*oh*ow,3),np.uint32);ref=np.empty((2,2,3,oh,ow));codes=np.zeros((2*oh*ow,18),np.uint8);mask=np.zeros_like(codes)
        quant=lower+span*np.rint(255*np.clip((x-lower)/span,0,1))/255
        effective=(center-radius)[None,:]+2*radius[None,:]*counts.astype(float)/t
        for image in range(2):
            for rr in range(oh):
                for cc in range(ow):
                    pos=(image*oh+rr)*ow+cc;yy=np.broadcast_to(b,(2,3)).copy()
                    for ci in range(2):
                        for kr in range(3):
                            for kc in range(3):
                                ix=(ci*3+kr)*3+kc;ir=rr*stride-pad+kr*dil;ic=cc*stride-pad+kc*dil
                                if 0<=ir<5 and 0<=ic<6:
                                    code=int(round(255*min(1,max(0,(float(x[image,ci,ir,ic])-lower)/span))));codes[pos,ix]=code;mask[pos,ix]=1
                                    gate[:,pos]+=code*counts[:,ix];gate[:,2*oh*ow+pos]+=counts[:,ix]
                                    yy+=quant[image,ci,ir,ic]*effective[:,ix,:]
                    ref[:,image,:,rr,cc]=yy
        np.testing.assert_array_equal(codes,d['codes']);np.testing.assert_array_equal(mask,d['mask']);np.testing.assert_array_equal(gate,d['gated']);np.testing.assert_allclose(ref,d['output'],rtol=2e-12,atol=2e-11)
        error=float(np.max(np.abs(ref-d['output'])));maxerr=max(maxerr,error);entries+=gate.size+d['column'].size;raw_bits+=states.size
        # Diagnostic: masking matters because quantized nominal zero is nonzero.
        assert abs(lower+span*round(-lower/span*255)/255)>1e-6
        rows.append({'name':row['name'],'counter_entries_exact':int(gate.size+d['column'].size),'max_output_abs_error':error,'raw_state_bits':states.size});print(json.dumps(rows[-1]),flush=True)
    assert raw_bits==m['native_updates']==31104 and len(rows)==12
    result={'passed':True,'cases':12,'counter_entries_exact':entries,'raw_state_bits':raw_bits,'production_updates':m['native_updates'],'verification_updates':raw_bits,'max_output_abs_error':maxerr,'source_sha256':p['source_sha256'],'rows':rows,'scope':p['scope']}
    (OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
