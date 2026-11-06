"""Predeclared native convolution witness, no trained network/quality scoring."""
import hashlib,json,subprocess,datetime
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from results.thrml_spw.convolution_native.operator import patches,decode,layer

OUT=Path('results/thrml_spw/convolution_native')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled
    assert not (OUT/'protocol.json').exists(),'No duplicate production'
    files=list(OUT.glob('*.py'))+[Path(p) for p in ['results/thrml_spw/integer_activation_native/gated_layer.py','research/thrml_spw/native.py','research/thrml_spw/controlled_positive_native.py','vendor/thrml/thrml/block_sampling.py','vendor/thrml/thrml/observers.py']]
    sources={str(p):sha(p) for p in files}
    vendor=subprocess.check_output(['git','-C','vendor/thrml','rev-parse','HEAD'],text=True).strip()
    assert vendor=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9'
    assert not subprocess.check_output(['git','-C','vendor/thrml','status','--short'],text=True)
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,'vendor_commit':vendor,'key_root':1477520,'numpy_seed':1477520,'chains':2,'samples':[1,7,64],'activation_lower':-.7,'activation_span':2.3,'weight_encoding':'Per-output minmax center/radius, one normalized probability per signed kernel weight. F32 finite8-bit field commands ±4; endpoints approximate.','geometries':[{'name':'valid','stride':1,'padding':0,'dilation':1},{'name':'pad','stride':1,'padding':1,'dilation':1},{'name':'stride','stride':2,'padding':1,'dilation':1},{'name':'dilated','stride':1,'padding':2,'dilation':2}],'scope':'Synthetic bounded convolution operator only. Shared weight trajectories across every spatial patch and two input images; integer gated/mask counters, deterministic signed-range correction and external F64 readout. Padding excluded by explicit validity counters. No CNN accuracy, independent spatial samples, physical arithmetic, energy or timing claim.','verification':'Separately constructed public sampler, all raw states; independent nested spatial convolution without im2col. Exact counters and padding-mask checks, F64 tolerance 2e-11 absolute/2e-12 relative.'}
    write(OUT/'protocol.json',protocol)
    rng=np.random.default_rng(protocol['numpy_seed']);x=rng.uniform(-1,2,(2,2,5,6));weights=rng.normal(0,.25,(3,2,3,3));bias=rng.normal(0,.1,3)
    # Include extrema, clipping and an actual interior zero on purpose.
    x[0,0,0,:4]=[-2,3,0,-.7]
    c=(weights.max((1,2,3))+weights.min((1,2,3)))/2;r=(weights.max((1,2,3))-weights.min((1,2,3)))/2
    p=((weights-c[:,None,None,None])/r[:,None,None,None]+1)/2
    eps=1/(1+np.exp(8));q=np.clip(p,eps,1-eps)
    commands=np.rint((np.log(q/(1-q))/2+4)*255/8).astype(np.uint8)
    fields=(-4+commands.astype(float)*8/255).astype(np.float32).reshape(3,-1).T
    np.savez_compressed(OUT/'inputs.npz',x=x,weight=weights,bias=bias,center=c,radius=r,commands=commands,fields=fields)
    rows=[];updates=0
    for gidx,geom in enumerate(protocol['geometries']):
        obs,codes,mask,shape=patches(x,(3,3),geom['stride'],geom['padding'],geom['dilation'],-.7,2.3)
        bank=layer((2,3,3),3,len(obs));assert len(bank.bank.free_indices)==54 and not len(bank.bank.clamped_indices)
        ff=np.broadcast_to(fields.ravel(),(2,54)).copy();aa=np.broadcast_to(obs,(2,*obs.shape)).copy()
        for t in protocol['samples']:
            root=jax.random.fold_in(jax.random.fold_in(jax.random.key(protocol['key_root']),gidx),t);keys=jax.random.split(root,2)
            gated,column=map(np.asarray,bank.kernel(t)(keys,jnp.asarray(ff),jnp.asarray(aa)))
            assert gated.dtype==column.dtype==np.int32
            y=decode(gated,shape,codes,mask,t,-.7,2.3,c,r,bias)
            name=f"{geom['name']}_T{t}";path=OUT/(name+'.npz')
            np.savez_compressed(path,codes=codes,mask=mask,gated=gated,column=column,output=y)
            u=2*54*t;updates+=u;rows.append({'name':name,'geometry':gidx,'T':t,'path':str(path),'sha256':sha(path),'native_updates':u,'positions_per_chain':len(codes),'weight_nodes':54,'activation_pbits':0,'padding_slots':int(np.count_nonzero(mask==0))})
            print(json.dumps(rows[-1]),flush=True)
    for file,h in sources.items():assert sha(file)==h
    write(OUT/'manifest.json',{'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'inputs_sha256':sha(OUT/'inputs.npz'),'native_updates':updates,'rows':rows,'independently_audited':False})

if __name__=='__main__':main()
