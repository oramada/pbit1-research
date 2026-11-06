"""Pure NumPy conventions and capacity/accounting for the full CNN cohort."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path('results/thrml_spw/fashion_cnn_native');OUT=ROOT/'cohort_evidence';TRAIN=ROOT/'training';PARENT=ROOT/'throughput_evidence'
CONTROLS=['FP32_F64','FP32_F32','ACT8_FP32','ACT8_INT2_minmax','ACT8_INT4_minmax','ACT8_INT8_minmax','ACT8_INT2_symmetric','ACT8_INT4_symmetric','ACT8_INT8_symmetric','ACT8_nominal_programmed_mean']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def save(p,**arrays):
    q=p.with_suffix('.tmp')
    with q.open('wb') as f:np.savez_compressed(f,**arrays)
    q.replace(p)
def guard(p):
    for f,h in {**p['source_sha256'],**p['input_sha256']}.items():assert sha(f)==h,f
def parent_closed():
    assert json.loads((PARENT/'pipeline.json').read_text())['status']=='GPU_CNN_validation_preflight_audited_goal_active'
    assert json.loads((PARENT/'evidence_audit.json').read_text())['passed']
    for f,h in json.loads((PARENT/'completion.json').read_text())['artifact_sha256'].items():assert sha(f)==h,f
def parameters(w):
    flat=np.asarray(w,float).reshape(w.shape[0],-1);lo=flat.min(1);hi=flat.max(1);c=(lo+hi)/2;r=(hi-lo)/2
    q=np.divide(flat-c[:,None],r[:,None],out=np.zeros_like(flat),where=r[:,None]>0)/2+.5;e=1/(1+np.exp(8));q=np.clip(q,e,1-e);code=np.rint((np.log(q/(1-q))/2+4)*255/8).astype(np.uint8);fields=(-4+code.astype(float)*8/255).astype(np.float32)
    return c,r,code,fields
def control_weights(model,name):
    out=[]
    for prefix in ['conv1','conv2','head']:
        w=model[prefix+'_weight'].astype(float);flat=w.reshape(w.shape[0],-1);c,r,code,fields=parameters(w)
        if '_minmax' in name:
            bits=int(name.split('_INT')[1].split('_')[0]);levels=2**bits-1;q=np.divide(flat-c[:,None],r[:,None],out=np.zeros_like(flat),where=r[:,None]>0)/2+.5;flat=(c-r)[:,None]+2*r[:,None]*np.rint(np.clip(q,0,1)*levels)/levels
        elif '_symmetric' in name:
            bits=int(name.split('_INT')[1].split('_')[0]);limit=2**(bits-1)-1;scale=np.max(abs(flat),axis=1)/limit;flat=np.rint(np.divide(flat,scale[:,None],out=np.zeros_like(flat),where=scale[:,None]>0)).clip(-limit,limit)*scale[:,None]
        elif name=='ACT8_nominal_programmed_mean':flat=(c-r)[:,None]+2*r[:,None]/(1+np.exp(-2*fields.astype(float)))
        out.append(flat.reshape(w.shape))
    return out
def numpy_control(x,model,weights,quantized):
    """Independent NumPy spatial view/contraction control, no Torch/JAX."""
    h=np.asarray(x,float)
    for li,w in enumerate(weights):
        if quantized:
            span=float(model['activation_span'][li]);h=span*np.rint(255*np.clip(h/span,0,1))/255
        if li<2:
            n,c,hh,ww=h.shape;pad=np.pad(h,((0,0),(0,0),(1,1),(1,1)));v=np.lib.stride_tricks.sliding_window_view(pad,(3,3),axis=(-2,-1));patch=v.transpose(0,2,3,1,4,5).reshape(n*hh*ww,-1);y=patch@w.reshape(w.shape[0],-1).T+model[['conv1_bias','conv2_bias'][li]];y=y.reshape(n,hh,ww,-1).transpose(0,3,1,2);y=np.maximum(y,0);h=y[:,:,:2*(hh//2),:2*(ww//2)].reshape(n,w.shape[0],hh//2,2,ww//2,2).max((3,5))
        else:h=h.reshape(len(h),-1)@w.T+model['head_bias']
    return h
def metrics(z,y):
    m=z.max(1);loss=m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(len(y)),y]
    return np.array([np.mean(z.argmax(1)==y),np.mean(loss)])
def key_bank(model,seed,t,layer,first,n):
    import jax
    import jax.numpy as jnp
    key=jax.random.key(1537520)
    for value in [model,seed,t,layer]:key=jax.random.fold_in(key,value)
    return jax.vmap(lambda bank:jax.random.fold_in(key,bank))(jnp.arange(first,first+n,dtype=jnp.uint32))
