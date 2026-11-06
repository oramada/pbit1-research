"""Fixed conventions for the matched-update internal positional CNN reference."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path('results/thrml_spw/fashion_cnn_native');OUT=ROOT/'word_comparison_evidence';PARENT=ROOT/'cohort_evidence';TRAIN=ROOT/'training'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def save(p,**a):
    q=p.with_suffix('.tmp')
    with q.open('wb') as f:np.savez_compressed(f,**a)
    q.replace(p)
def guard(p):
    for f,h in {**p['source_sha256'],**p['input_sha256']}.items():assert sha(f)==h,f
def parent_closed():
    assert json.loads((PARENT/'pipeline.json').read_text())['status']=='full_native_CNN_independently_audited_goal_active'
    for f,h in json.loads((PARENT/'completion.json').read_text())['artifact_sha256'].items():assert sha(f)==h,f
    assert json.loads((PARENT/'native_audit.json').read_text())['passed'] and json.loads((PARENT/'statistics_audit.json').read_text())['passed']
def keys(model,seed,t,li,first,n):
    import jax
    import jax.numpy as jnp
    key=jax.random.key(1577520)
    for value in [model,seed,t,li]:key=jax.random.fold_in(key,value)
    return jax.vmap(lambda bank:jax.random.fold_in(key,bank))(jnp.arange(first,first+n,dtype=jnp.uint32))
def mean_weights(model):
    epsilon=1/(1+np.exp(8));weights=[]
    for prefix in ['conv1','conv2','head']:
        w=model[prefix+'_weight'].astype(float);flat=w.reshape(len(w),-1);c=(flat.min(1)+flat.max(1))/2;r=(flat.max(1)-flat.min(1))/2;q=np.divide(flat-c[:,None],r[:,None],out=np.zeros_like(flat),where=r[:,None]>0)/2+.5;code=np.rint(np.clip(q,0,1)*15);normalized=epsilon+(1-2*epsilon)*code/15;weights.append(((c-r)[:,None]+2*r[:,None]*normalized).reshape(w.shape))
    return weights
def metrics(z,y):
    top=z.max(1);loss=top+np.log(np.exp(z-top[:,None]).sum(1))-z[np.arange(len(y)),y];return np.array([np.mean(z.argmax(1)==y),loss.mean()])
