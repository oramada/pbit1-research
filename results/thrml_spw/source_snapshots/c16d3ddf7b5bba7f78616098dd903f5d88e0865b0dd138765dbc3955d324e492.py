"""Frozen cohort conventions; no task scoring at import."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path('results/thrml_spw/boolean_classifier')
PREP=Path('results/thrml_spw/boolean_classifier_preparation_v2/preparation.json')
CHUNK=16
ROOT_KEY=1337520
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():return {str(p):sha(p) for p in sorted(Path('research/thrml_spw').glob('*.py'))}
def guard(p):
    for path,h in p['critical_sources'].items():assert sha(path)==h,(path,'source changed')
    assert sha(PREP)==p['preparation_sha256']
    a=json.loads(PREP.read_text());assert sha(a['data_path'])==a['data_sha256']
    for r in a['models']:assert sha(r['model_path'])==r['model_sha256'] and sha(r['controls_path'])==r['controls_sha256']
    return a
def encode(p):
    q=np.asarray(p,float)
    if np.any(~np.isfinite(q)) or np.any((q<0)|(q>1)):raise ValueError('probability out of range')
    eps=1/(1+np.exp(8));q=np.clip(q,eps,1-eps)
    return np.rint((.5*(np.log(q)-np.log1p(-q))+4)*255/8).astype(np.uint8)
def fields(code):return (-4+np.asarray(code,float)*8/255).astype(np.float32)
def keys(regime,model,seed,t,layer,first,count):
    import jax
    out=[]
    for bank in range(first,first+count):
        k=jax.random.key(ROOT_KEY)
        for v in [regime,model,seed,t,layer,bank]:k=jax.random.fold_in(k,v)
        out.append(k)
    return jax.numpy.stack(out)
def cases(p):
    return [{'regime':r,'model':m,'T':t,'seed':s,'name':f'{r}_model{m}_T{t}_seed{s}'} for r in ['R1','R3'] for m in range(5) for t in p['samples'][r] for s in range(10)]
def paths(c):return OUT/'cases'/(c['name']+'.npz')
def metrics(logits,labels):
    z=np.asarray(logits,float);v=z-z.max(1,keepdims=True);loss=np.log(np.exp(v).sum(1))-v[np.arange(len(v)),labels]
    return {'accuracy':float(np.mean(z.argmax(1)==labels)),'mean_cross_entropy':float(loss.mean())}
