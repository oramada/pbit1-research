"""Native full-matrix probes for fixed local-range/sample comparisons."""
import json,sys,time,traceback
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .localized_native import LocalizedLinear
from .block_cached_control_worker import sha,atomic_npy,atomic_json
from .calibrated_mnist import digest
from .provenance import manifest
OUT=Path('results/thrml_spw/range_localization')
class Worker:
    def __init__(self):
        self.p=json.loads((OUT/'protocol.json').read_text());self.layers={};self.fn={};self.sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['range_localization_worker.py','localized_native.py','localized_controls.py','calibrated_transformer_native.py','grouped_range_native.py','native.py','controlled_positive_native.py']}
    def execute(self,r):
        stage,seed=int(r['stage']),int(r['seed']);condition=next(c for c in self.p['native_conditions'] if c['name']==r['condition']);g,t=condition['group'],condition['samples'];assert 0<=stage<24 and seed in range(5)
        for name,h in self.sources.items():assert sha(name)==h
        c=self.p['control_artifacts'][str(g)][stage];assert sha(c['path'])==c['sha256'];xp,yp,tp=Path(r['inputs']),Path(r['output']),Path(r['target']);assert sha(xp)==r['inputs_sha256'] and sha(tp)==r['target_sha256'];signature={'condition':r['condition'],'stage':stage,'seed':seed,'group':g,'samples':t,'inputs_sha256':sha(xp),'target_sha256':sha(tp),'controls_sha256':c['sha256'],'source_sha256':self.sources}
        if yp.with_suffix('.json').exists():
            old=json.loads(yp.with_suffix('.json').read_text());assert old['signature']==signature and sha(yp)==old['output_sha256'];return {'status':'ok','metadata':str(yp.with_suffix('.json')),'cached':True}
        if (stage,g) not in self.layers:self.layers[stage,g]=LocalizedLinear(c['path'],g)
        layer=self.layers[stage,g]
        if (stage,g,t) not in self.fn:self.fn[stage,g,t]=layer.forward(t)
        x=np.load(xp);assert x.shape==(64,layer.shape[0]);key=jax.random.key(1200730)
        for v in [seed,stage]:key=jax.random.fold_in(key,v)
        start=time.monotonic();y=np.asarray(self.fn[stage,g,t](key,jnp.asarray(x),jnp.asarray(layer.nominal)));atomic_npy(yp,y);target=np.load(tp);q=(1/(1+np.exp(-2*layer.nominal.astype(float)))).reshape(layer.shape);mean=x.astype(float)@(layer.lower.astype(float)+layer.span.astype(float)*q)+np.asarray(layer.bias);variance=float(np.mean(x.astype(float)**2@(layer.span.astype(float)**2*q*(1-q)))/t);bias=float(np.mean((mean-target)**2));n=layer.bank.logical_pbits
        m={'status':'complete','signature':signature,'output_sha256':sha(yp),'input_shape':list(x.shape),'output_shape':list(y.shape),'field_sha256':digest(layer.nominal),'free_nodes':n,'clamped_nodes':0,'bank_batches':1,'observed_node_updates':t*n,'logical_weight_readouts':t*n,'binary_product_terms':64*t*n,'predicted_sampling_MSE':variance,'programmed_mean_bias_MSE':bias,'predicted_target_MSE':variance+bias,'observed_target_MSE':float(np.mean((y.astype(float)-target)**2)),'observed_MSE_vs_programmed_mean':float(np.mean((y.astype(float)-mean)**2)),'elapsed_seconds':time.monotonic()-start,'scope':'Actual official THRML direct linear sum, one free node per normalized weight. All64 fixed FP32 anchor inputs share one sampled weight bank. No full stochastic NN task inference or physical cost claim.'};manifest(yp.with_suffix('.manifest.json'),r,m);atomic_json(yp.with_suffix('.json'),m);return {'status':'ok','metadata':str(yp.with_suffix('.json')),'cached':False}
def main():
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled;w=Worker();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='quit':break
            print(json.dumps(w.execute(r)),flush=True)
        except Exception as e:traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)
if __name__=='__main__':main()
