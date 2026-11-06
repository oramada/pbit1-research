"""Complete-layer direct THRML execution for the frozen fine-range confirmation."""
import json,sys,traceback,time,os,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .localized_native import LocalizedLinear
from .fine_range_common import OUT,condition
from .block_cached_control_worker import sha,atomic_json
from .packed_float_outputs import save_output
from .provenance import manifest
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
class Worker:
    def __init__(self):
        assert os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0' and jax.default_backend()=='gpu' and not jax.config.x64_enabled
        self.p=json.loads((OUT/'protocol.json').read_text());self.execution=json.loads((OUT/'execution.json').read_text());assert self.execution['protocol_sha256']==sha(OUT/'protocol.json');self.layers={};self.fn={}
    def execute(self,r):
        assert sha(OUT/'protocol.json')==self.execution['protocol_sha256'] and json.loads((OUT/'execution.json').read_text())==self.execution
        for name,h in self.execution['source_sha256'].items():assert sha(name)==h
        stage,seed=int(r['stage']),int(r['seed']);assert stage in range(24) and seed in self.p['sampling_seeds'];cfg=condition(self.p,r['condition']);g,t=cfg['group'],cfg['samples'];assert r['samples']==t and r['group']==g;c=self.p['control_artifacts'][str(g)][stage];assert sha(c['path'])==c['sha256'];xp,yp=Path(r['inputs']),Path(r['output']);assert sha(xp)==r['input_sha256'];signature={'stage':stage,'seed':seed,'condition':r['condition'],'samples':t,'group':g,'contexts_per_bank':8,'inputs_sha256':r['input_sha256'],'controls_sha256':c['sha256'],'execution_sha256':sha(OUT/'execution.json'),'XLA_FLAGS':os.environ['XLA_FLAGS']}
        if yp.with_suffix('.json').exists():
            old=json.loads(yp.with_suffix('.json').read_text());assert old['signature']==signature and sha(yp)==old['output_sha256'];return {'status':'ok','metadata':str(yp.with_suffix('.json')),'cached':True}
        if (stage,g) not in self.layers:self.layers[stage,g]=LocalizedLinear(c['path'],g)
        layer=self.layers[stage,g]
        if (stage,g,t) not in self.fn:self.fn[stage,g,t]=layer.forward(t)
        x=np.load(xp);assert x.shape==(32,65,layer.shape[0]) and x.dtype==np.float32;start=time.monotonic();ys=[]
        for bank,first in enumerate(range(0,32,8)):
            key=jax.random.key(self.p['key_root'])
            for v in [seed,stage,bank]:key=jax.random.fold_in(key,v)
            y=self.fn[stage,g,t](key,jnp.asarray(x[first:first+8].reshape(520,-1)),jnp.asarray(layer.nominal));ys.append(np.asarray(y).reshape(8,65,-1))
        y=np.concatenate(ys);assert np.isfinite(y).all();save_output(yp,y);q=(1/(1+np.exp(-2*layer.nominal.astype(float)))).reshape(layer.shape);mean=x.astype(float)@(layer.lower.astype(float)+layer.span.astype(float)*q)+np.asarray(layer.bias);n=layer.bank.logical_pbits
        meta={'status':'complete','signature':signature,'input_shape':list(x.shape),'output_shape':list(y.shape),'output_sha256':sha(yp),'field_sha256':digest(layer.nominal),'free_nodes':n,'clamped_nodes':0,'bank_batches':4,'free_node_updates':4*t*n,'logical_weight_readouts':4*t*n,'binary_product_terms':32*65*t*n,'predicted_sampling_mse':float(np.mean((x.astype(float)**2)@(layer.span.astype(float)**2*q*(1-q)))/t),'sampling_mse_vs_programmed_expectation':float(np.mean((y.astype(float)-mean)**2)),'elapsed_seconds':time.monotonic()-start,'scope':'Native direct sums on actual current student activations; one free node per normalized converted weight, finite controls/ranges, external F32 arithmetic. No physical hardware costs.'};manifest(yp.with_suffix('.manifest.json'),r,meta);atomic_json(yp.with_suffix('.json'),meta);return {'status':'ok','metadata':str(yp.with_suffix('.json')),'cached':False}
def main():
    w=Worker();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='quit':break
            print(json.dumps(w.execute(r)),flush=True)
        except Exception as e:traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)
if __name__=='__main__':main()
