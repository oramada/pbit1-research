"""Actual ideal all-free THRML operators for the frozen six-block study."""
import json,sys,traceback,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .calibrated_transformer_native import CalibratedGroupedLinear
from .block_cached_control_worker import sha,atomic_json as save_json,atomic_npy as save_array
from .calibrated_mnist import digest
from .provenance import manifest
from .packed_float_outputs import save_output
OUT=Path('results/thrml_spw/sample_allocation')

class Worker:
    def __init__(self):
        self.p=json.loads((OUT/'protocol.json').read_text());self.layers={};self.fn={}
        self.sources={str(Path('research/thrml_spw')/name):sha(Path('research/thrml_spw')/name) for name in ['sample_allocation_test_worker.py','packed_float_outputs.py','calibrated_transformer_native.py','calibrated_transformer_controls.py','controlled_positive_native.py','native.py','grouped_range_native.py']}
    def execute(self,r):
        stage,seed=int(r['stage']),int(r['seed']);t=int(r['samples']);phase=r['phase'];condition=r['condition'];assert phase=='test';assert 0<=stage<24 and seed in range(5)
        if phase=='validation':assert t==256 and condition=='isolated'
        else:
            assert phase=='test' and condition in ['uniform','allocated'];selection=json.loads((OUT/'selection_audit.json').read_text());assert selection['passed'] and sha(OUT/'allocation.json')==selection['allocation_sha256'];allocation=json.loads((OUT/'allocation.json').read_text());assert t==(1024 if condition=='uniform' else allocation['samples'][stage])
        for name,h in self.sources.items():assert sha(name)==h
        c=self.p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];xp,yp=Path(r['inputs']),Path(r['output']);assert sha(xp)==r['input_sha256']
        signature={'stage':stage,'seed':seed,'phase':phase,'condition':condition,'samples':t,'contexts_per_bank':8,'inputs_sha256':r['input_sha256'],'controls_sha256':c['sha256'],'source_sha256':self.sources}
        if yp.with_suffix('.json').exists():
            old=json.loads(yp.with_suffix('.json').read_text());assert old['signature']==signature and sha(yp)==old['output_sha256'];return {'status':'ok','metadata':str(yp.with_suffix('.json')),'cached':True}
        if stage not in self.layers:
            self.layers[stage]=CalibratedGroupedLinear(c['path'])
        if (stage,t) not in self.fn:self.fn[stage,t]=self.layers[stage].forward(t)
        layer=self.layers[stage];x=np.load(xp);assert x.shape[:2]==(32,65);ys=[];start=time.monotonic();fields=layer.nominal
        for bank,first in enumerate(range(0,32,8)):
            key=jax.random.key(1180730 if phase=='validation' else 1190730)
            for v in [seed,stage,bank]:key=jax.random.fold_in(key,v)
            xx=x[first:first+8];y=self.fn[stage,t](key,jnp.asarray(xx.reshape(-1,xx.shape[-1])),jnp.asarray(fields));ys.append(np.asarray(y).reshape(8,65,-1))
        y=np.concatenate(ys);save_output(yp,y);q=(1/(1+np.exp(-2*fields.astype(float)))).reshape(layer.shape);mean=x.astype(float)@(layer.lower+layer.span*q)+np.asarray(layer.bias)
        n=layer.bank.logical_pbits
        meta={'status':'complete','signature':signature,'input_shape':list(x.shape),'output_shape':list(y.shape),'output_sha256':sha(yp),'field_sha256':digest(fields),'free_nodes':n,'clamped_nodes':0,'bank_batches':4,'free_node_updates':4*t*n,'logical_weight_readouts':4*t*n,'binary_product_terms':32*65*t*n,'predicted_sampling_mse':float(np.mean(np.mean(x.astype(float)**2,axis=(0,1))@(layer.span.astype(float)**2*q*(1-q)))/t),'sampling_mse_vs_programmed_expectation':float(np.mean((y.astype(float)-mean)**2)),'elapsed_seconds':time.monotonic()-start,'scope':'Actual official THRML ideal unary finite8bit operator. All weight nodes free; no reconstructed deterministic weight used in production.'}
        manifest(yp.with_suffix('.manifest.json'),r,meta);save_json(yp.with_suffix('.json'),meta);return {'status':'ok','metadata':str(yp.with_suffix('.json')),'cached':False}

def main():
    assert jax.default_backend()=='gpu';w=Worker();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='quit':break
            print(json.dumps(w.execute(r)),flush=True)
        except Exception as e:
            traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)
if __name__=='__main__':main()
