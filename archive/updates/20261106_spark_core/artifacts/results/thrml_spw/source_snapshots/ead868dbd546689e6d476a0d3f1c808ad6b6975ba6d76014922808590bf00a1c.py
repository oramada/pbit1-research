"""Known-parameter finite-grid native oracle, explicitly diagnostic only."""
import json,sys,time,traceback
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibrated_transformer_native import CalibratedGroupedLinear
from .calibrated_transformer_controls import device
from .transformer_floor_math import fields
from .calibrated_mnist import digest
from .block_cached_control_worker import sha,atomic_json,atomic_npy
from .provenance import manifest


class Worker:
    def __init__(self):
        self.layers={};self.fns={};self.sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['transformer_floor_worker.py','transformer_floor_math.py','calibrated_transformer_native.py','calibrated_transformer_controls.py','controlled_positive_native.py','grouped_range_native.py','native.py']}

    def execute(self,r):
        for n,h in self.sources.items():assert sha(n)==h
        assert r['condition']=='oracle8' and r['samples']==1024 and r['contexts_per_bank']==8
        stage,seed=int(r['stage']),int(r['seed']);cp=Path(r['controls']);xp=Path(r['inputs']);yp=Path(r['output']);jp=yp.with_suffix('.json')
        signature={k:r[k] for k in ['condition','stage','seed','samples','contexts_per_bank']};signature.update(controls_sha256=sha(cp),inputs_sha256=sha(xp),source_sha256=self.sources)
        if jp.exists() and yp.exists():
            old=json.loads(jp.read_text())
            if old['signature']==signature and old['output_sha256']==sha(yp):return {'status':'ok','metadata':str(jp),'cached':True}
        cache=stage,sha(cp)
        if cache not in self.layers:
            self.layers[cache]=CalibratedGroupedLinear(cp);self.fns[cache]=self.layers[cache].forward(1024)
        layer=self.layers[cache];n=layer.bank.logical_pbits;gain,offset=device(stage,seed,n)
        fs,code,inverse=fields(layer.nominal,gain,offset,np.zeros(n,np.uint8));actual=fs['oracle8'];x=np.load(xp);ys=[];start=time.monotonic()
        for bank,first in enumerate(range(0,len(x),8)):
            key=jax.random.key(1140730)
            for value in [seed,stage,bank]:key=jax.random.fold_in(key,value)
            xx=x[first:first+8];v=self.fns[cache](key,jnp.asarray(xx.reshape(-1,xx.shape[-1])),jnp.asarray(actual));ys.append(np.asarray(v).reshape(xx.shape[0],xx.shape[1],-1))
        y=np.concatenate(ys);atomic_npy(yp,y)
        probability=(1/(1+np.exp(-2*actual.astype(float)))).reshape(layer.shape);nominal=(1/(1+np.exp(-2*layer.nominal.astype(float)))).reshape(layer.shape)
        expected=x.astype(float)@(layer.lower+layer.span*probability)+np.asarray(layer.bias)
        nominal_output=x.astype(float)@(layer.lower+layer.span*nominal)+np.asarray(layer.bias)
        energy=np.mean(x.astype(float)**2,axis=(0,1));var=float(np.mean(energy@(layer.span.astype(float)**2*probability*(1-probability)))/1024)
        result={'status':'complete','signature':signature,'output_sha256':sha(yp),'field_sha256':digest(actual),'oracle_command_sha256':digest(code),'free_nodes':n,'clamped_nodes':0,'bank_batches':len(ys),'free_node_updates':len(ys)*1024*n,'logical_weight_readouts':len(ys)*1024*n,
            'binary_product_terms':int(np.prod(x.shape[:2])*1024*n),'known_inverse_bound_clipped':int(np.count_nonzero(np.abs(inverse)>4)),
            'mean_bias_mse_vs_nominal_program':float(np.mean((expected-nominal_output)**2)),'sampling_mse_vs_programmed_expectation':float(np.mean((y.astype(float)-expected)**2)),'predicted_sampling_mse':var,'elapsed_seconds':time.monotonic()-start,
            'scope':'Actual THRML all-free oracle8 finite controller with known true synthetic device parameters. Nondeployable diagnostic, no probes or estimator; same oldconsumedtext/devices/MCkeys. No claim of implementable calibrated quality or hardware advantage.'}
        manifest(yp.with_suffix('.manifest.json'),r,result);atomic_json(jp,result);return {'status':'ok','metadata':str(jp),'cached':False}


def main():
    if jax.default_backend()!='gpu':raise RuntimeError('CUDA required')
    worker=Worker();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='quit':break
            print(json.dumps(worker.execute(r)),flush=True)
        except Exception as e:
            traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)


if __name__=='__main__':main()
