"""Native stochastic projection requests from a live causal student forward."""
import os,json,sys,traceback,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .positional_student_common import OUT,sha,guard,condition,check_prerequisites
from .positional_operator_common import controls,digest
from .positional_operator_native import PositionalOperator
from .packed_float_outputs import save_output
from .block_cached_control_worker import atomic_json
from .provenance import manifest
class Worker:
    def __init__(self):
        assert jax.default_backend()=='gpu' and not jax.config.x64_enabled and os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0';self.p=json.loads((OUT/'protocol.json').read_text());check_prerequisites(self.p);self.banks={};self.layers={};runtime=OUT/'native_runtime_manifest.json'
        if not runtime.exists():manifest(runtime,self.p,{'backend':jax.default_backend(),'XLA_FLAGS':os.environ.get('XLA_FLAGS'),'matmul_precision':str(jax.config.jax_default_matmul_precision),'all_free':True,'new_native_samples':0})
    def execute(self,r):
        guard(self.p);cfg=condition(self.p,r['condition']);stage,seed,bank=[int(r[key]) for key in ['stage','seed','bank']];assert stage in range(24) and seed in self.p['sampling_seeds'] and bank in range(4);entry=self.p['matrices'][stage];k,t=cfg['K'],cfg['T'];c=entry['single_controls' if k==1 else 'positional_controls'];assert sha(c['path'])==c['sha256'];assert sha(r['input'])==r['input_sha256'];x=np.load(r['input']);assert x.shape==(8,65,entry['shape'][0]) and x.dtype==np.float32;yp=Path(r['output']);sig={'protocol_sha256':sha(OUT/'protocol.json'),'stage':stage,'seed':seed,'bank':bank,'condition':r['condition'],'K':k,'T':t,'input_sha256':r['input_sha256'],'controls_sha256':c['sha256']}
        if yp.with_suffix('.json').exists():
            old=json.loads(yp.with_suffix('.json').read_text());assert old['signature']==sig and sha(yp)==old['output_sha256'];return {'status':'ok','metadata':str(yp.with_suffix('.json')),'cached':True}
        if (stage,k) not in self.layers:self.layers[stage,k]=controls(c,k)
        shape,lo,span,bias,fields=self.layers[stage,k];cache=(shape,k)
        if cache not in self.banks:self.banks[cache]=PositionalOperator(shape,k)
        layer=self.banks[cache];key=jax.random.key(self.p['key_root'])
        for value in [seed,stage,k,bank]:key=jax.random.fold_in(key,value)
        start=time.monotonic();y=np.asarray(layer.forward(t)(key,jnp.asarray(x.reshape(520,-1)),jnp.asarray(fields.ravel()),jnp.asarray(lo),jnp.asarray(span),jnp.asarray(bias))).reshape(8,65,-1);assert y.dtype==np.float32 and np.isfinite(y).all();save_output(yp,y);meta={'signature':sig,'output_sha256':sha(yp),'input_shape':list(x.shape),'output_shape':list(y.shape),'fields_sha256':digest(fields),'free_nodes':k*entry['weights'],'clamped_nodes':0,'native_node_updates':k*t*entry['weights'],'sample_rounds':t,'binary_input_terms':520*k*t*entry['weights'],'elapsed_seconds':time.monotonic()-start,'scope':'Actual stochastic student inputs; direct native output accumulation, one sampled weight matrix shared520 positions per bank. External F32 arithmetic and ideal independent devices.'};atomic_json(yp.with_suffix('.json'),meta);return {'status':'ok','metadata':str(yp.with_suffix('.json')),'cached':False}
def main():
    worker=Worker();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='quit':break
            print(json.dumps(worker.execute(r)),flush=True)
        except Exception as e:traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)
if __name__=='__main__':main()
