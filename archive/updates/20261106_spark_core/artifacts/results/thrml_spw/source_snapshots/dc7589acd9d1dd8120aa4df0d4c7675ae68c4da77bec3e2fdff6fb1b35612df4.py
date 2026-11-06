"""Public-factor count/F32 replay and predetermined IEEE forward-error checks."""
import json,sys,traceback,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from thrml import AbstractObserver,FactorSamplingProgram,SpinNode,SamplingSchedule,block_state_to_global,from_global_state,sample_with_observation
from thrml.models.discrete_ebm import SpinEBMFactor,SpinGibbsConditional
from .calibrated_mnist_path_audit import PublicLayer
from .block_cached_control_worker import sha,atomic_json
OUT=Path('results/thrml_spw/sample_allocation')
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

class CountAndSum(AbstractObserver):
    readout:object
    x:jax.Array
    span:jax.Array
    shape:tuple
    def init(self):return jnp.zeros(int(np.prod(self.shape)),jnp.int32),jnp.zeros((len(self.x),self.shape[1]),jnp.float32)
    def __call__(self,program,free,clamped,carry,iteration):
        state=block_state_to_global(free+clamped,program.gibbs_spec);bits=from_global_state(state,program.gibbs_spec,[self.readout])[0];value=self.x@(bits.reshape(self.shape).astype(jnp.float32)*self.span)
        return (carry[0]+bits.astype(jnp.int32),carry[1]+value),None

class Worker:
    def __init__(self):self.p=json.loads((OUT/'protocol.json').read_text());self.layers={};self.fn={};self.rows=[];self.phase=None
    def execute(self,r):
        yp=Path(r['output']);xp=Path(r['inputs']);meta=json.loads(yp.with_suffix('.json').read_text());sig=meta['signature'];phase=sig['phase'];stage,seed,t=int(sig['stage']),int(sig['seed']),int(sig['samples']);condition=sig['condition'];c=self.p['control_artifacts'][stage];assert sha(c['path'])==sig['controls_sha256']==c['sha256'] and sha(xp)==sig['inputs_sha256']==r['input_sha256'] and sha(yp)==meta['output_sha256'];man=json.loads(yp.with_suffix('.manifest.json').read_text());assert man['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9' and man['packages']['thrml']=='0.1.4'
        if self.phase is None:self.phase=phase
        assert self.phase==phase
        if phase=='validation':assert t==256 and condition=='isolated'
        else:
            allocation=json.loads((OUT/'allocation.json').read_text());assert phase=='test' and condition in ['uniform','allocated'] and t==(1024 if condition=='uniform' else allocation['samples'][stage])
        for name,h in sig['source_sha256'].items():assert sha(name)==h
        a=np.load(c['path']);shape=a['codes'].shape;n=int(np.prod(shape));lo=np.repeat(a['lower'],128,0);sp=np.repeat(a['span'],128,0);b=(-4+a['codes'].reshape(-1).astype(float)*8/255).astype(np.float32);assert meta['field_sha256']==digest(b);x,y=np.load(xp),np.load(yp);assert x.shape==(32,65,shape[0]) and y.shape==(32,65,shape[1]) and x.dtype==y.dtype==np.float32
        cache=tuple(shape)
        if cache not in self.layers:self.layers[cache]=PublicLayer(np.full(shape,.5));assert self.layers[cache].n_free==n and not self.layers[cache].clamped
        if (stage,t) not in self.fn:
            layer=self.layers[cache];span=jnp.asarray(sp);lower=jnp.asarray(lo);bias=jnp.asarray(a['bias'])
            def run(key,xx,fields):
                program=FactorSamplingProgram(layer.spec,[SpinGibbsConditional()],[SpinEBMFactor(layer.free,fields)],[]);obs=CountAndSum(layer.readout,xx,span,shape);carry,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),layer.initial,[],obs.init(),obs);return carry[0],carry[1]/t+xx@lower+bias
            self.fn[stage,t]=jax.jit(run)
        counts=[];replayed=[]
        for bank in range(4):
            key=jax.random.key(1180730 if phase=='validation' else 1190730)
            for v in [seed,stage,bank]:key=jax.random.fold_in(key,v)
            cc,yy=self.fn[stage,t](key,jnp.asarray(x[bank*8:(bank+1)*8].reshape(520,shape[0])),jnp.asarray(b));counts.append(np.asarray(cc).reshape(shape));replayed.append(np.asarray(yy).reshape(8,65,-1))
        reproduced=np.concatenate(replayed);np.testing.assert_array_equal(reproduced,y);counts=np.stack(counts);assert np.all((counts>=0)&(counts<=t));q=counts.astype(float)/t;reference=(x.reshape(4,520,shape[0]).astype(float)@(lo.astype(float)+sp.astype(float)*q)+a['bias']).reshape(y.shape);error=np.abs(y.astype(float)-reference);scaled=float(error.max()/(1+np.max(np.abs(reference))))
        u=2.**-24;d=shape[0];gd=2*d*u/(1-2*d*u);gt=(t-1)*u/(1-(t-1)*u);g2=2*u/(1-2*u);magnitude=np.abs(x.astype(float))@np.abs(sp.astype(float));center=np.abs(x.astype(float))@np.abs(lo.astype(float));bias=np.abs(a['bias'].astype(float));refg=(2*d+4)*2.**-53/(1-(2*d+4)*2.**-53)
        bound=(gd+gt*(1+gd))*magnitude+gd*center+g2*((1+gt)*(1+gd)*magnitude+(1+gd)*center+bias);bound+=(4*d+2*t+8)*np.finfo(np.float32).tiny+refg*(magnitude+center+bias);assert np.max(t*(magnitude+center+bias))<np.finfo(np.float32).max;assert np.all(error<=bound)
        prob=(1/(1+np.exp(-2*b.astype(float)))).reshape(shape);mean=x.astype(float)@(lo.astype(float)+sp.astype(float)*prob)+a['bias'];pred=float(np.mean(np.mean(x.astype(float)**2,axis=(0,1))@(sp.astype(float)**2*prob*(1-prob)))/t)
        np.testing.assert_allclose(meta['predicted_sampling_mse'],pred,atol=1e-12,rtol=1e-11);np.testing.assert_allclose(meta['sampling_mse_vs_programmed_expectation'],np.mean((y.astype(float)-mean)**2),atol=1e-12,rtol=1e-11);assert meta['free_nodes']==n and meta['clamped_nodes']==0 and meta['bank_batches']==4 and meta['free_node_updates']==meta['logical_weight_readouts']==4*t*n and meta['binary_product_terms']==32*65*t*n
        row={'phase':phase,'condition':condition,'seed':seed,'stage':stage,'samples':t,'public_F32_output_bitwise_exact':True,'counts_sha256':digest(counts),'scaled_F64_frequency_error':scaled,'descriptive_2e_5_flag':bool(scaled<2e-5),'max_error_over_derived_bound':float(np.max(error/np.maximum(bound,1e-30))),'observed_node_updates':4*t*n};self.rows.append(row);return {'status':'ok','scaled_frequency_error':scaled,'bitwise_exact':True}
    def finish(self):
        phases=['isolated'] if self.phase=='validation' else ['uniform','allocated'];expected={(c,s,k) for c in phases for s in range(5) for k in range(24)};assert len(self.rows)==len(expected) and {(v['condition'],v['seed'],v['stage']) for v in self.rows}==expected
        r={'passed':True,'phase':self.phase,'native_operators':len(self.rows),'all_public_F32_outputs_bitwise_exact':True,'all_IEEE_forward_error_envelopes_checked':True,'descriptive_2e_5_failures':sum(not r['descriptive_2e_5_flag'] for r in self.rows),'observed_node_updates_replayed':sum(v['observed_node_updates'] for v in self.rows),'source_sha256':sha(__file__),'rows':self.rows,'scope':'Predeclared exact public THRML/defaultF32/dynamicfield replay with independent integer counts and F64 linear identity/IEEE bound. The2e-5 flag is descriptive and failures are retained; no posthoc tolerance adjustment. Raw counts from same-key verification represented by SHA, not new quality evidence. No physical hardware claim.'};atomic_json(OUT/f'{self.phase}_native_audit.json',r);return {'status':'ok','operators':len(self.rows)}

def main():
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled;w=Worker();print(json.dumps({'status':'ready'}),flush=True)
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='quit':break
            print(json.dumps(w.finish() if r.get('action')=='finish' else w.execute(r)),flush=True)
        except Exception as e:traceback.print_exc(file=sys.stderr);print(json.dumps({'status':'error','error':str(e)}),flush=True)
if __name__=='__main__':main()
