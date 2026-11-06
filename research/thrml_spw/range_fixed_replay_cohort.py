"""Prospective fixed-compilation operator check; no original evidence is replaced."""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from thrml import FactorSamplingProgram,SamplingSchedule,sample_with_observation
from thrml.models.discrete_ebm import SpinGibbsConditional,SpinEBMFactor
from .range_localization_worker import LocalizedLinear
from .verify_range_localization_native import SumOnly,PublicLayer
from .block_cached_control_worker import atomic_json,atomic_npy,sha
ROOT=Path('results/thrml_spw/range_localization')
OUT=ROOT/'fixed_compilation'
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def public_function(layer,span,lower,bias,shape,t):
    def run(key,x,fields):
        program=FactorSamplingProgram(layer.spec,[SpinGibbsConditional()],[SpinEBMFactor(layer.free,fields)],[]);obs=SumOnly(layer.readout,x,span,shape);total,_=sample_with_observation(key,program,SamplingSchedule(1,t,1),layer.initial,[],obs.init(),obs);return total/t+x@lower+bias
    return jax.jit(run)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicate',choices=[0,1],type=int,required=True);args=parser.parse_args();assert os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0' and jax.default_backend()=='gpu' and not jax.config.x64_enabled
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['worker_sha256']==sha(__file__);folder=OUT/f'replicate{args.replicate}';folder.mkdir(exist_ok=True);assert not (folder/'manifest.json').exists();p=json.loads((ROOT/'protocol.json').read_text());old=json.loads((ROOT/'native_evidence_audit.json').read_text());prior={(r['condition'],r['stage']):r for r in old['rows'] if r['seed']==0};rows=[];public_layers={}
    for stage in range(24):
        layers={};x=np.load(ROOT/f'anchors/stage{stage}_inputs.npy');target=np.load(ROOT/f'anchors/stage{stage}_target.npy')
        for cond in p['native_conditions']:
            g,t=cond['group'],cond['samples'];c=p['control_artifacts'][str(g)][stage];assert sha(c['path'])==c['sha256']
            if g not in layers:layers[g]=LocalizedLinear(c['path'],g)
            layer=layers[g];shape=layer.shape;n=layer.bank.logical_pbits;key=jax.random.fold_in(jax.random.fold_in(jax.random.key(1200730),0),stage);xx=jnp.asarray(x);fields=jnp.asarray(layer.nominal)
            y=np.asarray(layer.forward(t)(key,xx,fields));yp=folder/f'{cond["name"]}_stage{stage}.npy';atomic_npy(yp,y)
            if shape not in public_layers:public_layers[shape]=PublicLayer(np.full(shape,.5))
            pub=public_layers[shape];fn=public_function(pub,jnp.asarray(layer.span),jnp.asarray(layer.lower),layer.bias,shape,t);replay=np.asarray(fn(key,xx,fields));exact=bool(np.array_equal(y,replay))
            if not exact:atomic_npy(yp.with_name(yp.stem+'_public.npy'),replay)
            q=np.asarray(pub.frequencies(t)(key[None],fields))[0];counts=np.rint(q.astype(float)*t).astype(np.int32);count_hash=digest(counts.reshape(-1));assert count_hash==prior[cond['name'],stage]['integer_counts_sha256']
            reference=x.astype(float)@(layer.lower.astype(float)+layer.span.astype(float)*counts/t)+np.asarray(layer.bias);saved=np.load(ROOT/'probes'/f'{cond["name"]}_stage{stage}_seed0/output.npy');row={'stage':stage,'condition':cond['name'],'samples':t,'group':g,'output':str(yp),'output_sha256':sha(yp),'controls_sha256':c['sha256'],'inputs_sha256':sha(ROOT/f'anchors/stage{stage}_inputs.npy'),'target_sha256':sha(ROOT/f'anchors/stage{stage}_target.npy'),'fields_sha256':digest(layer.nominal),'public_sum_exact':exact,'max_public_sum_difference':float(np.max(np.abs(y.astype(float)-replay.astype(float)))),'public_integer_counts_match_prior':True,'integer_counts_sha256':count_hash,'max_F64_reference_error':float(np.max(np.abs(y-reference))),'target_MSE':float(np.mean((y.astype(float)-target)**2)),'old_target_MSE':float(np.mean((saved.astype(float)-target)**2)),'max_difference_from_original':float(np.max(np.abs(y.astype(float)-saved.astype(float)))),'native_node_updates':3*t*n};rows.append(row);atomic_json(yp.with_suffix('.json'),row);print(f'Fixed compilation replicate{args.replicate} stage{stage} {cond["name"]}: public exact={exact}',flush=True)
    assert len(rows)==72;commit=subprocess.check_output(['git','-C','vendor/thrml','rev-parse','HEAD'],text=True).strip();assert commit==protocol['thrml_commit'];r={'completed':True,'replicate':args.replicate,'protocol_sha256':sha(OUT/'protocol.json'),'worker_sha256':sha(__file__),'thrml_commit':commit,'jax':jax.__version__,'matmul_precision':str(jax.config.jax_default_matmul_precision),'XLA_FLAGS':os.environ['XLA_FLAGS'],'public_sum_all_exact':all(v['public_sum_exact'] for v in rows),'public_counts_all_match_original_audit':True,'native_node_updates':sum(v['native_node_updates'] for v in rows),'rows':rows,'scope':'Numerical reproducibility across all24 matrices and3 controls/sample conditions, seed0. No new heldout neural quality, changed original gates or hardware claim.'};atomic_json(folder/'manifest.json',r);print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
