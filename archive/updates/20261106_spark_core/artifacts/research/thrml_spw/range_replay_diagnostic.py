"""Prospective cold versus prefix-order diagnosis; original outputs stay immutable."""
import argparse,json,hashlib,os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
# Match the actual production module's imports and established kernel.
from .range_localization_worker import LocalizedLinear
from .calibrated_mnist_path_audit import PublicLayer
from .block_cached_control_worker import sha,atomic_json,atomic_npy
ROOT=Path('results/thrml_spw/range_localization')
OUT=ROOT/'reproducibility_diagnostic'
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def key(seed,stage):
    k=jax.random.key(1200730)
    for v in [seed,stage]:k=jax.random.fold_in(k,v)
    return k
def difference(a,b):return {'exact':bool(np.array_equal(a,b)),'differing_values':int(np.count_nonzero(a!=b)),'max_abs_difference':float(np.max(np.abs(a.astype(float)-b.astype(float))))}
def main():
    mode=argparse.ArgumentParser();mode.add_argument('--mode',choices=['cold','prefix'],required=True);args=mode.parse_args();assert jax.default_backend()=='gpu' and not jax.config.x64_enabled
    p=json.loads((ROOT/'protocol.json').read_text());folder=OUT/args.mode;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'result.json').exists();prefix=[];updates=0
    if args.mode=='prefix':
        x=np.load(ROOT/'anchors/stage0_inputs.npy');layers={}
        for cond in p['native_conditions']:
            g,t=cond['group'],cond['samples']
            if g not in layers:layers[g]=LocalizedLinear(p['control_artifacts'][str(g)][0]['path'],g)
            layer=layers[g];fn=layer.forward(t)
            for seed in range(5):
                y=np.asarray(fn(key(seed,0),jnp.asarray(x),jnp.asarray(layer.nominal)));old=np.load(ROOT/'probes'/f'{cond["name"]}_stage0_seed{seed}/output.npy');prefix.append({'condition':cond['name'],'seed':seed,**difference(y,old)});updates+=t*layer.bank.logical_pbits
    c=p['control_artifacts']['128'][1];layer=LocalizedLinear(c['path'],128);x=np.load(ROOT/'anchors/stage1_inputs.npy');old=np.load(ROOT/'probes/g128_T1024_stage1_seed0/output.npy');fn=layer.forward(1024);k=key(0,1);xx=jnp.asarray(x);fields=jnp.asarray(layer.nominal)
    y=np.asarray(fn(k,xx,fields));again=np.asarray(fn(k,xx,fields));updates+=2*1024*layer.bank.logical_pbits;atomic_npy(folder/'output.npy',y);atomic_npy(folder/'repeat.npy',again)
    counts=np.asarray(layer.counts(1024)(k,fields));public=PublicLayer(np.full(layer.shape,.5));freq=np.asarray(public.frequencies(1024)(k[None],fields))[0];pc=np.rint(freq.astype(float)*1024).reshape(-1).astype(np.int32);np.testing.assert_array_equal(counts,pc);updates+=2*1024*layer.bank.logical_pbits
    reference=x.astype(float)@(layer.lower.astype(float)+layer.span.astype(float)*counts.reshape(layer.shape)/1024)+np.asarray(layer.bias);target=np.load(ROOT/'anchors/stage1_target.npy');atomic_npy(folder/'reference.npy',reference)
    hlo=fn.lower(k,xx,fields).compiler_ir(dialect='hlo').as_hlo_text();(folder/'unoptimized.hlo').write_text(hlo)
    executable=fn.lower(k,xx,fields).compile();optimized=executable.as_text();(folder/'optimized.hlo').write_text(optimized)
    result={'mode':args.mode,'source_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'protocol.json'),'controls_sha256':sha(c['path']),'inputs_sha256':sha(ROOT/'anchors/stage1_inputs.npy'),'field_sha256':digest(layer.nominal),'key_data':np.asarray(jax.random.key_data(k)).tolist(),'jax':jax.__version__,'matmul_precision':str(jax.config.jax_default_matmul_precision),'x64':jax.config.x64_enabled,'XLA_FLAGS':os.environ.get('XLA_FLAGS',''),'prefix_cases':prefix,'preserved_output_difference':difference(y,old),'same_executable_repeat':difference(y,again),'production_public_integer_counts_exact':True,'integer_counts_sha256':digest(counts),'F64_reference_sha256':sha(folder/'reference.npy'),'observed_target_MSE':float(np.mean((y.astype(float)-target)**2)),'preserved_target_MSE':float(np.mean((old.astype(float)-target)**2)),'max_F64_reference_difference':float(np.max(np.abs(y-reference))),'native_observed_updates':updates,'scope':'Numerical diagnosis at one fixed matrix/seed, not new neural quality or changed original acceptance criteria.'};atomic_json(folder/'result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
