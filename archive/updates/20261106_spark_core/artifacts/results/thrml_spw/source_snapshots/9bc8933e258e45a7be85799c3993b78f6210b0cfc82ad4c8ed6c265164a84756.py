"""Bounded validation witness, separate public raw states and integer identity."""
from pathlib import Path
import json,hashlib,time,datetime,itertools
import numpy as np
import jax
import jax.numpy as jnp
from results.thrml_spw.integer_activation_native.gated_layer import IntegerGateLayer,activation_codes,readout
from research.thrml_spw.verify_boolean_classifier_study import sampler
OUT=Path('results/thrml_spw/integer_activation_native/witness')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert jax.default_backend()=='cpu' and not jax.config.x64_enabled;OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    prep=json.loads(Path('results/thrml_spw/boolean_classifier_preparation_v2/preparation.json').read_text());model=np.load(prep['models'][0]['controls_path']);x=np.load(prep['data_path'])['validation_inputs'][:16].astype(float).reshape(2,8,64);hidden=np.maximum(x@model['weight_0'].astype(float)+model['bias_0'],0)
    paths=['results/thrml_spw/integer_activation_native/gated_layer.py','results/thrml_spw/integer_activation_native/validate.py','research/thrml_spw/native.py','research/thrml_spw/controlled_positive_native.py','research/thrml_spw/verify_boolean_classifier_study.py'];sources={p:sha(p) for p in paths};protocol={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,'model_control_sha256':sha(prep['models'][0]['controls_path']),'data_sha256':sha(prep['data_path']),'sampling_budgets':[1,7,98,392,1568],'key_root':1437520,'chains':2,'validation_inputs':16,'scope':'Single linear layers with fixed validation inputs/hidden states; integer activation gates and output counters. No full classifier score or physical circuit claim; no original cohort source change.'};(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    updates=0;elements=0;rows=[]
    for layer,(ni,no,xx) in enumerate([(64,32,x),(32,10,hidden)]):
        low=float(model['input_lower'] if layer==0 else model['hidden_lower']);span=float(model['input_span'] if layer==0 else model['hidden_span']);codes=activation_codes(xx,low,span);bank=IntegerGateLayer(ni,no,8);field=np.broadcast_to(model[f'weight_field_{layer}'].ravel(),(2,ni*no)).copy();assert len(bank.bank.free_indices)==ni*no and len(bank.bank.clamped_indices)==0
        for t in protocol['sampling_budgets']:
            root=jax.random.fold_in(jax.random.fold_in(jax.random.key(1437520),layer),t);keys=jax.random.split(root,2);gated,column=map(np.asarray,bank.kernel(t)(keys,jnp.asarray(field),jnp.asarray(codes)));states=np.asarray(sampler(ni*no,t)(keys,jnp.asarray(field))).reshape(2,t,ni,no);weight_counts=states.sum(1,dtype=np.uint32);reference=np.einsum('cbi,cio->cbo',codes.astype(np.uint32),weight_counts,dtype=np.uint32,optimize=False);columns=weight_counts.sum(1,dtype=np.uint32);np.testing.assert_array_equal(gated,reference);np.testing.assert_array_equal(column,columns);assert gated.dtype==column.dtype==np.int32 and gated.min()>=0 and gated.max()<=255*ni*t
            y=readout(gated,column[:,None,:],t,low,span,model[f'bias_{layer}']);xr=low+span*codes.astype(float)/255;deterministic=xr@(weight_counts.astype(float)/t)+model[f'bias_{layer}'];np.testing.assert_allclose(y,deterministic,rtol=2e-12,atol=2e-11)
            np.savez_compressed(OUT/f'layer{layer}_T{t}.npz',fields=field,activation_codes=codes,gated_counts=gated,column_counts=column,public_weight_counts=weight_counts,readout=y);u=2*t*ni*no;updates+=u;elements+=gated.size+column.size;rows.append({'layer':layer,'T':t,'weight_nodes':ni*no,'activation_pbits':0,'counter_entries_exact':gated.size+column.size,'gated_counter_max_observed':int(gated.max()),'gated_counter_bound':255*ni*t,'native_updates':u,'raw_public_updates':u});print(json.dumps(rows[-1]),flush=True)
    # Maximal deterministic gates show counter width independently of any favorable observed stream.
    assert 255*64*1568==25589760 and (255*64*1568).bit_length()==25
    for path,h in sources.items():assert sha(path)==h
    assert updates==9784576
    result={'passed':True,'cases':len(rows),'integer_counter_entries_publicly_reproduced_exactly':elements,'native_observer_updates':updates,'independent_public_raw_updates':updates,'all_weight_nodes_free':True,'activation_pbits':0,'maximum_gated_counter_bound':25589760,'largest_minimum_counter_width':25,'rows':rows,'source_sha256':sources,'scope':'Native integer-gate operator implementation validated on both planned layer shapes and largest current weight-sampling budget. No full stochastic classifier accuracy, physical bit-serial rate, memory or energy advantage established.'};(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','source_sha256']},indent=2))
if __name__=='__main__':main()
