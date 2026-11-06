"""Actual two-layer GPU integer-gate preflight, paired with audited parent counts."""
import json,os,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from results.thrml_spw.integer_activation_native.gated_layer import IntegerGateLayer,activation_codes,readout
from results.thrml_spw.integer_activation_native.cohort_common import ROOT,PARENT,PREP,sha,atomic_json,atomic_npz,parent_rows
from research.thrml_spw.boolean_classifier_common import keys
OUT=ROOT/'gpu_preflight'
def main():
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled;assert os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0';OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    prep=json.loads(PREP.read_text());control=np.load(prep['models'][0]['controls_path']);x=np.load(prep['data_path'])['test_inputs'][:128].astype(float).reshape(16,8,64);audit=json.loads((PARENT/'native_audit_progress.json').read_text());verified={r['name']:r for r in audit['rows']};selected=[r for r in parent_rows() if r['model']==0 and r['seed']==0];assert len(selected)==3
    for r in selected:assert r['name'] in verified and verified[r['name']]['raw_sha256']==r['sha256'] and sha(r['path'])==r['sha256']
    files=[ROOT/'gated_layer.py',ROOT/'gpu_preflight.py',ROOT/'cohort_common.py',Path('research/thrml_spw/boolean_classifier_common.py'),Path('research/thrml_spw/native.py'),Path('research/thrml_spw/controlled_positive_native.py')];sources={str(f):sha(f) for f in files};atomic_json(OUT/'protocol.json',{'source_sha256':sources,'parent_protocol_sha256':sha(PARENT/'protocol.json'),'preparation_sha256':sha(PREP),'inputs':'128 already-consumed test inputs, no labels/quality scores','selection':'model0/seed0/all three frozen budgets/first16 banks; these parent cases already publicly replayed','T':[98,392,1568],'same_parent_keys':True,'XLA_FLAGS':os.environ.get('XLA_FLAGS')})
    layers=[IntegerGateLayer(64,32,8),IntegerGateLayer(32,10,8)];updates=0;entries=0;rows=[]
    for row in selected:
        t=row['T'];h=x.copy();ref=x.copy();raw=np.load(row['path']);data={};start=time.monotonic()
        for layer,(ni,no) in enumerate([(64,32),(32,10)]):
            lo=float(control['input_lower'] if layer==0 else control['hidden_lower']);span=float(control['input_span'] if layer==0 else control['hidden_span']);a=activation_codes(h,lo,span);ar=np.rint(255*np.clip((ref-lo)/span,0,1)).astype(np.uint8);np.testing.assert_array_equal(a,ar);field=np.broadcast_to(control[f'weight_field_{layer}'].ravel(),(16,ni*no)).copy();g,c=map(np.asarray,layers[layer].kernel(t)(keys(0,0,0,t,layer,0,16),jnp.asarray(field),jnp.asarray(a)));wc=raw[f'weight_{layer}'][:16].astype(np.uint32);rg=np.einsum('cbi,cio->cbo',ar.astype(np.uint32),wc,dtype=np.uint32,optimize=False);rc=wc.sum(1,dtype=np.uint32);np.testing.assert_array_equal(g,rg);np.testing.assert_array_equal(c,rc);y=readout(g,c[:,None,:],t,lo,span,control[f'bias_{layer}']);ry=lo*rc[:,None,:].astype(float)/t+(span/255)*rg.astype(float)/t+control[f'bias_{layer}'];np.testing.assert_array_equal(y,ry);h=np.maximum(y,0) if layer==0 else y;ref=np.maximum(ry,0) if layer==0 else ry;data[f'gated_{layer}']=g;data[f'column_{layer}']=c;data[f'activation_codes_{layer}']=a;updates+=16*t*ni*no;entries+=g.size+c.size
        data['logits']=h;atomic_npz(OUT/f'T{t}.npz',**data);rows.append({'T':t,'native_counters_and_full_causal_logits_exact':True,'elapsed_seconds':time.monotonic()-start});print(json.dumps(rows[-1]),flush=True)
    for f,h in sources.items():assert sha(f)==h
    assert updates==77973504;atomic_json(OUT/'evidence_audit.json',{'passed':True,'cases':3,'same_trajectory_parent_public_count_reference':True,'integer_counter_entries_exact':entries,'native_observer_updates':updates,'new_public_raw_replay_updates':0,'source_sha256':sources,'rows':rows,'scope':'Both actual causal layers at maximum intended16-bank chunk, all3budgets, paired to previously public-replayed native weight counts. No task-quality score; same parent/production trajectories are not independent replications.'})
if __name__=='__main__':main()
