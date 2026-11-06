"""Full native output-counter inference with fixed-point8-bit activations."""
import argparse,json,os,fcntl,shutil,time,subprocess
from pathlib import Path
import numpy as np
from results.thrml_spw.integer_activation_native.cohort_common import OUT,ROOT,PARENT,PREP,sha,guard,parent_closed,parent_rows,case_path,atomic_json,atomic_npz

def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    assert json.loads((ROOT/'witness/evidence_audit.json').read_text())['passed'] and json.loads((ROOT/'gpu_preflight/evidence_audit.json').read_text())['passed']
    parent=json.loads((PARENT/'protocol.json').read_text());prep=json.loads(PREP.read_text());sources={str(f):sha(f) for f in sorted(ROOT.glob('*.py'))};sources.update(parent['critical_sources']);rows=parent_rows();assert len(rows)==150
    p={'purpose':'Native direct integer-activation output-counter inference, no per-weight operational reconstruction or activation p-bits.','source_sha256':sources,'preparation_sha256':sha(PREP),'parent_protocol_sha256':sha(PARENT/'protocol.json'),'parent_manifest_sha256':sha(PARENT/'manifest.json'),'parent_R1_cases':rows,'samples':[98,392,1568],'models':5,'MC_seeds':10,'test_images':10000,'bank_size':8,'banks_per_case':1250,'chunk':16,'activation_bits':8,'activation_ranges':'Original validation-only input minmax/per-model hidden max*1.25, no reselection. Nearest-even uint8 codes after clipping.','field_control':'Same original finite8-bit fields±4, all2368 weight nodes free.','same_parent_weight_keys':True,'key_source':'research/thrml_spw/boolean_classifier_common.py keys(regime=0,model,seed,T,layer,bank)','production_updates':304584000000,'parent_pipeline_pid':3707781,'parent_pipeline_start_ticks':Path('/proc/3707781/stat').read_text().split()[21],'counter_storage':'uint32 gated and column counts, uint8 hidden activation codes, F64 logits; native observer int32','minimum_free_before_production':701500000,'analysis':'All150 cases. Compare with parentR1 at identical T/weight trajectories and parentR3 at equal scheduled active updates. Reuse parent5000 model/MC bootstrap index draws, five model units, fixed consumed images; no new independent replications or confirmatory quality gate. All three budgets reported.','controls':['ACT8_FP32_weights','ACT8_INT2_weights','ACT8_INT4_weights','ACT8_INT8_weights','ACT8_programmed_weight_mean'],'control_scope':'Quantized activations with dequantized software weights; programmed-mean plug-in is not E[stochastic nonlinear NN]. No optimized hardware kernel/energy claim.','audit':'Every output counter and derived hidden code checked using already public-replayed same-key parentR1 weight counts, complete causal output reconstruction. No new raw-state sampling needed for that exact sufficient-statistic reference; it is not a fresh trajectory replication. Bounded separate public-state operator witness and full GPU paired preflight retained.','scope':'Full fixed small positive-weight network on previously consumed test data, deterministic activation quantization and sampled weights. Integer gating uses wider values than one-bit AND; sampler/control and final F64 readout still use floating arithmetic. No CNN/Transformer/generalization/physical systems advantage asserted.','official_thrml_commit':subprocess.check_output(['git','-C','vendor/thrml','rev-parse','HEAD'],text=True).strip()}
    assert p['official_thrml_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9';guard(p);atomic_json(OUT/'protocol.json',p);print('Integer8 full cohort source sealed; no labels/quality evaluated',flush=True)

def run():
    import jax
    import jax.numpy as jnp
    from importlib.metadata import version
    from results.thrml_spw.integer_activation_native.gated_layer import IntegerGateLayer,activation_codes,readout
    from research.thrml_spw.boolean_classifier_common import keys
    p=json.loads((OUT/'protocol.json').read_text());prep=guard(p);parent_closed();assert sha(PARENT/'manifest.json')==p['parent_manifest_sha256'];assert jax.default_backend()=='gpu' and not jax.config.x64_enabled;assert os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0'
    (OUT/'cases').mkdir(exist_ok=True)
    with (OUT/'operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);mp=OUT/'manifest.json'
        if mp.exists():m=json.loads(mp.read_text());assert not m['complete'] and m['protocol_sha256']==sha(OUT/'protocol.json')
        else:
            assert shutil.disk_usage('.').free>=p['minimum_free_before_production'];m={'complete':False,'independently_audited':False,'protocol_sha256':sha(OUT/'protocol.json'),'packages':{n:version(n) for n in ['thrml','jax','jaxlib','numpy']},'backend':jax.default_backend(),'XLA_FLAGS':os.environ.get('XLA_FLAGS'),'rows':[],'native_updates':0};atomic_json(mp,m)
        x=np.load(prep['data_path'])['test_inputs'].astype(float).reshape(1250,8,64);lo=prep['input_lower'];sp=prep['input_span'];input_codes=activation_codes(x,lo,sp);layers=[IntegerGateLayer(64,32,8),IntegerGateLayer(32,10,8)];done={r['name']:r for r in m['rows']};start=time.monotonic()
        for parent in p['parent_R1_cases']:
            name=parent['name'].replace('R1_','integer8_',1);path=case_path(parent)
            if name in done:assert sha(path)==done[name]['sha256'];continue
            guard(p);assert sha(parent['path'])==parent['sha256'];assert shutil.disk_usage('.').free>103500000,'Capacity guard before new case; earlier cases retained'
            mi,seed,t=parent['model'],parent['seed'],parent['T'];model=np.load(prep['models'][mi]['controls_path']);data={f'{kind}_{l}':np.empty((1250,8,no) if kind=='gated' else (1250,no),np.uint32) for l,no in [(0,32),(1,10)] for kind in ['gated','column']};data['hidden_codes']=np.empty((1250,8,32),np.uint8);logits=np.empty((1250,8,10),float);clipped=0
            for first in range(0,1250,16):
                n=min(16,1250-first);h=x[first:first+n]
                for layer,(ni,no) in enumerate([(64,32),(32,10)]):
                    lower=float(model['input_lower'] if layer==0 else model['hidden_lower']);span=float(model['input_span'] if layer==0 else model['hidden_span']);a=input_codes[first:first+n] if layer==0 else activation_codes(h,lower,span)
                    if layer==1:data['hidden_codes'][first:first+n]=a
                    field=np.broadcast_to(model[f'weight_field_{layer}'].ravel(),(n,ni*no)).copy();g,c=map(np.asarray,layers[layer].kernel(t)(keys(0,mi,seed,t,layer,first,n),jnp.asarray(field),jnp.asarray(a)));assert g.dtype==c.dtype==np.int32 and g.min()>=0 and g.max()<=255*ni*t and c.min()>=0 and c.max()<=ni*t;data[f'gated_{layer}'][first:first+n]=g;data[f'column_{layer}'][first:first+n]=c;y=readout(g,c[:,None,:],t,lower,span,model[f'bias_{layer}']);h=np.maximum(y,0) if layer==0 else y
                    if layer==0:clipped+=int(np.count_nonzero((h<float(model['hidden_lower']))|(h>float(model['hidden_lower']+model['hidden_span']))))
                logits[first:first+n]=h
            assert np.isfinite(logits).all();atomic_npz(path,**data,logits=logits.reshape(10000,10));updates=1250*t*2368;m['rows'].append({'name':name,'model':mi,'seed':seed,'T':t,'path':str(path),'sha256':sha(path),'parent_name':parent['name'],'parent_path':parent['path'],'parent_sha256':parent['sha256'],'native_updates':updates,'hidden_clipped_coordinates':clipped,'input_outside_affine_range':int(np.count_nonzero(((x-lo)/sp<0)|((x-lo)/sp>1)))});m['native_updates']+=updates;atomic_json(mp,m);print(json.dumps({'completed_cases':len(m['rows']),'last':name,'native_updates':m['native_updates'],'elapsed_seconds':time.monotonic()-start}),flush=True)
        assert len(m['rows'])==150 and m['native_updates']==304584000000;guard(p);m['complete']=True;atomic_json(mp,m);print('Integer8 full native production complete; independent paired audit pending',flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args();prepare() if a.action=='prepare' else run()
if __name__=='__main__':main()
