"""Full five-model, ten-MC-seed native two-layer Boolean classifier cohort."""
import argparse,fcntl,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from .boolean_classifier_common import OUT,PREP,CHUNK,ROOT_KEY,sha,sources,guard,encode,fields,keys,cases,paths
from .fine_range_common import atomic_json,atomic_npz
CRITICAL=['boolean_classifier_common.py','boolean_classifier_study.py','boolean_classifier_native.py','verify_boolean_classifier_study.py','boolean_classifier_statistics.py','verify_boolean_classifier_statistics.py','finalize_boolean_classifier.py','boolean_classifier_pipeline.py','native.py','controlled_positive_native.py','fine_range_common.py','prepare_boolean_classifier.py']
def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    a=json.loads(PREP.read_text());assert json.loads(Path('results/thrml_spw/boolean_classifier_native_witness/evidence_audit.json').read_text())['passed']
    assert json.loads(Path('results/thrml_spw/boolean_classifier_gpu_preflight/evidence_audit.json').read_text())['passed']
    p={'stage':'prospective_full_classifier_protocol','preparation_sha256':sha(PREP),'samples':{'R1':[98,392,1568],'R3':[74,296,1184]},'critical_sources':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in CRITICAL},'all_research_sources_at_seal':sources(),'official_thrml_commit':subprocess.check_output(['git','-C','vendor/thrml','rev-parse','HEAD'],text=True).strip(),'key_root':ROOT_KEY,'chunk':CHUNK,'minimum_free_bytes':a['resources']['minimum_free_before_production'],'model_seeds':list(range(5)),'MC_seeds':list(range(10)),'bank_size':8,'bank_count':1250,'test_examples':10000,'input_range':'shared validation minmax','hidden_range':'per-model validation-only max with25% margin','readout':'F64 before ReLU; deterministic F32 stored weights/bias; R3 counts all AND terms and weight-column popcounts','count_storage':{'individual_weight':'uint16','AND_and_column':'uint32'},'statistics':'Report every budget and both regimes, five model-level units with ten MC seeds each. Paired R3-R1/model-baseline differences; descriptive hierarchical bootstrap5000 draws resamples5 model IDs and10 within-model seed IDs, all10000 already-consumed test images fixed. Report accuracy and mean CE; no posthoc quality selection.','quality_criterion':'Descriptive comparison only; no new confirmatory pass/fail threshold on previously consumed test examples. Original gates unchanged.','scope':'Full fixed small positive-weight classifier only. No CNN/Transformer-generalization, fresh held-out confirmation, physical AND/hardware energy or optimized arithmetic claim.'}
    assert p['official_thrml_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9';guard(p);atomic_json(OUT/'protocol.json',p);print('Sealed full classifier protocol; no new task scoring',flush=True)
def run():
    import jax
    import jax.numpy as jnp
    from importlib.metadata import version
    from .boolean_classifier_native import CountLayer,readout_boolean,readout_weight
    assert jax.default_backend()=='gpu' and not jax.config.x64_enabled
    p=json.loads((OUT/'protocol.json').read_text());a=guard(p)
    assert os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0'
    OUT.joinpath('cases').mkdir(exist_ok=True)
    with (OUT/'operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        mp=OUT/'manifest.json'
        if mp.exists():
            m=json.loads(mp.read_text());assert m['protocol_sha256']==sha(OUT/'protocol.json');assert not m['complete']
        else:
            assert shutil.disk_usage('.').free>=p['minimum_free_bytes'],'Insufficient persistent capacity before any new task scoring/sampling'
            m={'complete':False,'independently_audited':False,'protocol_sha256':sha(OUT/'protocol.json'),'backend':jax.default_backend(),'packages':{n:version(n) for n in ['thrml','jax','jaxlib','numpy']},'XLA_FLAGS':os.environ.get('XLA_FLAGS'),'rows':[],'native_updates':0}
            atomic_json(mp,m)
        data=np.load(a['data_path']);x=data['test_inputs'].astype(float);xb=x.reshape(1250,8,64)
        low=a['input_lower'];span=a['input_span'];xnorm=(xb-low)/span;xc=encode(np.clip(xnorm,0,1))
        start=time.monotonic();done={r['name']:r for r in m['rows']}
        layers={r:[CountLayer(64,32,8,r=='R3'),CountLayer(32,10,8,r=='R3')] for r in ['R1','R3']}
        for c in cases(p):
            if c['name'] in done:
                assert sha(paths(c))==done[c['name']]['sha256'];continue
            guard(p)
            # Bound one-case write without assuming compression; previous cases remain durable if capacity is lost externally.
            case_bytes=7_000_000 if c['regime']=='R1' else 3_500_000
            assert shutil.disk_usage('.').free>case_bytes+100_000_000,'Capacity fell during cohort; completed cases retained for explicit same-protocol continuation'
            model=np.load(a['models'][c['model']]['controls_path']);bias=[model[f'bias_{i}'] for i in range(2)];wf=[model[f'weight_field_{i}'].ravel() for i in range(2)]
            t=c['T'];r=c['regime'];boolean=r=='R3';ri=int(boolean);logits=np.empty((1250,8,10),float);clip=0
            if boolean:
                raw={'and_0':np.empty((1250,8,32),np.uint32),'column_0':np.empty((1250,32),np.uint32),'and_1':np.empty((1250,8,10),np.uint32),'column_1':np.empty((1250,10),np.uint32),'hidden_codes':np.empty((1250,8,32),np.uint8)}
            else:raw={'weight_0':np.empty((1250,64,32),np.uint16),'weight_1':np.empty((1250,32,10),np.uint16)}
            for first in range(0,1250,CHUNK):
                n=min(CHUNK,1250-first);h=xb[first:first+n];hlow=float(model['hidden_lower']);hspan=float(model['hidden_span'])
                for layer,(ni,no) in enumerate([(64,32),(32,10)]):
                    if boolean:
                        ac=xc[first:first+n] if layer==0 else encode(np.clip((h-hlow)/hspan,0,1))
                        if layer==1:raw['hidden_codes'][first:first+n]=ac
                        f=np.concatenate([np.broadcast_to(wf[layer],(n,ni*no)),fields(ac).reshape(n,-1)],axis=1)
                    else:f=np.broadcast_to(wf[layer],(n,ni*no)).copy()
                    result=layers[r][layer].kernel(t)(keys(ri,c['model'],c['seed'],t,layer,first,n),jnp.asarray(f))
                    if boolean:
                        aa,cc=map(np.asarray,result);assert aa.min()>=0 and aa.max()<=ni*t and cc.min()>=0 and cc.max()<=ni*t
                        raw[f'and_{layer}'][first:first+n]=aa;raw[f'column_{layer}'][first:first+n]=cc
                        y=readout_boolean(aa,cc[:,None,:],t,low if layer==0 else hlow,span if layer==0 else hspan,bias[layer])
                    else:
                        counts=np.asarray(result).reshape(n,ni,no);assert counts.min()>=0 and counts.max()<=t
                        raw[f'weight_{layer}'][first:first+n]=counts;y=readout_weight(counts,t,h,bias[layer])
                    h=np.maximum(y,0) if layer==0 else y
                    if boolean and layer==0:clip+=int(np.count_nonzero((h<hlow)|(h>hlow+hspan)))
                logits[first:first+n]=h
            assert np.isfinite(logits).all();atomic_npz(paths(c),**raw,logits=logits.reshape(10000,10))
            updates=1250*t*(3136 if boolean else 2368)
            row={**c,'path':str(paths(c)),'sha256':sha(paths(c)),'native_updates':updates,'hidden_clipped_coordinates':clip,'input_outside_requested_range':int(np.count_nonzero((xnorm<0)|(xnorm>1)))}
            m['rows'].append(row);m['native_updates']+=updates;atomic_json(mp,m)
            print(json.dumps({'completed_cases':len(m['rows']),'last':c['name'],'native_updates':m['native_updates'],'elapsed_seconds':time.monotonic()-start}),flush=True)
        assert len(m['rows'])==300 and m['native_updates']==609168000000;guard(p);m['complete']=True;atomic_json(mp,m);print('Full classifier production complete; independent audit pending',flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args();prepare() if args.action=='prepare' else run()
if __name__=='__main__':main()
