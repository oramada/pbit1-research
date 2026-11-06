"""Full CNN direct native production with compact recoverable evidence."""
import argparse,json,os,time,shutil,fcntl,datetime,subprocess
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import ROOT,OUT,TRAIN,PARENT,CONTROLS,sha,write,save,guard,parent_closed,key_bank
def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    for name in ['control_selfcheck.json','cohort_selfcheck.json']:assert json.loads((ROOT/name).read_text())['passed']
    parent=json.loads((PARENT/'protocol.json').read_text());sources=dict(parent['source_sha256']);sources.update({str(f):sha(f) for f in sorted((ROOT/'cohort').glob('*.py'))});train=json.loads((TRAIN/'manifest.json').read_text());trainingp=json.loads((TRAIN/'protocol.json').read_text());sources.update(trainingp['sources_sha256']);inputs={str(TRAIN/n):sha(TRAIN/n) for n in ['manifest.json','protocol.json','evidence_audit.json','validation.npz']};inputs.update(trainingp['dataset_raw_sha256'])
    for r in train['rows']:inputs[r['path']]=r['sha256']
    for name in ['control_selfcheck.json','cohort_selfcheck.json']:inputs[str(ROOT/name)]=sha(ROOT/name)
    proc=Path('/proc/3790833');stat=proc.joinpath('stat').read_text().split();assert stat[21]=='44969679'
    p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,'input_sha256':inputs,'parent_protocol_sha256':sha(PARENT/'protocol.json'),'parent_pid':3790833,'parent_start_ticks':stat[21],'models':5,'MC_seeds':10,'samples':[64,256,1024],'images':10000,'bank_size':8,'banks':1250,'chunk':4,'case_count':150,'key_root':1537520,'bootstrap_seed':1547520,'bootstrap_draws':5000,'weights':2664,'deterministic_biases':34,'native_updates':223776000000,'public_replay_updates':223776000000,'field_control':'Per-output-channel signed center/radius; normalized probabilities clipped to sigmoid(-8)/sigmoid(8), nearest uint8 field-grid commands±4, all weight nodes free. Endpoint probabilities approximated.','activation_control':'Deterministic nearest-even8-bit codes; validation-only ranges from frozen training exports; explicit padding masks. Integer temporal observer, external F64 signed-range readout/ReLU/maxpool.','controls':CONTROLS,'quantization':'minmax variants have2^b uniform endpoints; symmetric variants use qmax=2^(b-1)-1 with zero retained and unused negative endpoint code, per-output maxabs scale. Dequantized CPU arithmetic, not optimized integer hardware.','baseline':'FP32_F64 primary same stored weights/external F64; FP32_F32 separate conventional arithmetic reference. ACT8 nominal-programmed-mean is a nonlinear plug-in, not E[stochastic NN].','counter_receipts':'Every bank retains14 SHA256 name/dtype/shape/data digests; arrays must be independently regenerated via pinned public raw states and exact canonical readout. Full patch counters transient, not losslessly encoded in hashes. Final10000x10F64logits retained/case.','storage':{'case_receipts_bytes':560000,'case_logits_bytes':800000,'all_cases_raw_bytes':204000000,'full_patch_counter_bytes_not_retained':28309500000,'minimum_free_before_controls':450000000,'minimum_free_before_native':400000000,'minimum_free_before_case':50000000},'statistics':'All3budgets/all10baseline contrasts; accuracy and meanCE on fixed historically consumed10000images, five model units/ten paired MC seeds.5000 hierarchical model/seed bootstrap draws, no image resampling. Descriptive intervals, no new generalization confirmation or post-outcome gate.','control_tolerances':{'F64_rtol':2e-12,'F64_atol':2e-11,'F32_rtol':2e-5,'F32_atol':1e-4},'scope':'Full small signed FashionMNIST CNN with one free p-bit per normalized weight and range metadata. No literal negative probability, independent spatial weight draws, physical arithmetic/energy/latency/area or broader architectures claimed. All planned cases retained even when quality fails.'}
    assert subprocess.check_output(['git','-C','vendor/thrml','rev-parse','HEAD'],text=True).strip()=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9';assert not subprocess.check_output(['git','-C','vendor/thrml','status','--short'],text=True);guard(p);write(OUT/'protocol.json',p);print('Full CNN protocol source-sealed before test scoring',flush=True)
def run():
    import jax
    from importlib.metadata import version
    from results.thrml_spw.fashion_cnn_native.inference.native import encode
    from results.thrml_spw.fashion_cnn_native.inference.fast_native import forward
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    p=json.loads((OUT/'protocol.json').read_text());guard(p);parent_closed();assert jax.default_backend()=='gpu' and not jax.config.x64_enabled and os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0';controls=json.loads((OUT/'controls.json').read_text());assert controls['complete'] and controls['test_data_sha256']==sha(OUT/'test_data.npz') and controls['protocol_sha256']==sha(OUT/'protocol.json')
    OUT.joinpath('cases').mkdir(exist_ok=True)
    with (OUT/'operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=OUT/'manifest.json'
        if path.exists():m=json.loads(path.read_text());assert not m['complete'] and m['protocol_sha256']==sha(OUT/'protocol.json') and m['test_data_sha256']==sha(OUT/'test_data.npz')
        else:
            assert shutil.disk_usage('.').free>=p['storage']['minimum_free_before_native'];m={'complete':False,'independently_audited':False,'protocol_sha256':sha(OUT/'protocol.json'),'test_data_sha256':sha(OUT/'test_data.npz'),'controls_sha256':sha(OUT/'controls.json'),'backend':jax.default_backend(),'packages':{n:version(n) for n in ['thrml','jax','jaxlib','numpy']},'rows':[],'native_updates':0};write(path,m)
        x=np.load(OUT/'test_data.npz')['inputs'].astype(float).reshape(1250,8,1,14,14);train=json.loads((TRAIN/'manifest.json').read_text());done={r['name']:r for r in m['rows']};start=time.monotonic()
        for mi in range(5):
            model=np.load(train['rows'][mi]['path']);enc=encode(model)
            for t in p['samples']:
                for seed in range(10):
                    name=f'model{mi}_T{t}_seed{seed}';case=OUT/'cases'/(name+'.npz')
                    if name in done:assert sha(case)==done[name]['sha256'];continue
                    guard(p);assert shutil.disk_usage('.').free>=p['storage']['minimum_free_before_case'];hashes=np.empty((1250,14,32),np.uint8);logits=np.empty((10000,10),float);clipped=np.zeros(3,np.int64)
                    for first in range(0,1250,4):
                        n=min(4,1250-first);keys=[key_bank(mi,seed,t,li,first,n) for li in range(3)];raw=forward(x[first:first+n],model,enc,t,keys);hashes[first:first+n]=receipt(raw);logits[first*8:(first+n)*8]=raw['output_2'].reshape(n*8,10)
                        for li,h in enumerate([x[first:first+n],raw['output_0'],raw['output_1']]):clipped[li]+=np.count_nonzero((h<0)|(h>float(model['activation_span'][li])))
                    assert np.isfinite(logits).all();save(case,receipts=hashes,logits=logits,clipped_coordinates=clipped);u=1250*t*2664;m['native_updates']+=u;m['rows'].append({'name':name,'model':mi,'seed':seed,'T':t,'path':str(case),'sha256':sha(case),'native_updates':u,'clipped_coordinates':clipped.tolist()});write(path,m);print(json.dumps({'cases':len(m['rows']),'last':name,'native_updates':m['native_updates'],'elapsed_seconds':time.monotonic()-start}),flush=True)
        assert len(m['rows'])==150 and m['native_updates']==223776000000;guard(p);m['complete']=True;write(path,m)
def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args();prepare() if args.action=='prepare' else run()
if __name__=='__main__':main()
