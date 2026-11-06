"""Source-seal then execute all150matched-update positional CNN cases."""
import argparse,json,os,time,shutil,fcntl,datetime
from pathlib import Path
import numpy as np
from results.thrml_spw.fashion_cnn_native.word_comparison.common import ROOT,OUT,PARENT,TRAIN,sha,write,save,guard,parent_closed,keys
def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists();witness=ROOT/'word_witness';assert json.loads((witness/'evidence_audit.json').read_text())['passed'];check=json.loads((ROOT/'word_mean_selfcheck.json').read_text());assert check['passed']
    for f,h in check['source_sha256'].items():assert sha(f)==h,f
    sources=dict(json.loads((witness/'protocol.json').read_text())['source_sha256']);sources.update({str(f):sha(f) for f in (ROOT/'word_comparison').glob('*.py')});train=json.loads((TRAIN/'manifest.json').read_text());files=[TRAIN/'manifest.json',TRAIN/'validation.npz',ROOT/'word_mean_selfcheck.json',witness/'protocol.json',witness/'evidence_audit.json',PARENT/'protocol.json',PARENT/'controls.json',PARENT/'test_data.npz',PARENT/'control_logits.npz']+[Path(r['path']) for r in train['rows']];parent_status=json.loads((PARENT/'pipeline.json').read_text())['status'];proc=Path('/proc/3792481')
    if proc.exists():assert proc.joinpath('stat').read_text().split()[21]=='45024359'
    else:assert parent_status=='full_native_CNN_independently_audited_goal_active';parent_closed()
    p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,'input_sha256':{str(f):sha(f) for f in files},'parent_protocol_sha256':sha(PARENT/'protocol.json'),'parent_pid':3792481,'parent_start_ticks':'45024359','K':4,'word_samples':[16,64,256],'single_samples':[64,256,1024],'models':5,'MC_seeds':10,'images':10000,'bank_size':8,'banks':1250,'chunk':4,'cases':150,'key_root':1577520,'bootstrap_word_seed':1587520,'GPU_validation_key_root':1597520,'bootstrap_draws':5000,'native_updates':223776000000,'public_updates':223776000000,'field_control':'Every positional bit free atfixed F32field±4; targetword codes nearest-even normalized minmax16levels, coefficients1/2/4/8 dividedby15. Same per-output centers/radii/ACT8ranges/padding/biases as parent. Internal comparator, not asserted Extropic hardware architecture.','controls':['parent_FP32_F64','parent_ACT8_INT4_minmax','nominal_word_mean'],'contrasts':['word_minus_single','word_minus_FP32','word_minus_INT4_minmax','word_minus_nominal_word_mean'],'statistics':'Paired five model IDs from parent5000bootstrap draws. Resample word MC seeds independently using a new fixed5000x5x10index array; parent MC index array retained for single method. Fixed historicallyconsumedtest images, descriptive sensitivity only, not common random weight draws or fresh confirmation. All3budgets/all4contrasts retained, no outcome-selected gate.','minimum_free_before_stages':450000000,'minimum_free_before_case':50000000,'scope':'Full small signed CNN internal4nodepositional versus1node at equal scheduled evaluations;4times nodes,1/4sample rounds, different field control/counterwidths and word quantization. Software externalF64readout and integer gate arithmetic; no realdevice energy, latency, area or totalmemory advantage inferred.'};guard(p);write(OUT/'protocol.json',p);print('Matched-update word CNN protocol source-sealed',flush=True)
def run():
    import jax
    from importlib.metadata import version
    from results.thrml_spw.fashion_cnn_native.word_reference.native import encode,forward
    from results.thrml_spw.fashion_cnn_native.compact.receipts import receipt
    p=json.loads((OUT/'protocol.json').read_text());guard(p);parent_closed();assert json.loads((OUT/'GPU_preflight/audit.json').read_text())['passed'];assert jax.default_backend()=='gpu' and not jax.config.x64_enabled and os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0';OUT.joinpath('cases').mkdir(exist_ok=True)
    with (OUT/'operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);mp=OUT/'manifest.json'
        if mp.exists():m=json.loads(mp.read_text());assert not m['complete'] and m['protocol_sha256']==sha(OUT/'protocol.json')
        else:m={'complete':False,'independently_audited':False,'protocol_sha256':sha(OUT/'protocol.json'),'parent_completion_sha256':sha(PARENT/'completion.json'),'parent_test_data_sha256':sha(PARENT/'test_data.npz'),'rows':[],'native_updates':0,'backend':jax.default_backend(),'packages':{n:version(n) for n in ['thrml','jax','jaxlib','numpy']}};write(mp,m)
        x=np.load(PARENT/'test_data.npz')['inputs'].astype(float).reshape(1250,8,1,14,14);train=json.loads((TRAIN/'manifest.json').read_text());done={r['name']:r for r in m['rows']};start=time.monotonic()
        for mi in range(5):
            model=np.load(train['rows'][mi]['path']);enc=encode(model)
            for t in p['word_samples']:
                for seed in range(10):
                    name=f'word_model{mi}_T{t}_seed{seed}';file=OUT/'cases'/(name+'.npz')
                    if name in done:assert sha(file)==done[name]['sha256'];continue
                    guard(p);assert shutil.disk_usage('.').free>=p['minimum_free_before_case'];hashes=np.empty((1250,14,32),np.uint8);logits=np.empty((10000,10),float);clip=np.zeros(3,np.int64)
                    for first in range(0,1250,4):
                        n=min(4,1250-first);kk=[keys(mi,seed,t,li,first,n) for li in range(3)];raw=forward(x[first:first+n],model,enc,t,kk);hashes[first:first+n]=receipt(raw);logits[first*8:(first+n)*8]=raw['output_2'].reshape(n*8,10)
                        for li,h in enumerate([x[first:first+n],raw['output_0'],raw['output_1']]):clip[li]+=np.count_nonzero((h<0)|(h>float(model['activation_span'][li])))
                    assert np.isfinite(logits).all();save(file,receipts=hashes,logits=logits,clipped_coordinates=clip);u=1250*t*10656;m['native_updates']+=u;m['rows'].append({'name':name,'model':mi,'seed':seed,'T':t,'matched_single_T':4*t,'path':str(file),'sha256':sha(file),'native_updates':u,'clipped_coordinates':clip.tolist()});write(mp,m);print(json.dumps({'cases':len(m['rows']),'last':name,'native_updates':m['native_updates'],'elapsed_seconds':time.monotonic()-start}),flush=True)
        assert len(m['rows'])==150 and m['native_updates']==223776000000;guard(p);m['complete']=True;write(mp,m)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args();prepare() if args.action=='prepare' else run()
if __name__=='__main__':main()
