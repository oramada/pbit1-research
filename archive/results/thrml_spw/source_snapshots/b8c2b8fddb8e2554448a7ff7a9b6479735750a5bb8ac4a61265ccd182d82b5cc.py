"""Frozen end-to-end student confirmation; capacity/compiler gates precede scoring."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from .fine_range_common import OUT,require_capacity,condition,sha,atomic_json,atomic_npy,atomic_npz

def preflight():
    require_capacity()
    p=json.loads((OUT/'protocol.json').read_text());reservation=json.loads((OUT/'reservation_audit.json').read_text());assert reservation['passed'] and reservation['protocol_sha256']==sha(OUT/'protocol.json') and reservation['tokens_sha256']==p['tokens_sha256']==sha(OUT/'tokens.npz')
    fixed=Path('results/thrml_spw/range_localization/fixed_compilation/evidence_audit.json');assert json.loads(fixed.read_text())['passed'],'Fixed-compiler all-case gate must pass before model scoring'
    for name in ['coarse','fine']:condition(p,name)
    for controls in p['control_artifacts'].values():
        for c in controls:assert sha(c['path'])==c['sha256']
    return p,fixed

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();p,fixed=preflight()
    if args.check:print(json.dumps({'preflight_passed':True,'model_scoring_started':False,'native_sampling_started':False}));return
    # Torch/model imports and construction deliberately follow both hard gates.
    from .sample_allocation_model import runner
    from .transformer_block_gpu import write_csv
    from .gpu_environment import cuda_environment
    from .packed_float_outputs import load_output
    from .fine_range_statistics import main as statistics
    names=['fine_range_confirmation.py','fine_range_worker.py','fine_range_common.py','fine_range_statistics.py','confirmation_statistics.py','block_cached_control_worker.py','sample_allocation_model.py','sequential_transformer.py','transformer_block_gpu.py','localized_native.py','localized_controls.py','calibrated_transformer_native.py','controlled_positive_native.py','native.py','grouped_range_native.py','packed_float_outputs.py','provenance.py'];sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in names};execution={'protocol_sha256':sha(OUT/'protocol.json'),'reservation_audit_sha256':sha(OUT/'reservation_audit.json'),'fixed_compiler_audit_sha256':sha(fixed),'source_sha256':sources,'XLA_FLAGS':'--xla_gpu_autotune_level=0'};ep=OUT/'execution.json'
    if ep.exists():assert json.loads(ep.read_text())==execution
    else:atomic_json(ep,execution)
    for name,h in sources.items():
        target=Path('results/thrml_spw/source_snapshots')/(h+'.py')
        if not target.exists():target.write_bytes(Path(name).read_bytes())
    tokens=np.load(OUT/'tokens.npz')['test_tokens'];r=runner(p);start=time.monotonic();baseline=r.evaluate(tokens);bp=OUT/'baseline_tokens.npz'
    if bp.exists():
        b=np.load(bp);np.testing.assert_array_equal(b['FP32_nll'],baseline['nll']);np.testing.assert_array_equal(b['FP32_predictions'],baseline['predictions'])
    else:atomic_npz(bp,FP32_nll=baseline['nll'],FP32_predictions=baseline['predictions'])
    write_csv(OUT/'baseline_metrics.csv',[{'condition':'FP32',**baseline['scores']}]);work=OUT/'work';work.mkdir(exist_ok=True);xp=work/'current_inputs.npy';log=(OUT/'worker.log').open('a');env=cuda_environment();env['XLA_FLAGS']=execution['XLA_FLAGS'];worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.fine_range_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Native worker exited {worker.poll()}')
        v=json.loads(line)
        if v['status']=='error':raise RuntimeError(str(v))
        return v
    try:
        assert reply()['status']=='ready'
        for seed in p['sampling_seeds']:
            for name in ['coarse','fine']:
                cfg=condition(p,name);folder=OUT/'runs'/f'{name}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);rp=folder/'run.json'
                if rp.exists():
                    old=json.loads(rp.read_text());assert old['status']=='complete' and old['execution']==execution and sha(folder/'prefix_tokens.npz')==old['token_archive_sha256']
                    for s in old['stages']:assert sha(s['output'])==json.loads(Path(s['metadata']).read_text())['output_sha256']
                    continue
                replacements={};raw={};stages=[];checks=0
                for stage in range(24):
                    value=r.evaluate(tokens,replacements,capture=stage);checks+=value['input_checks'];raw[f'prefix{stage}_nll']=value['nll'];raw[f'prefix{stage}_predictions']=value['predictions']
                    if stage==0:np.testing.assert_array_equal(value['nll'],baseline['nll'])
                    atomic_npy(xp,value['inputs']);yp=folder/f'stage{stage}_native_output.npz';request={'condition':name,'stage':stage,'seed':seed,'samples':cfg['samples'],'group':cfg['group'],'inputs':str(xp),'input_sha256':sha(xp),'output':str(yp)};worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();meta=json.loads(Path(response['metadata']).read_text());assert sha(yp)==meta['output_sha256'];replacements[stage]=(value['inputs'],load_output(yp));stages.append({'stage':stage,'samples':cfg['samples'],'group':cfg['group'],'inputs_sha256':sha(xp),'input_shape':list(value['inputs'].shape),'output':str(yp),'metadata':response['metadata'],'prior_native_output_sha256':[sha(folder/f'stage{k}_native_output.npz') for k in range(stage)]});print(f'Fine-range test {name} seed{seed} stage{stage} elapsed={time.monotonic()-start:.1f}s',flush=True)
                value=r.evaluate(tokens,replacements);checks+=value['input_checks'];assert checks==9600;raw['prefix24_nll']=value['nll'];raw['prefix24_predictions']=value['predictions'];atomic_npz(folder/'prefix_tokens.npz',**raw);updates=sum(json.loads(Path(s['metadata']).read_text())['free_node_updates'] for s in stages);assert updates==4*42467328*cfg['samples'];atomic_json(rp,{'status':'complete','condition':name,'seed':seed,'scores':value['scores'],'stages':stages,'exact_prefix_input_checks':checks,'free_node_updates':updates,'token_archive_sha256':sha(folder/'prefix_tokens.npz'),'execution':execution});print(f'Completed {name} seed{seed}: {value["scores"]}',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    # Keep the final temporary input until an independent causal-path audit verifies its hash.
    rows=[]
    for name in ['coarse','fine']:
        for seed in p['sampling_seeds']:
            record=json.loads((OUT/'runs'/f'{name}_seed{seed}/run.json').read_text());rows.append({'condition':name,'seed':seed,**record['scores'],'free_node_updates':record['free_node_updates']})
    write_csv(OUT/'native_metrics.csv',rows);summary,pair=statistics();atomic_json(OUT/'manifest.json',{'execution':execution,'native_model_cases':10,'native_operators':240,'native_observed_updates':sum(v['free_node_updates'] for v in rows),'summary':summary,'paired_contrast':pair,'elapsed_seconds':time.monotonic()-start,'test_scores_used_for_selection':False,'quality_and_paths_independently_audited':False,'scope':p['scope']});print(json.dumps({'summary':summary,'paired':pair},indent=2))
if __name__=='__main__':main()
