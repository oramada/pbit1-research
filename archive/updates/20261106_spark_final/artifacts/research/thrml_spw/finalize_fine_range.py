"""Recover summary-only finalization after completed runs; never construct a model."""
import json,hashlib,re
from pathlib import Path
from .fine_range_common import OUT,condition,sha,atomic_json
from .transformer_block_gpu import write_csv
from .fine_range_statistics import main as statistics
def main():
    p=json.loads((OUT/'protocol.json').read_text());execution=json.loads((OUT/'execution.json').read_text());assert execution['protocol_sha256']==sha(OUT/'protocol.json') and sha(OUT/'tokens.npz')==p['tokens_sha256'];assert not (OUT/'manifest.json').exists(),'Do not overwrite an already finalized cohort'
    for name,h in execution['source_sha256'].items():assert sha(name)==h
    rows=[];operators=0
    for name in ['coarse','fine']:
        cfg=condition(p,name)
        for seed in p['sampling_seeds']:
            folder=OUT/'runs'/f'{name}_seed{seed}';r=json.loads((folder/'run.json').read_text());assert r['status']=='complete' and r['execution']==execution and r['condition']==name and r['seed']==seed and r['exact_prefix_input_checks']==9600 and sha(folder/'prefix_tokens.npz')==r['token_archive_sha256'] and len(r['stages'])==24;updates=0
            for stage,s in enumerate(r['stages']):
                m=json.loads(Path(s['metadata']).read_text());sig=m['signature'];assert s['stage']==sig['stage']==stage and sig['condition']==name and sig['seed']==seed and sig['group']==s['group']==cfg['group'] and sig['samples']==s['samples']==cfg['samples'] and sig['execution_sha256']==sha(OUT/'execution.json') and sha(s['output'])==m['output_sha256'];assert m['free_node_updates']==4*cfg['samples']*p['control_artifacts'][str(cfg['group'])][stage]['weights'];updates+=m['free_node_updates'];operators+=1
            assert updates==r['free_node_updates']==4*p['converted_weights']*cfg['samples'];rows.append({'condition':name,'seed':seed,**r['scores'],'free_node_updates':updates})
    assert operators==240 and len(rows)==10;incident=OUT/'execution_incidents';incident.mkdir(exist_ok=True);log=Path('/tmp/thrml-fine-range-confirmation.log').read_bytes();assert b'TimeoutExpired' in log and b'Completed fine seed4' in log;(incident/'original_launcher_shutdown_failure.log').write_bytes(log);elapsed=[float(x) for x in re.findall(rb'elapsed=([0-9.]+)s',log)];recovery={'parent_terminal_exit_code':1,'failure':'Launcher60s shutdown wait expired after all10 complete model records; both parent and native worker subsequently observed absent','production_execution_sha256':sha(OUT/'execution.json'),'all_model_cases_complete':10,'all_native_operators_verified_by_record_hash':240,'no_model_constructed':True,'no_native_sampling_repeated':True,'finalizer_sha256':sha(__file__),'original_failure_log_sha256':hashlib.sha256(log).hexdigest()};atomic_json(incident/'summary_only_recovery.json',recovery);write_csv(OUT/'native_metrics.csv',rows);summary,pair=statistics();manifest={'execution':execution,'native_model_cases':10,'native_operators':240,'native_observed_updates':sum(v['free_node_updates'] for v in rows),'summary':summary,'paired_contrast':pair,'elapsed_seconds':None,'last_native_log_elapsed_seconds':max(elapsed),'test_scores_used_for_selection':False,'quality_and_paths_independently_audited':False,'summary_only_recovery':recovery,'scope':p['scope']};atomic_json(OUT/'manifest.json',manifest);print(json.dumps({'postprocessing_recovered':True,'new_model_or_native_runs':0,'summary':summary,'paired':pair},indent=2))
if __name__=='__main__':main()
