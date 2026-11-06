"""Frozen native full-weight operator probes; validation only, no NN quality claim."""
import json,subprocess,time
from pathlib import Path
from .transformer_block_gpu import sha,save_json,write_csv
from .gpu_environment import cuda_environment
OUT=Path('results/thrml_spw/range_localization')
def main():
    p=json.loads((OUT/'protocol.json').read_text());analytic=json.loads((OUT/'analytic_manifest.json').read_text());assert analytic['all_coordinate_nested_ideal_variance_passed'] and analytic['execution']['protocol_sha256']==sha(OUT/'protocol.json');sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['range_localization_probes.py','range_localization_worker.py','localized_native.py','localized_controls.py','calibrated_transformer_native.py','grouped_range_native.py']};execution={'protocol_sha256':sha(OUT/'protocol.json'),'analytic_manifest_sha256':sha(OUT/'analytic_manifest.json'),'source_sha256':sources};ep=OUT/'native_execution.json'
    if ep.exists():assert json.loads(ep.read_text())==execution
    else:save_json(ep,execution)
    if (OUT/'native_manifest.json').exists():
        m=json.loads((OUT/'native_manifest.json').read_text());assert m['execution']==execution
        for c in m['cases']:assert sha(c['output'])==json.loads(Path(c['metadata']).read_text())['output_sha256']
        print(json.dumps({'cached_complete':True,'new_native_sampling':False,'manifest_sha256':sha(OUT/'native_manifest.json')}));return
    for name,h in sources.items():
        target=Path('results/thrml_spw/source_snapshots')/(h+'.py')
        if not target.exists():target.write_bytes(Path(name).read_bytes())
    log=(OUT/'worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.range_localization_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment());start=time.monotonic();cases=[];rows=[]
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Worker exited {worker.poll()}')
        v=json.loads(line)
        if v['status']=='error':raise RuntimeError(str(v))
        return v
    try:
        assert reply()['status']=='ready'
        for stage,anchor in enumerate(analytic['anchors']):
            assert sha(anchor['inputs'])==anchor['inputs_sha256'] and sha(anchor['target'])==anchor['target_sha256']
            for condition in p['native_conditions']:
                for seed in range(5):
                    folder=OUT/'probes'/f'{condition["name"]}_stage{stage}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);yp=folder/'output.npy';request={'condition':condition['name'],'stage':stage,'seed':seed,'inputs':anchor['inputs'],'inputs_sha256':anchor['inputs_sha256'],'target':anchor['target'],'target_sha256':anchor['target_sha256'],'output':str(yp)};worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();meta=json.loads(Path(response['metadata']).read_text());assert sha(yp)==meta['output_sha256'];cases.append({'condition':condition['name'],'stage':stage,'seed':seed,'output':str(yp),'metadata':response['metadata'],'inputs':anchor['inputs'],'target':anchor['target']});rows.append({'condition':condition['name'],'group':condition['group'],'samples':condition['samples'],'stage':stage,'seed':seed,**{k:meta[k] for k in ['predicted_sampling_MSE','programmed_mean_bias_MSE','predicted_target_MSE','observed_target_MSE','observed_MSE_vs_programmed_mean','observed_node_updates']}});print(f'Native range probe stage{stage} {condition["name"]} seed{seed} elapsed={time.monotonic()-start:.1f}s',flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    assert len(cases)==360;write_csv(OUT/'native_probe_metrics.csv',rows);m={'execution':execution,'native_operators':360,'native_observed_updates':sum(v['observed_node_updates'] for v in rows),'cases':cases,'elapsed_seconds':time.monotonic()-start,'full_stochastic_neural_models_executed':0,'heldout_test_models_executed':0,'scope':p['scope']};save_json(OUT/'native_manifest.json',m);print(json.dumps({k:v for k,v in m.items() if k!='cases'},indent=2))
if __name__=='__main__':main()
