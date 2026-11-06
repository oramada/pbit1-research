"""Previously frozen C8192 spatial controls, after bridge and input gates."""
from pathlib import Path
import json
import modal
app=modal.App('spw-h100-C8192-v1')
volume=modal.Volume.from_name('spw-h100-validation-v1')
image=modal.Image.from_id('im-5Z024b95pCDtV6oIXrPpIn')

def setup_workspace():
    import os
    root=Path('/workspace')
    for name,target in [('.venv','/opt/torch'),('.venv-thrml-gpu','/opt/thrml'),('research','/code/research'),('cloud','/code/cloud')]:
        path=root/name
        if os.path.lexists(path):
            assert path.is_symlink() and os.readlink(path)==target
        else:path.symlink_to(target)
    if not os.path.lexists('/code/.venv'):Path('/code/.venv').symlink_to('/opt/torch')


@app.function(image=image,gpu='H100!',cpu=8,memory=65536,ephemeral_disk=524288,volumes={'/workspace':volume},timeout=14400,retries=0,max_containers=4,scaledown_window=10)
def run_case(module: str, arguments: list[str], label: str):
    """Only versioned cloud runners may write new case namespaces."""
    import os
    import subprocess
    assert module.startswith('cloud.modal_h100_v1.')
    setup_workspace()
    volume.reload()
    logdir = Path('/workspace/cloud_execution_logs_v1')
    logdir.mkdir(exist_ok=True)
    path = logdir/(label+'.log')
    with path.open('x') as log:
        code = subprocess.call(['/opt/torch/bin/python', '-m', module, *arguments], cwd='/workspace',
            env={**os.environ, 'PYTHONPATH':'/code'}, stdout=log, stderr=log)
    volume.commit()
    return {'label':label,'exit_code':code,'log':str(path)}


@app.function(image=image,cpu=4,memory=8192,volumes={'/workspace':volume},timeout=86400,retries=0,max_containers=1)
def campaign():
    import time,hashlib,concurrent.futures
    setup_workspace();state={'status':'waiting_full_bridge_campaign_and_C8192_metadata','completed':[]}
    p=Path('/workspace/campaign_C8192_v1.json')
    def save():
        p.write_text(json.dumps(state,indent=2)+'\n');volume.commit();print(json.dumps(state),flush=True)
    save();deadline=time.time()+24*3600
    while True:
        volume.reload()
        current=Path('/workspace/campaign_v2.json');ready=Path('/workspace/C8192_assets_v1.json')
        if current.exists():
            c=json.loads(current.read_text())
            assert not c.get('failed'),'Prior H100 bridge/comparison failure; no continuation'
            if c['status']=='cloud_comparison_cases_and_independent_audits_complete_statistics_pending' and ready.exists():break
        assert time.time()<deadline,'Continuation infrastructure deadline'
        time.sleep(30)
    m=json.loads(ready.read_text())
    for name,rec in m['files'].items():
        q=Path('/workspace')/name
        assert hashlib.sha256(q.read_bytes()).hexdigest()==rec['sha256'],name
    specs=[(bits,seed) for bits in [4,8] for seed in range(202610600,202610605)]
    completed=set();active={};state['status']='parallel_C8192_matched_controls';save()
    def execute(bits,seed):
        return run_case.remote('cloud.modal_h100_v1.word_case_v2',['--budget','8192','--bits',str(bits),'--seed',str(seed)],f'C8192_K{bits}_{seed}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while specs or active:
            for bits,seed in list(specs):
                if len(active)>=4:break
                if seed==202610600 or (bits,202610600) in completed:
                    active[pool.submit(execute,bits,seed)]=(bits,seed);specs.remove((bits,seed))
            assert active,'Unmet first-seed audit dependency'
            state['running']=list(active.values());save()
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                item=active.pop(future);result=future.result()
                assert result['exit_code']==0,result
                completed.add(item);state['completed'].append(item)
    state['running']=[];state['status']='all_C8192_spatial_cases_and_independent_audits_complete_statistics_pending';save()
    return state
@app.local_entrypoint()
def launch():
    print(json.dumps(campaign.remote()))
