"""Deployed CPU-only durable claim/Volume recovery fault fixture; no model data."""
import hashlib
import json
from pathlib import Path
import modal
from cloud.broad_validation_v2.app import image, setup_workspace, volume

app = modal.App('pbit1-recovery-preflight-v1')
claims = modal.Dict.from_name('pbit1-recovery-preflight-v1', create_if_missing=True)


@app.function(image=image, cpu=4, memory=8192, volumes={'/workspace': volume},
              timeout=600, retries=0, max_containers=3, scaledown_window=2)
def step(job: str, action: str):
    import os
    import subprocess
    if action not in ('after_commit', 'resume') or not job.isalnum():
        raise ValueError('Invalid fixture identity/action')
    call_id = modal.current_function_call_id()
    key = job + ':' + action
    owner = {'call_id': call_id, 'job': job, 'action': action}
    won = claims.put(key, owner, skip_if_exists=True)
    previous = claims.get(key)
    if not won and previous['call_id'] != call_id:
        return {'status': 'duplicate_rejected', 'owner': previous}
    setup_workspace()
    volume.reload()
    root = Path('/workspace/recovery_preflight_v1') / job
    root.mkdir(parents=True, exist_ok=True)
    log = root / (action + '-' + call_id + '.log')
    # Preemption of this function reattaches to its own claim; the StageStore
    # decides whether the interrupted subprocess stage needs same-key replay.
    with log.open('a') as stream:
        command = ['/opt/thrml/bin/python', '-m',
                   'research.thrml_spw.lfm3.preflight_stage_checkpoint_v1',
                   '--worker', '--root', str(root / 'store'), '--interrupt',
                   'after_commit' if action == 'after_commit' else 'none']
        completed = subprocess.run(command, cwd='/workspace',
            env={**os.environ, 'PYTHONPATH': '/code'}, stdout=stream, stderr=stream)
    expected = 73 if action == 'after_commit' else 0
    volume.commit()
    assert completed.returncode == expected, {'returncode': completed.returncode, 'log': str(log)}
    result = {'status': 'stage_committed_then_subprocess_exit' if action == 'after_commit' else 'resumed',
              'owner': previous, 'subprocess_exit': completed.returncode}
    claims.put(key + ':result', result)
    return result


@app.function(image=image, cpu=1, memory=2048, volumes={'/workspace': volume}, timeout=120,
              retries=0, max_containers=1, scaledown_window=2)
def inspect(job: str):
    import subprocess
    import os
    setup_workspace()
    volume.reload()
    root = Path('/workspace/recovery_preflight_v1') / job
    # A distinct container/process loads and independently checks the closed files.
    code = '''import json, numpy as np
from pathlib import Path
from research.thrml_spw.lfm3.stage_checkpoint_v1 import StageStore
from research.thrml_spw.lfm3.preflight_stage_checkpoint_v1 import CONTRACT
root = Path(__import__('sys').argv[1])
s = StageStore(root/'store', CONTRACT)
records = [s.load(i,'native') for i in range(2)]
assert all(records)
events = [json.loads(v) for v in (s.root/'sampling_events.jsonl').read_text().splitlines()]
assert [v['batch'] for v in events] == [0,1], events
for record in records:
    values = np.load(s.root/record['output']['path'])
    assert values.shape == (65,) and np.isfinite(values).all()
print(json.dumps({'status':'cross_container_volume_checkpoints_verified', 'sampling_events':events,
    'record_outputs':[v['output'] for v in records], 'scope':'CPU officialTHRML fixture; no model quality or GPU recovery claim'}))
'''
    p = subprocess.run(['/opt/thrml/bin/python', '-c', code, str(root)], cwd='/workspace',
        env={**os.environ, 'PYTHONPATH': '/code'}, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


@app.local_entrypoint()
def run(job: str):
    root = Path('results/thrml_spw/broad_verification_20261105/cloud_recovery_preflight_v1') / job
    root.mkdir(parents=True, exist_ok=False)
    calls = [step.spawn(job, 'after_commit') for _ in range(2)]
    (root/'calls.json').write_text(json.dumps({'job':job,'first_call_ids':[v.object_id for v in calls]},indent=2)+'\n')
    results = [v.get(timeout=600) for v in calls]
    assert sorted(v['status'] for v in results) == ['duplicate_rejected','stage_committed_then_subprocess_exit']
    resume = step.spawn(job, 'resume')
    with (root/'resume_call.json').open('x') as f: json.dump({'call_id':resume.object_id},f)
    resumed = modal.FunctionCall.from_id(resume.object_id).get(timeout=600)
    assert resumed['status'] == 'resumed'
    checked = inspect.remote(job)
    report = {'status':'durable_calls_atomic_duplicate_claim_and_cross_container_recovery_passed',
              'job':job,'first_calls':results,'resume':resumed,'independent_reader':checked,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
