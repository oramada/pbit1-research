"""Durable local observer of deployed jobs. Poll timeouts never resubmit work."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import modal

ROOT=Path('results/thrml_spw/broad_verification_20261105')


def main():
    root=ROOT/'holdout_v3_monitor';root.mkdir(exist_ok=True)
    lock=(root/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=Path('results/thrml_spw/lfm3_1p7b/holdout_native_v3_protocol.json')
    digest=hashlib.sha256(protocol.read_bytes()).hexdigest();p=json.loads(protocol.read_text())
    path=root/'state.json'
    state=json.loads(path.read_text()) if path.exists() else {'protocol_sha256':digest,'calls':{},'completed':{},'status':'starting'}
    assert state['protocol_sha256']==digest
    def save():
        state['updated_unix']=time.time();temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)
        print(json.dumps(state),flush=True)
    proc=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
    state['observer']={'pid':os.getpid(),'start_ticks':int(proc[19]),'argv':__import__('sys').argv};save()
    fn=modal.Function.from_name('pbit1-broad-heldout-v3','case')
    stats=modal.Function.from_name('pbit1-broad-heldout-v3','statistics')
    def ensure(key,seed=None):
        if key not in state['calls']:
            call=fn.spawn(seed,digest) if seed is not None else stats.spawn(digest)
            state['calls'][key]=call.object_id;save()
    def poll(key):
        try:result=modal.FunctionCall.from_id(state['calls'][key]).get(timeout=0)
        except modal.exception.FunctionTimeoutError:raise
        except modal.exception.TimeoutError:return False
        except (modal.exception.ConnectionError,modal.exception.ServiceError) as error:
            state['last_observation_error']={'type':type(error).__name__,'message':str(error)};save();return False
        if result.get('status')=='duplicate_rejected_reattach_to_owner':
            state.setdefault('rejected_duplicate_calls',[]).append(state['calls'][key])
            state['calls'][key]=result['owner']['call_id'];save();return False
        state['completed'][key]=result;save();return True
    try:
        first=str(p['seeds'][0]);ensure(first,p['seeds'][0]);state['status']='first_seed_running_or_waiting';save()
        while first not in state['completed']:
            if not poll(first):time.sleep(15)
        assert state['completed'][first]['status']=='heldout_case_and_independent_audit_complete'
        for seed in p['seeds'][1:]:ensure(str(seed),seed)
        state['status']='remaining_four_seeds_running_or_waiting';save()
        while any(str(seed) not in state['completed'] for seed in p['seeds'][1:]):
            for seed in p['seeds'][1:]:
                if str(seed) not in state['completed']:poll(str(seed))
            time.sleep(15)
        assert all(state['completed'][str(s)]['status']=='heldout_case_and_independent_audit_complete' for s in p['seeds'])
        ensure('statistics');state['status']='independent_statistics_running';save()
        while 'statistics' not in state['completed']:
            if not poll('statistics'):time.sleep(15)
        state['status']='all_cases_and_statistics_audited';save()
    except Exception as error:
        state['status']='observer_stopped_preserve_handles_no_resubmission'
        state['error']={'type':type(error).__name__,'message':str(error)};save();raise

if __name__=='__main__':main()
