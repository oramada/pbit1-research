"""Persisted call handles and complete predeclared queue, independent of quality."""
from pathlib import Path
import fcntl,hashlib,json,os,time
import modal
from .layout import APP_NAME,PROTOCOL,OUTPUT


def main():
    p=json.loads(Path(PROTOCOL).read_text());digest=hashlib.sha256(Path(PROTOCOL).read_bytes()).hexdigest()
    root=Path(OUTPUT)/'observer';root.mkdir(exist_ok=True);lock=(root/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=root/'state.json';state=json.loads(path.read_text()) if path.exists() else {'protocol_sha256':digest,'calls':{},'completed':{},'status':'starting'}
    assert state['protocol_sha256']==digest
    def save():
        state['updated_unix']=time.time();temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)
        print(json.dumps({'status':state['status'],'completed':len(state['completed']),'submitted':len(state['calls'])}),flush=True)
    stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
    state['observer']={'pid':os.getpid(),'start_ticks':int(stat[19]),'argv':__import__('sys').argv};save()
    fn=modal.Function.from_name(APP_NAME,'case');stats=modal.Function.from_name(APP_NAME,'statistics')
    primary=set(p['primary_cases']);controls=set(p['control_cases']);queue=p['case_order']
    assert set(queue)==set(p['cases']) and len(queue)==len(set(queue))==104
    def poll(name):
        try:result=modal.FunctionCall.from_id(state['calls'][name]).get(timeout=0)
        except modal.exception.FunctionTimeoutError:raise
        except (TimeoutError,modal.exception.TimeoutError):return
        except (modal.exception.ConnectionError,modal.exception.ServiceError) as e:
            state['last_observation_error']={'type':type(e).__name__,'message':str(e)};save();return
        if result.get('status')=='duplicate_rejected_reattach_to_owner':
            state.setdefault('duplicate_submission_ids',[]).append(state['calls'][name]);state['calls'][name]=result['owner']['call_id'];save();return
        state['completed'][name]=result;save()
    try:
        state['status']='complete_predeclared_R1_queue_running';save()
        while len([x for x in queue if x in state['completed']])<len(queue):
            active=[x for x in queue if x in state['calls'] and x not in state['completed']]
            for slug in queue:
                if len(active)>=4:break
                if slug not in state['calls']:
                    call=fn.spawn(slug,digest);state['calls'][slug]=call.object_id;active.append(slug);save()
            for name in active:poll(name)
            if (primary|controls)<=set(state['completed']) and 'statistics_primary' not in state['calls']:
                call=stats.spawn('primary',digest);state['calls']['statistics_primary']=call.object_id;save()
            if 'statistics_primary'in state['calls'] and 'statistics_primary'not in state['completed']:poll('statistics_primary')
            time.sleep(15)
        if 'statistics_primary'not in state['calls']:
            call=stats.spawn('primary',digest);state['calls']['statistics_primary']=call.object_id;save()
        if 'statistics_full'not in state['calls']:
            call=stats.spawn('full',digest);state['calls']['statistics_full']=call.object_id;save()
        state['status']='all104cases_audited_waiting_statistics';save()
        while any(name not in state['completed'] for name in ['statistics_primary','statistics_full']):
            for name in ['statistics_primary','statistics_full']:
                if name not in state['completed']:poll(name)
            time.sleep(15)
        state['status']='all104cases_and_primary_full_statistics_audited';save()
    except Exception as e:
        state['status']='observer_stopped_preserve_all_live_handles_no_resubmission';state['error']={'type':type(e).__name__,'message':str(e)};save();raise

if __name__=='__main__':main()
