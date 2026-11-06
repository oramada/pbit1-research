"""Five immutable follow-up cases, durable handles, no timeout resubmission."""
from pathlib import Path
import fcntl,hashlib,json,os,time
import modal
from .layout import APP_NAME,PROTOCOL,OUTPUT


def main():
    p=json.loads(Path(PROTOCOL).read_text());digest=hashlib.sha256(Path(PROTOCOL).read_bytes()).hexdigest()
    queue=p['case_order'];assert len(queue)==len(set(queue))==5 and set(queue)==set(p['cases'])
    root=Path(OUTPUT)/'observer';root.mkdir(exist_ok=True);lock=(root/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=root/'state.json';state=json.loads(path.read_text())if path.exists()else{'protocol_sha256':digest,'calls':{},'completed':{},'status':'starting'}
    assert state['protocol_sha256']==digest
    def save():
        state['updated_unix']=time.time();tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
        print(json.dumps({'status':state['status'],'completed':len(state['completed']),'submitted':len(state['calls'])}),flush=True)
    stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
    state['observer']={'pid':os.getpid(),'start_ticks':int(stat[19]),'argv':__import__('sys').argv};save()
    fn=modal.Function.from_name(APP_NAME,'case');stats=modal.Function.from_name(APP_NAME,'statistics')
    def poll(name):
        try:r=modal.FunctionCall.from_id(state['calls'][name]).get(timeout=0)
        except modal.exception.FunctionTimeoutError:raise
        except (TimeoutError,modal.exception.TimeoutError):return
        except (modal.exception.ConnectionError,modal.exception.ServiceError)as e:
            state['last_observation_error']={'type':type(e).__name__,'message':str(e)};save();return
        if r.get('status')=='duplicate_rejected_reattach_to_owner':
            state.setdefault('duplicate_submission_ids',[]).append(state['calls'][name]);state['calls'][name]=r['owner']['call_id'];save();return
        state['completed'][name]=r;save()
    try:
        state['status']='four_parallel_H100_fixed_highT_followup_cases_running';save()
        while any(slug not in state['completed']for slug in queue):
            active=[x for x in queue if x in state['calls']and x not in state['completed']]
            for slug in queue:
                if len(active)>=4:break
                if slug not in state['calls']:
                    call=fn.spawn(slug,digest);state['calls'][slug]=call.object_id;active.append(slug);save()
            for slug in active:poll(slug)
            time.sleep(15)
        if 'statistics'not in state['calls']:
            call=stats.spawn(digest);state['calls']['statistics']=call.object_id;save()
        state['status']='all_five_cases_audited_waiting_independent_statistics';save()
        while 'statistics'not in state['completed']:poll('statistics');time.sleep(15)
        state['status']='all_five_highT_cases_and_independent_statistics_complete';save()
    except Exception as e:
        state['status']='observer_stopped_preserve_all_live_handles_no_resubmission';state['error']={'type':type(e).__name__,'message':str(e)};save();raise

if __name__=='__main__':main()
