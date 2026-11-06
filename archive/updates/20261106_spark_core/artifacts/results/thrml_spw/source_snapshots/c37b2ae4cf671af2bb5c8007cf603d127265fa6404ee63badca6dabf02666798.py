"""Read-only discovery/polling of all current campaign calls.

Never spawns, cancels, reattaches, resubmits or rewrites campaign state. Future
LFM3 seeds/statistics are discovered from its existing observer, not hardcoded.
"""
from pathlib import Path
import hashlib,json,time
import modal

BASE=Path('results/thrml_spw/broad_verification_20261105')
STUDIES={
    'r1_execution_v1':('r1_execution_v1/observer/state.json','r1_execution_v1/protocol.json','cloud.r1_execution_v1.client'),
    'highT_execution_v1':('highT_execution_v1/observer/state.json','highT_execution_v1/protocol.json','cloud.highT_execution_v1.client'),
    'LFM3':('holdout_v5_monitor/state.json','../lfm3_1p7b/holdout_native_v4_protocol.json','cloud.broad_validation_v5.client'),
}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def observer_identity(record,module):
    p=Path('/proc')/str(record.get('pid',0))
    try:
        stat=(p/'stat').read_text().split(') ',1)[1].split()
        args=(p/'cmdline').read_bytes().split(b'\0')
    except (OSError,IndexError):return False
    return int(stat[19])==record.get('start_ticks')and module.encode()in args


def poll(cid):
    try:r=modal.FunctionCall.from_id(cid).get(timeout=0)
    except modal.exception.FunctionTimeoutError as e:return {'status':'terminal_function_timeout','error_type':type(e).__name__}
    except (TimeoutError,modal.exception.TimeoutError):return {'status':'existing_call_live_or_waiting'}
    except (modal.exception.ConnectionError,modal.exception.ServiceError)as e:return {'status':'transient_observation_error_keep_same_handle','error_type':type(e).__name__}
    except Exception as e:return {'status':'requires_investigation_not_resubmission','error_type':type(e).__name__}
    else:
        result={'status':'returned','result_status':r.get('status')}
        if 'owner'in r and isinstance(r['owner'],dict):result['owner_call_id']=r['owner'].get('call_id')
        return result


def main():
    observations=[];studies={};seen={}
    for name,(state_name,protocol_name,module)in STUDIES.items():
        state=json.loads((BASE/state_name).read_text());protocol=BASE/protocol_name
        assert sha(protocol)==state['protocol_sha256'],name+' protocol/state mismatch'
        identity=observer_identity(state.get('observer',{}),module)
        studies[name]={'observer_status':state['status'],'observer_identity_confirmed':identity,
            'observer_recorded_audited_cases':len([k for k in state['completed']if not k.startswith('statistics')]),
            'submitted_calls':len(state['calls']),'protocol_sha256':state['protocol_sha256']}
        for key,cid in state['calls'].items():
            if key in state['completed']:continue
            result=seen.get(cid)
            if result is None:result=poll(cid);seen[cid]=result
            observations.append({'study':name,'case':key,'call_id':cid,**result})
    problems=[r for r in observations if r['status']in ['terminal_function_timeout','requires_investigation_not_resubmission']]
    dead=[name for name,r in studies.items()if not r['observer_identity_confirmed']and not r['observer_status'].startswith(('all_cases_and_statistics_audited','all104cases_and_primary_full_statistics_audited','all_five_highT_cases_and_independent_statistics_complete'))]
    record={'status':'existing_campaign_calls_polled_read_only','observed_unix':time.time(),'source_sha256':sha(__file__),
        'studies':studies,'observations':observations,'requires_investigation':bool(problems or dead),
        'observer_identity_problems':dead,'scope':'Protocol and localobserveridentitieschecked, every currently activehandle independentlypolled. No taskqualityconclusion, newtrial, taskcontrolaction or assumptionthatobservationtimeoutsmeanfailure.'}
    out=BASE/'verified_wait_observations';out.mkdir(exist_ok=True);path=out/(str(time.time_ns())+'.json');path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'record':str(path),'studies':studies,'active_handles_polled':len(seen),
        'status_counts':{s:sum(r['status']==s for r in observations)for s in sorted({r['status']for r in observations})},
        'requires_investigation':record['requires_investigation']}))

if __name__=='__main__':main()
