"""Client of deployed functions; result handles outlive this process."""
import argparse
import hashlib
import json
from pathlib import Path
import modal


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    root=Path('results/thrml_spw/broad_verification_20261105/cloud_recovery_preflight_v1')/a.job
    root.mkdir(parents=True,exist_ok=False)
    fn=modal.Function.from_name('pbit1-recovery-preflight-v1','step')
    reader=modal.Function.from_name('pbit1-recovery-preflight-v1','inspect')
    calls=[fn.spawn(a.job,'after_commit') for _ in range(2)]
    (root/'calls.json').write_text(json.dumps({'job':a.job,'first_call_ids':[v.object_id for v in calls]},indent=2)+'\n')
    results=[modal.FunctionCall.from_id(v.object_id).get(timeout=600) for v in calls]
    assert sorted(v['status'] for v in results)==['duplicate_rejected','stage_committed_then_subprocess_exit']
    resume=fn.spawn(a.job,'resume')
    (root/'resume_call.json').write_text(json.dumps({'call_id':resume.object_id})+'\n')
    resumed=modal.FunctionCall.from_id(resume.object_id).get(timeout=600)
    assert resumed['status']=='resumed'
    checked=reader.remote(a.job)
    report={'status':'deployed_durable_calls_atomic_duplicate_claim_and_cross_container_recovery_passed',
        'job':a.job,'first_calls':results,'resume':resumed,'independent_reader':checked,
        'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [Path(__file__),Path(__file__).with_name('app.py')]}}
    (root/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
