"""Versioned delta migration; never overwrite the historical plan or audits."""
import fcntl, hashlib, json, shutil
from pathlib import Path
from .lossless_evidence_archive import OUT, archive_one, durable_json
from .verify_npy_archive import independently_restore
DELTA=Path('results/thrml_spw/lossless_archive_capacity_20261027')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    DELTA.mkdir(exist_ok=True)
    with (OUT/'operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (DELTA/'protocol.json').exists(), 'No implicit restart'
        plan=json.loads((OUT/'plan.json').read_text())
        for p,h in plan['source_sha256'].items(): assert sha(p)==h
        baseline={str(p):sha(p) for p in [OUT/'plan.json',OUT/'evidence_audit.json',OUT/'migration.json'] if p.exists()}
        for p in (OUT/'records').glob('*.json'): baseline[str(p)]=sha(p)
        rows=[r for r in plan['entries'] if Path(r['original_path']).exists() and not Path(r['archive_path']).exists() and not (OUT/'records'/f"{r['id']:05d}.json").exists()]
        protocol={'purpose':'Recover capacity with unchanged lossless codec; preserve earlier closed audit and records. No scientific scores or samples changed.', 'target_free_bytes':2_100_000_000,'free_before':shutil.disk_usage('.').free,'entries':rows,'preserved_sha256':baseline,'source_sha256':sha(__file__),'original_plan_sha256':sha(OUT/'plan.json')}
        durable_json(DELTA/'protocol.json',protocol)
        results=[]; verified=[]
        for row in rows:
            if shutil.disk_usage('.').free>=protocol['target_free_bytes']:break
            r=archive_one(row);results.append(r)
            if r['status']=='archived':
                raw,meta=independently_restore(Path(r['archive_path']))
                assert hashlib.sha256(raw).hexdigest()==row['original_sha256'] and len(raw)==row['original_bytes']
                assert meta['original_path']==row['original_path'] and meta['mode']==row['mode'] and meta['mtime_ns']==row['mtime_ns']
                assert not Path(row['original_path']).exists()
                verified.append({'id':row['id'],'original_sha256':row['original_sha256'],'original_bytes':len(raw),'archive_sha256':sha(r['archive_path']),'archive_bytes':Path(r['archive_path']).stat().st_size,'independent_recovery_exact':True})
            progress={'processed':len(results),'archived':len(verified),'free_bytes':shutil.disk_usage('.').free,'saved_bytes':sum(x['original_bytes']-x['archive_bytes'] for x in verified)}
            durable_json(DELTA/'progress.json',progress)
            if len(results)%25==0:print(json.dumps(progress),flush=True)
        for p,h in baseline.items():assert sha(p)==h
        durable_json(DELTA/'migration.json',{'complete':True,'results':results,**progress})
        durable_json(DELTA/'evidence_audit.json',{'passed':True,'all_prior_records_and_audits_unchanged':True,'protocol_sha256':sha(DELTA/'protocol.json'),'rows':verified,'free_bytes':shutil.disk_usage('.').free,'scope':'New delta archives independently decoded after persisted write; preexisting closed audit and every prior record byte unchanged. Standard historical restore CLI handles added records.'})
        print(json.dumps({'complete':True,**progress}),flush=True)
if __name__=='__main__':main()
