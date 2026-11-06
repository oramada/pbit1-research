"""Verify every candidate's original bytes after physical representation changes."""
import json,hashlib,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .verify_npy_archive import independently_restore
OUT=Path('results/thrml_spw/lossless_archive')
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    plan=json.loads((OUT/'plan.json').read_text());index=json.loads(Path(plan['input_index']).read_text());assert sha(Path(plan['input_index']).read_bytes())==plan['input_index_sha256'];expected={r['id']:r for r in plan['entries']};assert len(expected)==2970
    def verify(row):
        original=Path(row['original_path']);rp=OUT/'records'/f"{row['id']:05d}.json";assert row['original_sha256']==index[str(original)]['sha256'] and row['original_bytes']==index[str(original)]['bytes']
        if rp.exists():
            r=json.loads(rp.read_text());assert all(r[k]==row[k] for k in row);archive=Path(r['archive_path'])
            if archive.exists():
                assert sha(archive.read_bytes())==r['archive_sha256'] and archive.stat().st_size==r['archive_bytes'];raw,m=independently_restore(archive);assert len(raw)==row['original_bytes'] and sha(raw)==row['original_sha256'] and m['original_path']==row['original_path'] and m['mode']==row['mode'] and m['mtime_ns']==row['mtime_ns']
            else:assert r['status']=='restored' and original.exists()
            if original.exists():assert sha(original.read_bytes())==row['original_sha256']
            else:assert archive.exists() and r['status']=='archived' and r['raw_removed']
            return {'id':row['id'],'original_path':str(original),'status':r['status'],'original_bytes':row['original_bytes'],'archive_bytes':r['archive_bytes'],'original_sha256':row['original_sha256'],'exact_recovery_verified':True}
        assert original.exists() and sha(original.read_bytes())==row['original_sha256'];return {'id':row['id'],'original_path':str(original),'status':'unchanged_raw','original_bytes':row['original_bytes'],'exact_recovery_verified':True}
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(verify,plan['entries']))
    archived=[r for r in rows if r['status']=='archived'];r={'passed':True,'candidate_files':len(rows),'archived_files':len(archived),'unchanged_or_restored_raw_files':len(rows)-len(archived),'every_original_file_byte_checksum_recoverable':True,'original_file_bytes_checked':sum(r['original_bytes'] for r in rows),'archived_original_bytes':sum(r['original_bytes'] for r in archived),'archive_bytes':sum(r['archive_bytes'] for r in archived),'file_bytes_saved':sum(r['original_bytes']-r['archive_bytes'] for r in archived),'actual_free_bytes':shutil.disk_usage('.').free,'plan_sha256':sha((OUT/'plan.json').read_bytes()),'source_sha256':sha(Path(__file__).read_bytes()),'rows':rows,'scope':'Every archived original verified via independently decoded persisted bytes; every unarchived original compared with frozen pre-migration index. Header/array bit patterns and original hashes retained. Historical paths must be restored before old commands. No neural experiment or score altered.'};(OUT/'evidence_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
