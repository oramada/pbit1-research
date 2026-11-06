"""Evict only redundant cache wheels with matching installed major payloads."""
from pathlib import Path
import zipfile,hashlib,json,os,shutil,datetime
OUT=Path('results/thrml_spw/pip_installer_cache_cleanup_remaining_20261028');CACHE=Path('/home/omarramadan/.cache/pip/http-v2')
TARGETS={('nvidia-curand', '10.4.0.35'):Path('.venv/lib/python3.12/site-packages'),('nvidia-nvshmem-cu13', '3.4.5'):Path('.venv/lib/python3.12/site-packages'),('pyarrow', '26.0.0'):Path('.venv/lib/python3.12/site-packages'),('nvidia-cuda-nvrtc', '13.0.88'):Path('.venv/lib/python3.12/site-packages'),('nvidia-nvjitlink', '13.4.92'):Path('.venv/lib/python3.12/site-packages'),('scipy', '1.18.1'):Path('.venv-thrml/lib/python3.12/site-packages'),('numpy', '2.5.4'):Path('.venv-thrml/lib/python3.12/site-packages')}
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(path,value):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def main():
    assert not (OUT/'inventory.json').exists();rows=[];seen=set()
    for body in CACHE.rglob('*.body'):
        st=body.stat()
        if st.st_size<15000000 or st.st_nlink!=1 or st.st_uid!=os.getuid() or body.is_symlink():continue
        try:
            with zipfile.ZipFile(body) as z:
                met=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
                if len(met)!=1:continue
                text=z.read(met[0]).decode().splitlines();name=next(v[6:] for v in text if v.startswith('Name: '));version=next(v[9:] for v in text if v.startswith('Version: '));key=(name,version)
                if key not in TARGETS:continue
                assert key not in seen;site=TARGETS[key];installed_meta=site/met[0];assert installed_meta.exists() and digest(installed_meta)==hashlib.sha256(z.read(met[0])).hexdigest()
                payload=max((v for v in z.infolist() if not v.is_dir()),key=lambda v:v.file_size);installed=site/payload.filename;assert installed.is_file() and installed.stat().st_size==payload.file_size
                h=hashlib.sha256()
                with z.open(payload) as f:
                    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
                ph=h.hexdigest();assert digest(installed)==ph
        except zipfile.BadZipFile:continue
        header=body.with_suffix('');assert body.resolve().is_relative_to(CACHE.resolve());assert not header.is_symlink()
        if header.exists():assert header.stat().st_nlink==1 and header.stat().st_uid==os.getuid()
        row={'cache_body':str(body),'cache_body_bytes':st.st_size,'cache_body_sha256':digest(body),'package':name,'version':version,'cache_inode':st.st_ino,'cache_mtime_ns':st.st_mtime_ns,'cache_header':str(header) if header.exists() else None,'cache_header_bytes':header.stat().st_size if header.exists() else 0,'cache_header_sha256':digest(header) if header.exists() else None,'installed_metadata':str(installed_meta),'installed_metadata_sha256':digest(installed_meta),'installed_major_payload':str(installed),'installed_major_payload_sha256':ph,'installed_major_payload_bytes':payload.file_size};rows.append(row);seen.add(key)
    assert seen==set(TARGETS),(seen,set(TARGETS));before=shutil.disk_usage('.').free;inventory={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':digest(__file__),'free_before':before,'entries':rows,'scope':'Only disposable cached installer ZIP/response entries, all matched byte-for-byte installed metadata and largest runtime payload. Installed packages, source snapshots, models and research evidence are untouched. Cache wheels may be downloaded again; their checksums/versions are retained.'};save(OUT/'inventory.json',inventory)
    for r in rows:
        body=Path(r['cache_body']);st=body.stat();assert (st.st_ino,st.st_mtime_ns,st.st_size)==(r['cache_inode'],r['cache_mtime_ns'],r['cache_body_bytes']) and digest(body)==r['cache_body_sha256'];assert digest(r['installed_major_payload'])==r['installed_major_payload_sha256']
        if r['cache_header']:
            header=Path(r['cache_header']);assert digest(header)==r['cache_header_sha256'];header.unlink()
        body.unlink()
    for r in rows:assert digest(r['installed_major_payload'])==r['installed_major_payload_sha256'] and digest(r['installed_metadata'])==r['installed_metadata_sha256']
    save(OUT/'completion.json',{'completed':True,'cache_wheels_removed':len(rows),'cache_file_bytes_removed':sum(r['cache_body_bytes']+r['cache_header_bytes'] for r in rows),'installed_major_payloads_unchanged':True,'free_before':before,'actual_free_after':shutil.disk_usage('.').free,'inventory_sha256':digest(OUT/'inventory.json'),'source_sha256':digest(__file__),'scope':inventory['scope']});print((OUT/'completion.json').read_text(),flush=True)
if __name__=='__main__':main()
