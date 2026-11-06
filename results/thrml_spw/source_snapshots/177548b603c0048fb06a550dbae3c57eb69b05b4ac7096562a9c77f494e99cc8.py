"""Durable reversible archival of selected completed-run NPY evidence only.

Every raw unlink follows verified disk decoding and a durable restoration record.
Scientific arrays, their original file bytes and hashes are preserved exactly.
"""
import argparse,json,os,stat,hashlib,shutil,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .npy_archive_codec import encode,decode,sha_bytes
from .verify_npy_archive import independently_restore
from .probe_lossless_archival import inspect
ROOT=Path.cwd()
OUT=Path('results/thrml_spw/lossless_archive')
def safe_path(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Relative project path required')
    full=root/p
    if full.is_symlink() or not full.resolve().is_relative_to(root.resolve()):raise ValueError('Symlink/outside-workspace evidence excluded')
    return full
def sync_dir(path):
    fd=os.open(path,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def durable_bytes(path,data,mode=0o644):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    if temp.is_symlink():raise ValueError('Unsafe temporary link')
    try:
        with temp.open('wb') as f:f.write(data);f.flush();os.fchmod(f.fileno(),mode);os.fsync(f.fileno())
        os.replace(temp,path);sync_dir(path.parent)
    except BaseException:
        if temp.exists():temp.unlink()
        raise
def durable_json(path,value):durable_bytes(path,(json.dumps(value,indent=2)+'\n').encode())
def archive_one(row,root=ROOT,out=OUT):
    root=Path(root);out=Path(out);source=safe_path(root,row['original_path']);archive=safe_path(root,row['archive_path']);record=out/'records'/f"{row['id']:05d}.json"
    if archive.exists():
        raw,meta=independently_restore(archive);assert sha_bytes(raw)==row['original_sha256'] and meta['original_path']==row['original_path'];blob_hash=sha_bytes(archive.read_bytes())
    else:
        if not source.exists():raise RuntimeError(f'Missing raw evidence and archive: {source}')
        before=source.stat();assert source.is_file() and before.st_nlink==1;raw=source.read_bytes();assert len(raw)==row['original_bytes'] and sha_bytes(raw)==row['original_sha256'];blob,meta=encode(raw,row['original_path'],row['mode'],row['mtime_ns'])
        if len(blob)>=len(raw):return {'status':'not_smaller','id':row['id'],'original_path':row['original_path']}
        temp=archive.with_name(archive.name+'.tmp')
        if temp.exists():
            assert not temp.is_symlink();temp.unlink();sync_dir(temp.parent)
        if shutil.disk_usage(root).free<len(blob)+16_000_000:raise RuntimeError('Insufficient temporary archive capacity; original retained')
        durable_bytes(archive,blob,0o444);blob_hash=sha_bytes(blob);restored,check=independently_restore(archive);assert restored==raw and check==meta;again,_=decode(archive.read_bytes());assert again==raw
        current=source.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)
    assert meta['original_sha256']==row['original_sha256'] and meta['original_bytes']==row['original_bytes'];result={**row,'status':'archived','archive_sha256':blob_hash,'archive_bytes':archive.stat().st_size,'original_bytes_verified_on_disk':True,'independent_decoder_exact':True,'raw_removed':False}
    # Record is durable while both representations still exist.
    durable_json(record,result)
    if source.exists():
        assert not source.is_symlink() and source.stat().st_nlink==1 and sha_bytes(source.read_bytes())==row['original_sha256'];source.unlink();sync_dir(source.parent)
    result['raw_removed']=True;durable_json(record,result);return result
def restore_one(record,root=ROOT,release_archive=False):
    root=Path(root);r=json.loads(Path(record).read_text());source=safe_path(root,r['original_path']);archive=safe_path(root,r['archive_path'])
    if not archive.exists():
        assert source.exists() and sha_bytes(source.read_bytes())==r['original_sha256'];return r
    blob=archive.read_bytes();assert sha_bytes(blob)==r['archive_sha256'];raw,meta=independently_restore(archive);assert sha_bytes(raw)==r['original_sha256'] and meta['original_path']==r['original_path']
    if source.exists():assert sha_bytes(source.read_bytes())==r['original_sha256']
    else:
        if shutil.disk_usage(root).free<len(raw)+16_000_000:raise RuntimeError('Insufficient temporary restore capacity; archive retained')
        durable_bytes(source,raw,r['mode']);os.utime(source,ns=(r['mtime_ns'],r['mtime_ns']))
        with source.open('rb') as f:os.fsync(f.fileno())
        sync_dir(source.parent)
    assert sha_bytes(source.read_bytes())==r['original_sha256'];r.update(status='restored',raw_removed=False);durable_json(record,r)
    if release_archive:archive.unlink();sync_dir(archive.parent)
    return r
def prepare():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'plan.json').exists();input_index=Path('results/thrml_spw/archival_feasibility/input_artifact_index.json');index=json.loads(input_index.read_text());probe=json.loads(Path('results/thrml_spw/archival_feasibility/probe.json').read_text());assert sha_bytes(input_index.read_bytes())==probe['index_sha256'];entries=[]
    for name,info in index.items():
        p=Path(name)
        if p.suffix!='.npy' or 'runs' not in p.parts or p.is_symlink() or info['bytes']<1000000:continue
        s=p.stat()
        if s.st_nlink!=1:continue
        header=inspect(name)
        if header is None or header['payload_bytes']+header['header_bytes']!=info['bytes']:continue
        assert p.resolve().is_relative_to(ROOT.resolve());entries.append({'original_path':name,'archive_path':str(OUT/'payloads'/Path(name+'.spwnpy')),'original_bytes':info['bytes'],'original_sha256':info['sha256'],'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns})
    # Prefer diagnostic references, then the largest gains/files; no scientific selection.
    entries.sort(key=lambda r:('conditional_reference' not in r['original_path'],-r['original_bytes'],r['original_path']))
    for i,r in enumerate(entries):r['id']=i
    assert len(entries)==probe['candidate_files'] and sum(r['original_bytes'] for r in entries)==probe['candidate_bytes'];names=['lossless_evidence_archive.py','npy_archive_codec.py','verify_npy_archive.py','probe_lossless_archival.py'];sources={str(Path('research/thrml_spw')/n):sha_bytes((Path('research/thrml_spw')/n).read_bytes()) for n in names};plan={'format':'spw-lossless-evidence-archive-plan-v1','input_index':str(input_index),'input_index_sha256':probe['index_sha256'],'source_sha256':sources,'entries':entries,'goal':'Obtain actual free capacity>=10.2GB for frozen neural confirmation; every original byte preserved and recoverable. Original paths require explicit restoration before historical commands.','target_free_bytes':10_200_000_000};durable_json(OUT/'plan.json',plan);print(json.dumps({'prepared':True,'candidate_files':len(entries),'candidate_raw_bytes':probe['candidate_bytes']}))
def migrate():
    plan=json.loads((OUT/'plan.json').read_text())
    for name,h in plan['source_sha256'].items():assert sha_bytes(Path(name).read_bytes())==h
    results=[];target=plan['target_free_bytes'];entries=iter(plan['entries']);exhausted=False
    with ThreadPoolExecutor(max_workers=2) as pool:
        while shutil.disk_usage(ROOT).free<target:
            batch=[];width=2 if shutil.disk_usage(ROOT).free>1_000_000_000 else 1
            for _ in range(width):
                try:batch.append(next(entries))
                except StopIteration:exhausted=True;break
            if not batch:break
            results.extend(pool.map(archive_one,batch));free=shutil.disk_usage(ROOT).free
            progress={'processed_this_invocation':len(results),'archived_this_invocation':sum(r['status']=='archived' for r in results),'free_bytes':free,'target_free_bytes':target,'target_reached':free>=target};durable_json(OUT/'progress.json',progress)
            if len(results)%20<width:print(json.dumps(progress),flush=True)
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'records').glob('*.json'))];r={'status':'migration_complete' if shutil.disk_usage(ROOT).free>=target else 'candidates_exhausted','target_reached':shutil.disk_usage(ROOT).free>=target,'actual_free_bytes':shutil.disk_usage(ROOT).free,'archived_records':sum(v['status']=='archived' for v in rows),'original_bytes_preserved':sum(v['original_bytes'] for v in rows),'archive_bytes':sum(v['archive_bytes'] for v in rows),'file_bytes_saved':sum(v['original_bytes']-v['archive_bytes'] for v in rows),'source_sha256':sha_bytes(Path(__file__).read_bytes()),'scope':'Reversible physical-file representation migration only. No scientific value, old source, recorded model score, numerical failure or quality criterion changed.'};durable_json(OUT/'migration.json',r);print(json.dumps(r,indent=2))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','migrate','restore']);parser.add_argument('--study');parser.add_argument('--original');parser.add_argument('--release-archive',action='store_true');args=parser.parse_args();OUT.mkdir(exist_ok=True)
    with (OUT/'operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='prepare':prepare()
        elif args.action=='migrate':migrate()
        else:
            if not args.study and not args.original:raise ValueError('Specify --study or --original; no implicit mass restoration')
            count=0
            for p in sorted((OUT/'records').glob('*.json')):
                r=json.loads(p.read_text());parts=Path(r['original_path']).parts
                if args.original and r['original_path']!=args.original:continue
                if args.study and (len(parts)<3 or parts[2]!=args.study):continue
                restore_one(p,release_archive=args.release_archive);count+=1
            print(json.dumps({'restored_records':count,'release_archive':args.release_archive}))
if __name__=='__main__':main()
