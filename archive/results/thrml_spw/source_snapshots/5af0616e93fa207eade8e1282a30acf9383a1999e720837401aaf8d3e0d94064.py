"""Reversibly share byte-identical closed-study arrays; never remove evidence paths."""
import argparse,hashlib,json,os,shutil,stat,uuid
from pathlib import Path
from collections import defaultdict
OUT=Path('results/thrml_spw/storage_deduplication')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(1024*1024),b''):h.update(value)
    return h.hexdigest()
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def checked_path(name):
    p=Path(name);assert name.startswith('results/thrml_spw/') and p.suffix=='.npy' and not p.is_symlink() and p.is_file();assert len(p.parts)>3
    assert (Path(*p.parts[:3])/'manifest.json').exists(),p
    return p

def plan():
    index=json.loads(Path('results/thrml_spw/artifact_index.json').read_text());groups=defaultdict(list)
    for name,r in index.items():
        if name.endswith('.npy') and r['bytes']>=1024*1024 and name.startswith('results/thrml_spw/') and ('_inputs.npy' in name or '_conditional_reference.npy' in name or '/teacher/' in name):
            p=Path(name)
            if (Path(*p.parts[:3])/'manifest.json').exists():groups[(r['sha256'],r['bytes'])].append(name)
    result=[]
    for (digest,size),names in sorted(groups.items()):
        if len(names)<2:continue
        entries=[]
        for name in names:
            p=checked_path(name);s=p.stat();assert s.st_size==size and sha(p)==digest
            entries.append({'path':name,'original_mode':stat.S_IMODE(s.st_mode),'original_mtime_ns':s.st_mtime_ns,'original_atime_ns':s.st_atime_ns,'original_inode':s.st_ino,'original_device':s.st_dev})
        assert len({r['original_device'] for r in entries})==1
        result.append({'sha256':digest,'bytes':size,'files':entries})
    return {'schema':'spw-content-identical-closed-array-hardlink-v1','all_files_content_verified_before_any_mutation':True,'groups':result,'scope':'Only deterministic input/reference arrays in previously completed studies. All original names and exact file bytes retained; no unique stochastic output removed. Shared canonical inodes set read-only to prevent accidental in-place edits. Atomic path replacement remains independent. Restore makes private copies and restores original modes/timestamps. Logical artifact hashes remain unchanged; physical allocation only is reduced.'}

def apply(p):
    # Recheck whole plan before even the first chmod/link.
    for g in p['groups']:
        for r in g['files']:assert sha(checked_path(r['path']))==g['sha256']
    total=0;linked=0
    for g in p['groups']:
        canonical=checked_path(g['files'][0]['path']);os.chmod(canonical,0o444)
        for r in g['files'][1:]:
            dest=checked_path(r['path'])
            if not os.path.samefile(canonical,dest):
                temp=dest.with_name(dest.name+'.dedup-'+uuid.uuid4().hex)
                os.link(canonical,temp);os.replace(temp,dest);linked+=1;total+=g['bytes']
            assert sha(dest)==g['sha256'] and os.path.samefile(canonical,dest)
        save(OUT/'progress.json',{'last_verified_group_sha256':g['sha256'],'new_paths_linked':linked,'logical_duplicate_bytes_shared':total})
    result={'passed':True,'groups':len(p['groups']),'new_paths_linked':linked,'logical_duplicate_bytes_shared':total,'every_original_path_hash_verified_after_mutation':True,'read_only_shared_inodes':True,'source_sha256':sha(__file__),'scope':p['scope']};save(OUT/'evidence_audit.json',result);return result

def restore(p):
    restored=0
    for g in p['groups']:
        for r in g['files']:
            dest=checked_path(r['path']);assert sha(dest)==g['sha256'];temp=dest.with_name(dest.name+'.restore-'+uuid.uuid4().hex);shutil.copyfile(dest,temp);os.chmod(temp,r['original_mode']);os.utime(temp,ns=(r['original_atime_ns'],r['original_mtime_ns']));os.replace(temp,dest);assert sha(dest)==g['sha256'];restored+=1
    result={'restored_paths':restored,'all_bytes_unchanged':True};save(OUT/'restoration.json',result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');parser.add_argument('--restore',action='store_true');args=parser.parse_args();assert not(args.apply and args.restore);OUT.mkdir(exist_ok=True);path=OUT/'plan.json'
    if path.exists():p=json.loads(path.read_text())
    else:p=plan();save(path,p)
    if args.restore:result=restore(p)
    elif args.apply:
        previous=OUT/'evidence_audit.json'
        if previous.exists():
            old=json.loads(previous.read_text());assert old['passed']
            for group in p['groups']:
                canonical=checked_path(group['files'][0]['path'])
                for entry in group['files']:
                    dest=checked_path(entry['path']);assert os.path.samefile(canonical,dest) and sha(dest)==group['sha256'] and dest.stat().st_mode&0o222==0
            result={'cached_complete':True,'new_path_mutations':0,'preserved_evidence_audit_sha256':sha(previous)}
        else:result=apply(p)
    else:result={'verified_groups':len(p['groups']),'potential_duplicate_bytes':sum(g['bytes']*(len(g['files'])-1) for g in p['groups']),'mutations':0}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
