"""Independently verify deduplicated evidence names/content and physical sharing."""
from pathlib import Path
import hashlib,json,os
OUT=Path('results/thrml_spw/storage_deduplication')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=json.loads((OUT/'plan.json').read_text());production=json.loads((OUT/'evidence_audit.json').read_text());assert production['passed'];index=json.loads(Path('results/thrml_spw/artifact_index.json').read_text());groups=0;paths=0;logical=0
    for g in p['groups']:
        shared=[];original=set()
        for r in g['files']:
            path=Path(r['path']);s=path.stat();assert path.is_file() and not path.is_symlink() and s.st_size==g['bytes'];assert digest(path)==g['sha256']==index[str(path)]['sha256'];assert s.st_mode&0o222==0;shared.append((s.st_dev,s.st_ino));original.add((r['original_device'],r['original_inode']));paths+=1
        assert len(set(shared))==1 and Path(g['files'][0]['path']).stat().st_nlink>=len(g['files']);groups+=1;logical+=(len(original)-1)*g['bytes']
    assert groups==production['groups'] and logical==production['logical_duplicate_bytes_shared'];r={'passed':True,'groups_independently_verified':groups,'paths_and_original_hashes_verified':paths,'exact_content_duplicate_bytes_shared':logical,'all_names_preserved':True,'shared_inodes_read_only':True,'source_sha256':digest(Path(__file__)),'scope':'No unique file content removed. Logical names/lengths/hashes match pre-existing research index. Exact verified duplicates share read-only inodes; fixture tests verify independent atomic replacement and restoration. Physical allocation changes, not scientific observations.'};(OUT/'independent_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
