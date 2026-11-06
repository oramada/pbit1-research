"""Read-only byte-plane compression feasibility; never alters original evidence."""
import json,hashlib,io,zlib,re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
OUT=Path('results/thrml_spw/archival_feasibility')
def inspect(name):
    with Path(name).open('rb') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
        else:return None
        offset=f.tell()
    if dtype.kind!='f' or dtype.itemsize not in [4,8]:return None
    return {'itemsize':dtype.itemsize,'header_bytes':offset,'payload_bytes':int(np.prod(shape))*dtype.itemsize}
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'probe.json').exists();index_path=Path('results/thrml_spw/artifact_index.json');index=json.loads(index_path.read_text());groups={};seen=set()
    for name,row in index.items():
        p=Path(name)
        if p.suffix!='.npy' or 'runs' not in p.parts or p.is_symlink() or row['bytes']<1000000:continue
        stat=p.stat()
        if stat.st_nlink!=1 or (stat.st_dev,stat.st_ino) in seen:continue
        seen.add((stat.st_dev,stat.st_ino));info=inspect(name)
        if info is None or info['payload_bytes']+info['header_bytes']!=row['bytes']:continue
        kind=re.sub(r'^stage\d+_','',p.stem);group='/'.join(p.parts[:3])+'/'+kind;groups.setdefault(group,[]).append({'path':name,'sha256':row['sha256'],'bytes':row['bytes'],**info})
    rng=np.random.default_rng(2026118100);selected=[]
    for group,files in sorted(groups.items()):
        chosen=rng.choice(len(files),min(2,len(files)),replace=False)
        for i in chosen:selected.append({'group':group,**files[int(i)]})
    def probe(row):
        b=Path(row['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256'];off,size=row['header_bytes'],row['itemsize'];payload=np.frombuffer(b,dtype=np.uint8,offset=off).reshape(-1,size);planes=payload.T.copy().tobytes();compressed=zlib.compress(planes,6);restored=b[:off]+np.frombuffer(zlib.decompress(compressed),np.uint8).reshape(size,-1).T.copy().tobytes();assert restored==b;return {**row,'probe_archive_bytes':len(compressed)+off+4096,'exact_original_file_roundtrip':True,'payload_plane_compressed_bytes':len(compressed)}
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(probe,selected))
    strata=[]
    for name,files in sorted(groups.items()):
        sample=[r for r in rows if r['group']==name];ratio=sum(r['probe_archive_bytes'] for r in sample)/sum(r['bytes'] for r in sample);raw=sum(r['bytes'] for r in files);worst=max(r['probe_archive_bytes']/r['bytes'] for r in sample);strata.append({'group':name,'files':len(files),'raw_bytes':raw,'samples':len(sample),'sample_archive_ratio':ratio,'sample_based_estimated_saved_bytes':raw*(1-ratio),'sample_max_ratio_saved_bytes':raw*(1-worst)})
    r={'read_only':True,'original_files_modified':0,'exact_roundtrip_samples':len(rows),'candidate_files':sum(g['files'] for g in strata),'candidate_bytes':sum(g['raw_bytes'] for g in strata),'sample_based_estimated_saved_bytes':sum(g['sample_based_estimated_saved_bytes'] for g in strata),'sample_max_ratio_saved_bytes':sum(g['sample_max_ratio_saved_bytes'] for g in strata),'index_sha256':hashlib.sha256(index_path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'strata':strata,'samples':rows,'scope':'Read-only feasibility, single-link completed historical run arrays only. Exact NPY bytes round-trip in RAM, including headers/F32/F64 and signed/NaN bit patterns present. No actual evidence archived or deleted. Savings are sample extrapolations with4KiB/file provision, not guaranteed capacity. Any migration needs a durable reversible archive/restore manifest and changed historical replay instructions before mutation.'};(OUT/'probe.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['strata','samples']},indent=2))
if __name__=='__main__':main()
