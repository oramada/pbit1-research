"""Frozen resource and condition rules shared by the neural confirmation."""
import shutil,hashlib,json,os
from pathlib import Path
OUT=Path('results/thrml_spw/fine_range_confirmation')
MINIMUM_FREE_BYTES=10_000_000_000
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def atomic_npy(path,value):
    import numpy as np
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.save(f,value)
    os.replace(tmp,path)
def require_capacity(path=OUT):
    free=shutil.disk_usage(path).free
    if free<MINIMUM_FREE_BYTES:raise RuntimeError(f'Full neural confirmation requires at least {MINIMUM_FREE_BYTES} persistent free bytes; available={free}. No model scoring or native sampling started.')
def condition(protocol,name):
    matches=[c for c in protocol['conditions'] if c['name']==name]
    if len(matches)!=1:raise ValueError('Unknown or duplicate confirmation condition')
    c=matches[0]
    if (name,c['group'],c['samples']) not in [('coarse',128,1024),('fine',16,512)]:raise ValueError('Frozen range/sample condition changed')
    return c
