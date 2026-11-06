"""Lossless F32 byte-plane packing for research outputs, not model compression."""
import os
from pathlib import Path
import numpy as np
TAG='spw-f32-byteplanes-v1'
def save_output(path,value):
    path=Path(path);array=np.ascontiguousarray(value);assert array.dtype==np.dtype('<f4');planes=array.view(np.uint8).reshape(-1,4).T.copy();tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,tag=np.array(TAG),shape=np.array(array.shape,np.int64),byteplanes=planes)
    os.replace(tmp,path)
def load_output(path):
    path=Path(path)
    if path.suffix=='.npy':return np.load(path,mmap_mode='r')
    with np.load(path,allow_pickle=False) as a:
        assert a['tag'].item()==TAG;shape=tuple(int(v) for v in a['shape']);p=a['byteplanes'];assert p.dtype==np.uint8 and p.shape==(4,int(np.prod(shape)));return p.T.copy().reshape(-1).view('<f4').reshape(shape)
