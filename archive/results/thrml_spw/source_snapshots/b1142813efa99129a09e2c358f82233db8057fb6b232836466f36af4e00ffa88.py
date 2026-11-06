"""Frozen serial two-layer network clock, including actual calibration windows."""
import numpy as np
SEGMENT=64
SAMPLES_PER_LAYER=1024
LAYER_SEGMENTS=SAMPLES_PER_LAYER//SEGMENT
PROBE_SEGMENTS=32
CAL_SEGMENTS=64
BATCHES=313
REFRESH_BATCHES=32
WINDOWS=(BATCHES+REFRESH_BATCHES-1)//REFRESH_BATCHES
TOTAL_SEGMENTS=CAL_SEGMENTS*WINDOWS+BATCHES*2*LAYER_SEGMENTS
RATES=(0.,.01,.04) # field-offset SD per 1024 native sweeps


def calibration_start(window):
    if not 0<=window<WINDOWS:raise ValueError('Window outside frozen protocol')
    return window*(CAL_SEGMENTS+REFRESH_BATCHES*2*LAYER_SEGMENTS)


def batch_start(batch):
    if not 0<=batch<BATCHES:raise ValueError('Batch outside frozen protocol')
    return batch*2*LAYER_SEGMENTS+(batch//REFRESH_BATCHES+1)*CAL_SEGMENTS


def standard_path(model,device,size):
    path=np.empty((TOTAL_SEGMENTS+1,size),dtype=np.float64);path[0]=0
    rng=np.random.default_rng(2026110500+100*model+device)
    rng.standard_normal((TOTAL_SEGMENTS,size),out=path[1:]);np.cumsum(path,axis=0,out=path)
    return path/np.sqrt(SAMPLES_PER_LAYER//SEGMENT)


def static_device(model,device,size):
    noise=np.random.default_rng(2026110400+100*model+device).standard_normal((2,size))
    return np.exp(.2*noise[1]-.02),.2*noise[0]


def field_schedule(gain,offset,path,indices,rate,command):
    # Retain F64 device/clock arithmetic until the native field is emitted as F32.
    return (gain*command+offset+rate*path[indices]).astype(np.float32)
