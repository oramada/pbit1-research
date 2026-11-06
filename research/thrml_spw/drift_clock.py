"""Piecewise Brownian field drift on an explicit native-sweep clock."""
import numpy as np

SEGMENT_SAMPLES=64
INFERENCE_SAMPLES=512
CALIBRATION_SAMPLES=4096
AGES=(0,8,32)
QUERY_SEGMENTS=INFERENCE_SAMPLES//SEGMENT_SAMPLES
CALIBRATION_SEGMENTS=CALIBRATION_SAMPLES//SEGMENT_SAMPLES
PROBE_SEGMENTS=CALIBRATION_SEGMENTS//2
TOTAL_SEGMENTS=CALIBRATION_SEGMENTS+max(AGES)*QUERY_SEGMENTS+QUERY_SEGMENTS


def query_start(age):
    if age not in AGES:raise ValueError('Age must be in the frozen clock design')
    return CALIBRATION_SEGMENTS+age*QUERY_SEGMENTS


def calibration_start(age):
    return query_start(age)-CALIBRATION_SEGMENTS


def brownian_path(seed,size):
    if size<1:raise ValueError('Positive device count required')
    path=np.empty((TOTAL_SEGMENTS+1,size),dtype=np.float64)
    path[0]=0
    np.random.default_rng(2026103000+seed).standard_normal(
        (TOTAL_SEGMENTS,size),out=path[1:])
    np.cumsum(path,axis=0,out=path)
    return path


def offset_at(initial,path,segment,sigma):
    if not 0<=segment<=TOTAL_SEGMENTS or sigma<0:raise ValueError('Valid time and nonnegative drift SD required')
    return initial+(sigma/np.sqrt(QUERY_SEGMENTS))*path[segment]
