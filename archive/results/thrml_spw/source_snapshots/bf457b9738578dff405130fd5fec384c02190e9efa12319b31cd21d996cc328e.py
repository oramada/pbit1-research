"""Finite-controller calibration using only observed calibration counts."""
import numpy as np


def estimate_command(counts, samples, nominal_fields, probe_field):
    counts = np.asarray(counts)
    nominal_fields = np.asarray(nominal_fields)
    if not isinstance(samples, int) or samples < 1 or counts.shape != nominal_fields.shape:
        raise ValueError('Positive sample count and matching arrays required')
    if counts.dtype.kind not in 'iu' or np.any(counts < 0) or np.any(counts > samples):
        raise ValueError('Integer binomial counts required')
    if not np.isfinite(nominal_fields).all() or not np.isfinite(probe_field):
        raise ValueError('Finite known controls required')
    # Jeffreys smoothing keeps both all-zero and all-one observations finite.
    frequency = (counts.astype(np.float64)+.5)/(samples+1)
    offset = .5*np.log(frequency/(1-frequency))-float(probe_field)
    requested = nominal_fields.astype(np.float64)-offset
    codes = np.rint((np.clip(requested, -4, 4)+4)*(255/8)).astype(np.uint8)
    return codes, offset


def decode_command(codes):
    return (-4+np.asarray(codes, dtype=np.float64)*(8/255)).astype(np.float32)
