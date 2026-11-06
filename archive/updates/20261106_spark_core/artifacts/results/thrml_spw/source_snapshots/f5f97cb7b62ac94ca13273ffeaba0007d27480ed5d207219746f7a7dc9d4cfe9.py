"""Two-probe gain/offset identification from observed binomial counts only."""
import numpy as np


def fit_two_probe(count_low, count_high, samples_each, field_low, field_high):
    low, high = np.asarray(count_low), np.asarray(count_high)
    if not isinstance(samples_each, int) or samples_each < 1 or low.shape != high.shape:
        raise ValueError('Matching counts and positive integer sample budget required')
    if any(c.dtype.kind not in 'iu' or np.any(c < 0) or np.any(c > samples_each) for c in [low, high]):
        raise ValueError('Integer counts within the sample budget required')
    if not np.isfinite([field_low, field_high]).all() or field_high <= field_low:
        raise ValueError('Ordered finite probe fields required')
    # Jeffreys smoothing bounds the inferred logit for finite observations.
    p_low = (low.astype(np.float64)+.5)/(samples_each+1)
    p_high = (high.astype(np.float64)+.5)/(samples_each+1)
    u_low = .5*np.log(p_low/(1-p_low))
    u_high = .5*np.log(p_high/(1-p_high))
    raw_gain = (u_high-u_low)/(field_high-field_low)
    offset = (u_low+u_high)/2-raw_gain*(field_low+field_high)/2
    # A declared estimator guard, not a measured physical device specification.
    gain = np.clip(raw_gain, .25, 4.)
    return gain, offset, raw_gain


def corrected_codes(nominal_fields, estimated_gain, estimated_offset):
    nominal = np.asarray(nominal_fields, dtype=np.float64)
    gain, offset = np.asarray(estimated_gain), np.asarray(estimated_offset)
    if gain.shape != nominal.shape or offset.shape != nominal.shape:
        raise ValueError('Matching per-device arrays required')
    if not all(np.isfinite(a).all() for a in [nominal, gain, offset]) or np.any(gain <= 0):
        raise ValueError('Finite parameters and positive estimated gain required')
    requested = (nominal-offset)/gain
    code = np.rint((np.clip(requested, -4, 4)+4)*255/8).astype(np.uint8)
    return code, requested
