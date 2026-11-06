"""Four-bit positional conversion of learned means with original ranges retained."""
import numpy as np


def encode(model):
    result = []
    for prefix in ['conv1', 'conv2', 'head']:
        w = model[prefix+'_weight'].astype(float)
        flat = w.reshape(len(w), -1)
        c, r = model[prefix+'_center'], model[prefix+'_radius']
        q = np.divide(flat-c[:, None], r[:, None], out=np.zeros_like(flat), where=r[:, None] > 0)/2+.5
        code = np.rint(np.clip(q, 0, 1)*15).astype(np.uint8).T.copy()
        bits = (code[:, :, None] >> np.arange(4)) & 1
        result.append({'center': c.copy(), 'radius': r.copy(), 'codes': code,
            'fields': np.where(bits, 4., -4.).astype(np.float32),
            'bias': model[prefix+'_bias'].astype(float)})
    return result


def mean_weights(model):
    epsilon = 1/(1+np.exp(8))
    result = []
    for prefix, e in zip(['conv1', 'conv2', 'head'], encode(model)):
        q = epsilon+(1-2*epsilon)*e['codes'].T/15
        w = (e['center']-e['radius'])[:, None]+2*e['radius'][:, None]*q
        result.append(w.reshape(model[prefix+'_weight'].shape))
    return result
