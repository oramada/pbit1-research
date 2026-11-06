"""Public-state replay with independent learned-mean/range/word construction."""
import numpy as np
import jax.numpy as jnp
from results.thrml_spw.fashion_cnn_native.compact.recover import public_draw, pooling
from results.thrml_spw.fashion_cnn_native.cohort.replay import patches


def parameters(model, original, prefix):
    w = original[prefix+'_weight'].astype(float)
    no, ni = len(w), w.size//len(w)
    c = np.array([(max(row.ravel())+min(row.ravel()))/2 for row in w])
    r = np.array([(max(row.ravel())-min(row.ravel()))/2 for row in w])
    np.testing.assert_array_equal(c, model[prefix+'_center'])
    np.testing.assert_array_equal(r, model[prefix+'_radius'])
    commands = model[prefix+'_commands']
    fields = (-4+commands.astype(float)*8/255).astype(np.float32)
    np.testing.assert_array_equal(fields, model[prefix+'_fields'])
    probability = 1/(1+np.exp(-2*fields.astype(float)))
    mean = (c-r)[:, None]+2*r[:, None]*probability
    np.testing.assert_array_equal(mean.reshape(model[prefix+'_weight'].shape), model[prefix+'_weight'])
    code = np.empty((ni, no), np.uint8)
    for output in range(no):
        code[:, output] = np.round(np.minimum(1, np.maximum(0, probability[output] if r[output] else np.full(ni, .5)))*15)
    f = np.empty((ni, no, 4), np.float32)
    for bit in range(4):
        f[:, :, bit] = np.where((code >> bit) & 1, 4., -4.)
    return c, r, code, f


def recover(x, model, original, t, keys, enc):
    h = x.astype(float); chains, batch = h.shape[:2]; saved = {}
    for li, prefix in enumerate(['conv1', 'conv2', 'head']):
        c, r, code, f = parameters(model, original, prefix)
        ni, no = code.shape
        for key, actual in [('center', c), ('radius', r), ('codes', code), ('fields', f)]:
            np.testing.assert_array_equal(enc[li][key], actual)
        raw = np.asarray(public_draw(ni*no*4, t)(keys[li], jnp.asarray(np.broadcast_to(f.ravel(), (chains, ni*no*4)).copy())))
        bit_counts = raw.sum(1, dtype=np.uint32).reshape(chains, ni, no, 4)
        wc = sum(bit_counts[:, :, :, bit]*np.uint32(2**bit) for bit in range(4))
        col = wc.sum(1, dtype=np.uint32).astype(np.int32)
        saved[f'column_{li}'] = col; span = float(model['activation_span'][li])
        if li < 2:
            height, width = h.shape[-2:]
            parts = [patches(h[ch], span) for ch in range(chains)]
            codes = np.stack([v[0] for v in parts]); mask = np.stack([v[1] for v in parts])
            g = np.einsum('cpi,cio->cpo', codes.astype(np.uint32), wc, dtype=np.uint32, optimize=False)
            m = np.einsum('cpi,cio->cpo', mask.astype(np.uint32), wc, dtype=np.uint32, optimize=False)
            saved[f'gate_{li}'] = np.concatenate([g, m], axis=1).astype(np.int32)
            saved[f'codes_{li}'] = codes; saved[f'mask_{li}'] = mask; ys = []
            for ch in range(chains):
                sx = np.sum(mask[ch]*(0.+span*codes[ch].astype(float)/255), axis=1)
                sample = 0.*m[ch].astype(float)/(15*t)+(span/255)*g[ch].astype(float)/(15*t)
                y = sx[:, None]*(c-r)+2*r*sample+model[prefix+'_bias'].astype(float)
                ys.append(y.reshape(batch, height, width, no).transpose(0, 3, 1, 2))
            h = pooling(np.stack(ys))
        else:
            h = h.reshape(chains, batch, ni)
            codes = np.rint(255*np.clip(h/span, 0, 1)).astype(np.uint8)
            g = np.einsum('cbi,cio->cbo', codes.astype(np.uint32), wc, dtype=np.uint32, optimize=False).astype(np.int32)
            saved[f'codes_{li}'] = codes; saved[f'gate_{li}'] = g
            qx = 0.+span*codes.astype(float)/255
            sample = 0.*col[:, None, :].astype(float)/(15*t)+(span/255)*g.astype(float)/(15*t)
            h = qx.sum(-1)[:, :, None]*(c-r)+2*r*sample+model[prefix+'_bias'].astype(float)
        saved[f'output_{li}'] = h
    return saved
