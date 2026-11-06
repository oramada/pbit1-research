"""Per-layer sample counts; native counters and independent public-state replay."""
import numpy as np
import jax
import jax.numpy as jnp
from results.thrml_spw.fashion_cnn_native.inference.native import gate, pool
from results.thrml_spw.fashion_cnn_native.inference.fast_native import vector_patches
from results.thrml_spw.integer_activation_native.gated_layer import activation_codes, readout
from results.thrml_spw.convolution_native.operator import decode
from results.thrml_spw.fashion_cnn_native.cohort.replay import patches
from results.thrml_spw.fashion_cnn_native.compact.recover import public_draw, independent_parameters, pooling


def key_bank(phase, model, condition, seed, layer, first, count):
    key = jax.random.key(1707520)
    for value in [phase, model, condition, seed, layer]:
        key = jax.random.fold_in(key, value)
    return jax.vmap(lambda bank: jax.random.fold_in(key, bank))(jnp.arange(first, first+count, dtype=jnp.uint32))


def forward(x, model, enc, budgets, keys):
    h = np.asarray(x, float)
    chains, batch = h.shape[:2]
    saved = {}
    for li, e in enumerate(enc):
        t = int(budgets[li])
        assert 1 <= t <= 1024 and float(model['activation_lower'][li]) == 0.
        span = float(model['activation_span'][li])
        ni, no = e['fields'].shape
        if li < 2:
            prepared = [vector_patches(h[ch], span) for ch in range(chains)]
            a = np.stack([v[0] for v in prepared])
            codes = np.stack([v[1] for v in prepared])
            mask = np.stack([v[2] for v in prepared])
        else:
            h = h.reshape(chains, batch, ni)
            codes = activation_codes(h, 0., span)
            a = codes
        fields = np.broadcast_to(e['fields'].ravel(), (chains, ni*no)).copy()
        g, col = map(np.asarray, gate(ni, no, a.shape[1]).kernel(t)(keys[li], jnp.asarray(fields), jnp.asarray(a)))
        saved[f'gate_{li}'], saved[f'column_{li}'], saved[f'codes_{li}'] = g, col, codes
        if li < 2:
            saved[f'mask_{li}'] = mask
            y = np.concatenate([decode(g[ch:ch+1], prepared[ch][3], codes[ch], mask[ch], t, 0., span, e['center'], e['radius'], e['bias']) for ch in range(chains)], axis=0)
            h = pool(np.maximum(y, 0))
        else:
            qx = 0.+span*codes.astype(float)/255
            sampled = readout(g, col[:, None, :], t, 0., span, 0)
            h = qx.sum(-1)[:, :, None]*(e['center']-e['radius'])+2*e['radius']*sampled+e['bias']
        saved[f'output_{li}'] = h
    return saved


def recover(x, model, budgets, keys):
    h = np.asarray(x, float)
    chains, batch = h.shape[:2]
    saved = {}
    for li, prefix in enumerate(['conv1', 'conv2', 'head']):
        t = int(budgets[li])
        assert float(model['activation_lower'][li]) == 0.
        span = float(model['activation_span'][li])
        center, radius, fields = independent_parameters(model, prefix)
        ni, no = fields.shape
        states = np.asarray(public_draw(ni*no, t)(keys[li], jnp.asarray(np.broadcast_to(fields.ravel(), (chains, ni*no)).copy())))
        wc = states.sum(1, dtype=np.uint32).reshape(chains, ni, no)
        col = wc.sum(1, dtype=np.uint32).astype(np.int32)
        saved[f'column_{li}'] = col
        if li < 2:
            height, width = h.shape[-2:]
            parts = [patches(h[ch], span) for ch in range(chains)]
            codes = np.stack([v[0] for v in parts])
            mask = np.stack([v[1] for v in parts])
            g = np.einsum('cpi,cio->cpo', codes.astype(np.uint32), wc, dtype=np.uint32, optimize=False)
            m = np.einsum('cpi,cio->cpo', mask.astype(np.uint32), wc, dtype=np.uint32, optimize=False)
            saved[f'gate_{li}'] = np.concatenate([g, m], axis=1).astype(np.int32)
            saved[f'codes_{li}'], saved[f'mask_{li}'] = codes, mask
            outputs = []
            for ch in range(chains):
                sx = np.sum(mask[ch]*(0.+span*codes[ch].astype(float)/255), axis=1)
                sampled = 0.*m[ch].astype(float)/t+(span/255)*g[ch].astype(float)/t
                y = sx[:, None]*(center-radius)+2*radius*sampled+model[prefix+'_bias'].astype(float)
                outputs.append(y.reshape(batch, height, width, no).transpose(0, 3, 1, 2))
            h = pooling(np.stack(outputs))
        else:
            h = h.reshape(chains, batch, ni)
            codes = np.rint(255*np.clip(h/span, 0, 1)).astype(np.uint8)
            g = np.einsum('cbi,cio->cbo', codes.astype(np.uint32), wc, dtype=np.uint32, optimize=False).astype(np.int32)
            saved[f'gate_{li}'], saved[f'codes_{li}'] = g, codes
            qx = 0.+span*codes.astype(float)/255
            sampled = 0.*col[:, None, :].astype(float)/t+(span/255)*g.astype(float)/t
            h = qx.sum(-1)[:, :, None]*(center-radius)+2*radius*sampled+model[prefix+'_bias'].astype(float)
        saved[f'output_{li}'] = h
    return saved
