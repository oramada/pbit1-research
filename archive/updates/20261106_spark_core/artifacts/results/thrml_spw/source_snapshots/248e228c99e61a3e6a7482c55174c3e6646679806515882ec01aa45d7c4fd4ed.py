"""Larger-budget derivative of the frozen allocated native operator; no reconstruction."""
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
        assert 1 <= t <= 4096 and float(model['activation_lower'][li]) == 0.
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

