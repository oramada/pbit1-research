"""Public-state causal recovery with independently reconstructed learned fields."""
import numpy as np
import jax.numpy as jnp

from results.thrml_spw.fashion_cnn_native.compact.recover import public_draw, im2col_independent, pooling


def recover(x, exported, original, t, keys):
    h = np.asarray(x, float)
    chains, batch = h.shape[:2]
    saved = {}
    for li, prefix in enumerate(['conv1', 'conv2', 'head']):
        assert float(exported['activation_lower'][li]) == 0.
        span = float(exported['activation_span'][li])
        source = original[prefix+'_weight'].astype(float)
        no, ni = source.shape[0], source.size//source.shape[0]
        centers, radii, codes = [], [], []
        for output in range(no):
            v = source[output].ravel()
            centers.append((max(v)+min(v))/2)
            radii.append((max(v)-min(v))/2)
            latent = exported[prefix+'_latent_fields'][output]
            assert np.max(abs(latent)) <= 4
            codes.append(np.round((latent+4)*255/8).astype(np.uint8))
        c, r = np.array(centers), np.array(radii)
        np.testing.assert_array_equal(c, exported[prefix+'_center'])
        np.testing.assert_array_equal(r, exported[prefix+'_radius'])
        np.testing.assert_array_equal(codes, exported[prefix+'_commands'])
        np.testing.assert_array_equal(original[prefix+'_bias'], exported[prefix+'_bias'])
        fields = np.array([[-4+float(code)*8/255 for code in row] for row in codes], np.float32).T.copy()
        np.testing.assert_array_equal(fields.T, exported[prefix+'_fields'])
        raw = np.asarray(public_draw(ni*no, t)(keys[li], jnp.asarray(np.broadcast_to(fields.ravel(), (chains, ni*no)).copy())))
        counts = raw.sum(1, dtype=np.uint32).reshape(chains, ni, no)
        columns = counts.sum(1, dtype=np.uint32).astype(np.int32)
        saved[f'column_{li}'] = columns
        if li < 2:
            height, width = h.shape[-2:]
            parts = [im2col_independent(h[ch], span) for ch in range(chains)]
            codes = np.stack([v[0] for v in parts])
            mask = np.stack([v[1] for v in parts])
            gates = np.einsum('cpi,cio->cpo', codes.astype(np.uint32), counts, dtype=np.uint32, optimize=False)
            masks = np.einsum('cpi,cio->cpo', mask.astype(np.uint32), counts, dtype=np.uint32, optimize=False)
            saved[f'gate_{li}'] = np.concatenate([gates, masks], axis=1).astype(np.int32)
            saved[f'codes_{li}'] = codes
            saved[f'mask_{li}'] = mask
            output = []
            for ch in range(chains):
                sum_x = np.sum(mask[ch]*(0.+span*codes[ch].astype(float)/255), axis=1)
                sampled = 0.*masks[ch].astype(float)/t+(span/255)*gates[ch].astype(float)/t
                y = sum_x[:, None]*(c-r)+2*r*sampled+original[prefix+'_bias'].astype(float)
                output.append(y.reshape(batch, height, width, no).transpose(0, 3, 1, 2))
            h = pooling(np.stack(output))
        else:
            h = h.reshape(chains, batch, ni)
            codes = np.rint(255*np.clip(h/span, 0, 1)).astype(np.uint8)
            gates = np.einsum('cbi,cio->cbo', codes.astype(np.uint32), counts, dtype=np.uint32, optimize=False).astype(np.int32)
            saved[f'gate_{li}'] = gates
            saved[f'codes_{li}'] = codes
            qx = 0.+span*codes.astype(float)/255
            sampled = 0.*columns[:, None, :].astype(float)/t+(span/255)*gates.astype(float)/t
            h = qx.sum(-1)[:, :, None]*(c-r)+2*r*sampled+original[prefix+'_bias'].astype(float)
        saved[f'output_{li}'] = h
    return saved
