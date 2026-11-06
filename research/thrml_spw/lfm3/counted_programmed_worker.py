"""Software count-factorized finite-program worker with four independent tile lanes.

Preserves original keys and per-tile NumPy FP64 readout. Adds explicit per-weight
int32 simulator counters; does not represent the original physical direct-output
counter datapath or lower the required sampling budget. Not used by any
currently frozen full-model study. Simulator concurrency is not hardware speed.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from research.thrml_spw.native import NativeProbabilityBank
from .counted_bank import build_count_functions


def main():
    banks, functions = {}, {}
    fold_tiles = jax.jit(jax.vmap(jax.random.fold_in, in_axes=(None, 0)))
    print(json.dumps({'ready': True, 'devices': list(map(str, jax.devices()))}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if req.get('stop'):
            break
        begin = time.monotonic()
        with np.load(req['input']) as a:
            assert 'w' not in a.files
            xf, codes_all, lower_all, span_all = [a[n] for n in ['x', 'codes', 'lower', 'span']]
        assert codes_all.dtype == np.uint16 and codes_all.max() <= 4095
        assert lower_all.dtype == span_all.dtype == np.float32
        assert req['field_bits'] == req['activation_bits'] == 12
        assert req['mode'] in ['mean', 'native']
        depthwise = req['kind'] != 'linear'
        fanin, channels = codes_all.shape
        size = min(128, fanin); groups = fanin // size
        samples = int(req['samples'])
        assert fanin % size == 0 and samples > 0 and 2047 * size * samples < 2 ** 31
        assert lower_all.shape == span_all.shape == (groups, channels)
        scale = float(np.abs(xf).max()) / 2047 if np.any(xf) else 1.0
        x = np.rint(xf / scale).clip(-2047, 2047).astype(np.int32)
        bank_shape = (fanin, 64)
        if bank_shape not in banks:
            banks[bank_shape] = NativeProbabilityBank(np.full(bank_shape, .5))
        bank = banks[bank_shape]
        cache_key = (bank_shape, depthwise, samples, x.shape[0])
        if cache_key not in functions:
            functions[cache_key] = build_count_functions(bank, samples, depthwise)
        run = functions[cache_key]
        tiles = (channels + 63) // 64
        if req['mode'] == 'native':
            base = jax.random.fold_in(jax.random.key(req['key']), req['call'])
            keys = fold_tiles(base, jnp.arange(tiles, dtype=jnp.uint32))
        outputs, padding = [], 0
        for chunk in range(0, tiles, 4):
            payload = []
            for tile in range(chunk, min(chunk + 4, tiles)):
                first = tile * 64; width = min(64, channels - first)
                codes = codes_all[:, first:first + width].reshape(groups, size, width)
                lower = lower_all[:, first:first + width].astype(np.float64)
                span = span_all[:, first:first + width].astype(np.float64)
                if width < 64:
                    codes = np.pad(codes, ((0, 0), (0, 0), (0, 64 - width)), constant_values=2048)
                    lower = np.pad(lower, ((0, 0), (0, 64 - width)))
                    span = np.pad(span, ((0, 0), (0, 64 - width)))
                    padding += fanin * (64 - width)
                fields = (-4 + codes.astype(np.float32) * np.float32(8 / 4095)).reshape(-1)
                xt = x[:, :, first:first + width] if depthwise else x
                if depthwise and width < 64:
                    xt = np.pad(xt, ((0, 0), (0, 0), (0, 64 - width)))
                xg = xt.reshape(xt.shape[0], groups, size, 64) if depthwise else xt.reshape(xt.shape[0], groups, size)
                offset = xg.sum(2) if depthwise else xg.sum(2)[:, :, None]
                payload.append((tile, width, codes, lower, span, fields, xt, xg, offset))
            if req['mode'] == 'native':
                xx = jnp.asarray(np.stack([v[6] for v in payload]))
                hh = jnp.asarray(np.stack([v[5] for v in payload]))
                kk = keys[chunk:chunk + len(payload)]
                device_totals, device_counts = run['batched'](kk, xx, hh)
                totals = np.asarray(device_totals)
                states = np.stack([np.asarray(run['public'](kk[i], hh[i])) for i in range(len(payload))]) if req.get('replay') else None
            for lane, (tile, width, codes, lower, span, fields, xt, xg, offset) in enumerate(payload):
                if req['mode'] == 'native':
                    total = totals[lane]
                    if states is not None:
                        z = states[lane].reshape(samples, groups, size, 64)
                        counts = z.sum(0, dtype=np.int64)
                        expected = ((xg.astype(np.int64) * counts[None]).sum(2, dtype=np.int64) if depthwise else
                                    np.stack([xg[:, g].astype(np.int64) @ counts[g] for g in range(groups)], axis=1))
                        np.testing.assert_array_equal(total.astype(np.int64), expected)
                        returned_counts = np.asarray(device_counts)[lane].reshape(groups, size, 64)
                        np.testing.assert_array_equal(returned_counts.astype(np.int64), counts)
                        if req.get('state_dir'):
                            path = Path(req['state_dir']); path.mkdir(exist_ok=True)
                            with (path / f'call{req["call"]}_tile{tile}.npz').open('xb') as f:
                                np.savez_compressed(f, states=z, codes=codes, total=total, inputs=xt, counts=returned_counts)
                    y = np.sum(total.astype(np.float64) / samples * span[None] + offset * lower[None], axis=1) * scale
                else:
                    p = (1 / (1 + np.exp(-2 * fields.astype(np.float64)))).reshape(groups, size, 64)
                    wm = lower[:, None, :] + span[:, None, :] * p
                    y = (np.sum(xg.astype(np.float64) * wm[None], axis=(1, 2)) * scale if depthwise else
                         np.einsum('bgk,gko->bo', xg.astype(np.float64) * scale, wm))
                outputs.append(y[:, :width])
        result = np.concatenate(outputs, axis=1)
        np.save(req['output'], result)
        print(json.dumps({'ok': True, 'call': req['call'], 'weight_coordinates': codes_all.size,
            'program_codes_used_directly': True, 'reference_weights_supplied': False, 'range_metadata_dtype': 'float32',
            'padding_nodes': padding, 'field_bits': 12, 'activation_bits': 12,
            'public_replay_checked': req.get('replay', False),
            'retained_evaluations': (codes_all.size + padding) * samples if req['mode'] == 'native' else 0,
            'warmup_evaluations': codes_all.size + padding if req['mode'] == 'native' else 0,
            'simulation_observer': 'per_weight_counts_then_exact_integer_contraction',
            'simulation_count_workspace_bytes': (4 * min(4, tiles) * fanin * 64 if req['mode'] == 'native' else 0),
            'tile_lanes_limit': 4, 'max_simultaneous_tile_lanes': min(4, tiles),
            'max_simultaneous_sampler_state_slots': min(4, tiles) * fanin * 64,
            'tile_batches': (tiles + 3) // 4,
            'elapsed_seconds': time.monotonic() - begin, 'finite': bool(np.isfinite(result).all())}), flush=True)


if __name__ == '__main__':
    main()
