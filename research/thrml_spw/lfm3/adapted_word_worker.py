"""Tiled calibrated K4/K8 controls, official THRML and counted software observer.

Consumes exported words and fixed ranges directly, not reference float weights.
Bitplanes are sequential; up to four independent 64-column tiles are batched.
Observer carry storage is explicit and excludes additional raw-replay retention.
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
    fold_bits = jax.jit(jax.vmap(jax.random.fold_in, in_axes=(0, None)))
    print(json.dumps({'ready': True, 'devices': list(map(str, jax.devices()))}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if req.get('stop'):
            break
        begin = time.monotonic()
        with np.load(req['input']) as loaded:
            assert set(loaded.files) == {'x', 'words', 'lower', 'span'}
            xf, words_all, lower_all, span_all = [loaded[k] for k in ['x', 'words', 'lower', 'span']]
        bits, samples = int(req['word_bits']), int(req['samples'])
        assert bits in [4, 8] and samples > 0
        assert req['activation_bits'] == 12 and req['mode'] in ['mean', 'native']
        assert req['kind'] in ['linear', 'conv', 'embedding', 'elementwise']
        assert words_all.dtype == np.uint8 and words_all.ndim == 2 and words_all.size > 0
        assert np.max(words_all) < (1 << bits)
        assert np.isfinite(xf).all() and xf.size > 0
        assert lower_all.dtype == span_all.dtype == np.float32
        assert np.isfinite(lower_all).all() and np.isfinite(span_all).all() and np.min(span_all) >= 0
        fanin, channels = words_all.shape
        size = min(128, fanin)
        assert fanin % size == 0 and 2047 * size * samples < 2 ** 31
        groups = fanin // size
        assert lower_all.shape == span_all.shape == (groups, channels)
        depthwise = req['kind'] != 'linear'
        assert xf.shape[1] == fanin and xf.ndim == (3 if depthwise else 2)
        if depthwise:
            assert xf.shape[2] == channels
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
                first = tile * 64
                width = min(64, channels - first)
                words = words_all[:, first:first + width].reshape(groups, size, width)
                lower = lower_all[:, first:first + width].astype(np.float64)
                span = span_all[:, first:first + width].astype(np.float64)
                if width < 64:
                    words = np.pad(words, ((0, 0), (0, 0), (0, 64 - width)))
                    lower = np.pad(lower, ((0, 0), (0, 64 - width)))
                    span = np.pad(span, ((0, 0), (0, 64 - width)))
                    padding += fanin * (64 - width)
                xt = x[:, :, first:first + width] if depthwise else x
                if depthwise and width < 64:
                    xt = np.pad(xt, ((0, 0), (0, 0), (0, 64 - width)))
                xg = xt.reshape(xt.shape[0], groups, size, 64) if depthwise else xt.reshape(xt.shape[0], groups, size)
                offset = xg.sum(axis=2, dtype=np.int64)
                if not depthwise:
                    offset = offset[:, :, None]
                payload.append((tile, width, words, lower, span, xt, xg, offset))
            if req['mode'] == 'native':
                xx = jnp.asarray(np.stack([p[5] for p in payload]))
                weighted = np.zeros((len(payload), x.shape[0], groups, 64), np.float64)
                for bit in range(bits):
                    fields = np.stack([np.where((p[2] >> bit) & 1, 4, -4).astype(np.float32).reshape(-1) for p in payload])
                    hh = jnp.asarray(fields)
                    kk = fold_bits(keys[chunk:chunk + len(payload)], bit)
                    device_totals, device_counts = run['batched'](kk, xx, hh)
                    totals = np.asarray(device_totals)
                    weighted += totals.astype(np.float64) * ((1 << bit) / ((1 << bits) - 1))
                    if req.get('replay'):
                        for lane, p in enumerate(payload):
                            tile, width, words, lower, span, xt, xg, offset = p
                            states = np.asarray(run['public'](kk[lane], hh[lane])).reshape(samples, groups, size, 64)
                            counts = np.sum(states, axis=0, dtype=np.int64)
                            np.testing.assert_array_equal(np.asarray(device_counts)[lane].reshape(groups, size, 64), counts)
                            expected = ((xg.astype(np.int64) * counts[None]).sum(axis=2, dtype=np.int64) if depthwise else
                                        np.stack([xg[:, g].astype(np.int64) @ counts[g] for g in range(groups)], axis=1))
                            np.testing.assert_array_equal(totals[lane], expected)
                            if req.get('state_dir'):
                                path = Path(req['state_dir'])
                                path.mkdir(exist_ok=True)
                                with (path / f'call{req["call"]}_tile{tile}_bit{bit}.npz').open('xb') as handle:
                                    np.savez_compressed(handle, states=states, words=words, total=totals[lane], inputs=xt,
                                        counts=np.asarray(device_counts)[lane].reshape(groups, size, 64),
                                        lower=lower, span=span, scale=np.asarray(scale), width=np.asarray(width))
            for lane, (_, width, words, lower, span, xt, xg, offset) in enumerate(payload):
                if req['mode'] == 'native':
                    y = np.sum(weighted[lane] / samples * span[None] + offset * lower[None], axis=1) * scale
                else:
                    p0, p1 = 1 / (1 + np.exp(8.0)), 1 / (1 + np.exp(-8.0))
                    probability = p0 + (p1 - p0) * words.astype(np.float64) / ((1 << bits) - 1)
                    wm = lower[:, None, :] + span[:, None, :] * probability
                    y = (np.sum(xg.astype(np.float64) * wm[None], axis=(1, 2)) * scale if depthwise else
                         np.einsum('bgk,gko->bo', xg.astype(np.float64) * scale, wm))
                outputs.append(y[:, :width])
        result = np.concatenate(outputs, axis=1)
        np.save(req['output'], result)
        native = req['mode'] == 'native'
        print(json.dumps({'ok': True, 'call': req['call'], 'weight_coordinates': words_all.size,
            'word_bits': bits, 'pbits_per_parameter': bits, 'programmable_field_levels_per_node': 2,
            'field_bound': 4, 'activation_bits': 12, 'program_words_used_directly': True,
            'reference_weights_supplied': False, 'range_metadata_dtype': 'float32',
            'padding_parameter_slots': padding, 'padding_nodes': bits * padding,
            'retained_evaluations': bits * (words_all.size + padding) * samples if native else 0,
            'warmup_evaluations': bits * (words_all.size + padding) if native else 0,
            'public_replay_checked': bool(req.get('replay')),
            'simulation_observer': 'per_weight_counts_per_bitplane_then_integer_contraction',
            'simulation_count_carry_bytes': 4 * min(4, tiles) * fanin * 64 if native else 0,
            'count_memory_scope': 'Observer carry only; excludes total runtime buffers and additional raw-replay retention.',
            'max_simultaneous_sampler_state_slots': min(4, tiles) * fanin * 64,
            'bitplanes_executed_sequentially': True, 'tile_lanes_limit': 4,
            'elapsed_seconds': time.monotonic() - begin, 'finite': bool(np.isfinite(result).all())}), flush=True)


if __name__ == '__main__':
    main()
