"""Internal eight-free-p-bit positional baseline using official THRML sampling.

Finite fields +/-4, ACT12 inputs, shared block128 ranges, direct output sums.
Eight nodes per scalar; two programmable field levels per node. This is not an
assertion about Extropic's physical weight representation.
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
from thrml import SamplingSchedule, sample_states, sample_with_observation
from research.thrml_spw.native import NativeProbabilityBank
from research.thrml_spw.controlled_positive_native import set_fields
from research.thrml_spw.lfm3.precision_worker import GroupSum


def main():
    banks = {}
    functions = {}
    print(json.dumps({'ready': True, 'devices': [str(d) for d in jax.devices()]}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if req.get('stop'):
            break
        start = time.monotonic()
        a = np.load(req['input'])
        xf, w, kind = a['x'], a['w'], req['kind']
        depthwise = kind != 'linear'
        assert req.get('activation_bits', 12) == 12
        t = int(req['samples'])
        scale = float(np.abs(xf).max()) / 2047 if np.any(xf) else 1.0
        x = np.rint(xf / scale).clip(-2047, 2047).astype(np.int32)
        fanin = w.shape[0]
        size = min(128, fanin)
        shape = (fanin, 64)
        assert fanin % size == 0 and 2047 * size * t < np.iinfo(np.int32).max
        groups = fanin // size
        if shape not in banks:
            banks[shape] = NativeProbabilityBank(np.full(shape, .5))
        bank = banks[shape]
        cachekey = (shape, depthwise, t, x.shape[0])
        if cachekey not in functions:
            def build(bank, size, t, depthwise):
                initial = [jnp.zeros(bank.logical_pbits, dtype=jnp.bool_)]

                def run(key, inputs, fields):
                    obs = GroupSum(bank.readout, inputs, bank.shape, depthwise, size)
                    total, _ = sample_with_observation(key, set_fields(bank, fields),
                        SamplingSchedule(1, t, 1), initial, [], obs.init(), obs)
                    return total

                def public(key, fields):
                    return sample_states(key, set_fields(bank, fields), SamplingSchedule(1, t, 1),
                                         initial, [], [bank.readout])[0]
                return jax.jit(run), jax.jit(public)
            functions[cachekey] = build(bank, size, t, depthwise)
        run, public = functions[cachekey]
        outputs = []
        padding = 0
        for tile, first in enumerate(range(0, w.shape[1], 64)):
            width = min(64, w.shape[1] - first)
            wt = w[:, first:first + width].astype(np.float64)
            if width < 64:
                wt = np.pad(wt, ((0, 0), (0, 64 - width)))
                padding += fanin * (64 - width)
            wg = wt.reshape(groups, size, 64)
            lower, span = wg.min(1), np.ptp(wg, axis=1)
            if kind in ['embedding', 'elementwise']:
                assert fanin == 1
                lower = np.asarray(a['lower'][first:first + width], dtype=np.float64)
                span = np.asarray(a['span'][first:first + width], dtype=np.float64)
                if width < 64:
                    lower = np.pad(lower, (0, 64 - width))
                    span = np.pad(span, (0, 64 - width))
                lower, span = lower[None, :], span[None, :]
            normalized = np.divide(wg - lower[:, None, :], span[:, None, :],
                                   out=np.full_like(wg, .5), where=span[:, None, :] != 0).clip(0, 1)
            word = np.rint(normalized * 255).astype(np.uint8)
            xt = x[:, :, first:first + width] if depthwise else x
            if depthwise and width < 64:
                xt = np.pad(xt, ((0, 0), (0, 0), (0, 64 - width)))
            xg = xt.reshape(xt.shape[0], groups, size, 64) if depthwise else xt.reshape(xt.shape[0], groups, size)
            offset = xg.sum(2) if depthwise else xg.sum(2)[:, :, None]
            if req['mode'] == 'native':
                weighted_total = np.zeros((xt.shape[0], groups, 64), dtype=np.float64)
                for bit in range(8):
                    signs = ((word >> bit) & 1).astype(np.uint8)
                    fields = np.where(signs, 4, -4).astype(np.float32).reshape(-1)
                    key = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(
                        jax.random.key(req['key']), req['call']), tile), bit)
                    total = np.asarray(run(key, jnp.asarray(xt), jnp.asarray(fields)))
                    if req.get('replay', False):
                        states = np.asarray(public(key, jnp.asarray(fields))).reshape(t, groups, size, 64)
                        checked = np.zeros_like(total, dtype=np.int64)
                        for state in states:
                            if depthwise:
                                checked += np.sum(xg.astype(np.int64) * state[None, :, :, :], axis=2, dtype=np.int64)
                            else:
                                for group in range(groups):
                                    checked[:, group, :] += xg[:, group, :].astype(np.int64) @ state[group].astype(np.int64)
                        np.testing.assert_array_equal(total.astype(np.int64), checked)
                        if req.get('state_dir'):
                            path = Path(req['state_dir'])
                            path.mkdir(exist_ok=True)
                            np.savez_compressed(path / f"call{req['call']}_tile{tile}_bit{bit}.npz",
                                                states=states, signs=signs, total=total, inputs=xt)
                    weighted_total += total.astype(np.float64) * ((1 << bit) / 255)
                y = np.sum(weighted_total / t * span[None, :, :] + offset * lower[None, :, :], axis=1) * scale
            else:
                endpoint_error = 1 / (1 + np.exp(8.0))
                probability = endpoint_error + (1 - 2 * endpoint_error) * word.astype(np.float64) / 255
                expected = lower[:, None, :] + span[:, None, :] * probability
                y = np.sum(xg.astype(np.float64) * expected[None, :, :, :], axis=(1, 2)) * scale if depthwise else np.einsum('bgk,gko->bo', xg.astype(np.float64) * scale, expected)
            outputs.append(y[:, :width])
        result = np.concatenate(outputs, axis=1)
        np.save(req['output'], result)
        print(json.dumps({'ok': True, 'call': req['call'], 'weight_coordinates': w.size,
            'pbits_per_parameter': 8, 'padding_parameter_slots': padding,
            'logical_pbit_uses': 8 * w.size, 'padding_nodes': 8 * padding,
            'programmable_field_levels_per_node': 2, 'field_bound': 4, 'activation_bits': 12,
            'public_replay_checked': req.get('replay', False),
            'retained_evaluations': 8 * (w.size + padding) * t if req['mode'] == 'native' else 0,
            'warmup_evaluations': 8 * (w.size + padding) if req['mode'] == 'native' else 0,
            'elapsed_seconds': time.monotonic() - start, 'finite': bool(np.isfinite(result).all())}), flush=True)


if __name__ == '__main__':
    main()
