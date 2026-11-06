"""Official-THRML positional-word operator on one padded, integer-input bank.

Temporary per-weight int32 counts are simulator workspace, not free hardware.
Caller handles ACT12 input quantization, tiling, padding, and module layout.
"""
import jax
import jax.numpy as jnp
import numpy as np
from .counted_bank import build_count_functions


def execute(bank, inputs, words, lower, span, bits, samples, key, depthwise=False, replay=False):
    x, words = np.asarray(inputs), np.asarray(words)
    lower, span = np.asarray(lower), np.asarray(span)
    if bits not in (4, 8) or samples <= 0:
        raise ValueError("Expected K4/K8 and positive sampling budget")
    if words.dtype != np.uint8 or tuple(words.shape) != tuple(bank.shape) or np.max(words) >= 2 ** bits:
        raise ValueError("Invalid word-bank shape, type or range")
    if x.dtype != np.int32 or np.max(np.abs(x.astype(np.int64))) > 2047:
        raise ValueError("Expected signed ACT12 integer inputs")
    fanin, channels = bank.shape
    size = min(128, fanin)
    if channels != 64 or fanin % size or 2047 * size * samples >= 2 ** 31:
        raise ValueError("Unsupported bank geometry or int32 bound")
    groups = fanin // size
    assert lower.shape == span.shape == (groups, 64)
    assert np.isfinite(lower).all() and np.isfinite(span).all() and (span >= 0).all()
    assert x.ndim == (3 if depthwise else 2) and x.shape[1] == fanin
    if depthwise:
        assert x.shape[2] == 64
        xg = x.reshape(x.shape[0], groups, size, 64)
        offset = np.sum(xg, axis=2, dtype=np.int64)
    else:
        xg = x.reshape(x.shape[0], groups, size)
        offset = np.sum(xg, axis=2, dtype=np.int64)[:, :, None]
    functions = build_count_functions(bank, samples, depthwise)
    weighted = np.zeros((x.shape[0], groups, 64), np.float64)
    planes = []
    for bit in range(bits):
        fields = np.where((words >> bit) & 1, 4, -4).astype(np.float32).reshape(-1)
        bit_key = jax.random.fold_in(key, bit)
        total, counts = functions['serial'](bit_key, jnp.asarray(x), jnp.asarray(fields))
        total, counts = np.asarray(total), np.asarray(counts)
        weighted += total.astype(np.float64) * ((1 << bit) / ((1 << bits) - 1))
        if replay:
            states = np.asarray(functions['public'](bit_key, jnp.asarray(fields)))
            planes.append({'bit': bit, 'states': states, 'counts': counts, 'total': total})
    y = np.sum(weighted / samples * span[None] + offset * lower[None], axis=1)
    return y, {
        'planes': planes,
        'retained_scheduled_evaluations': bits * bank.logical_pbits * samples,
        'warmup_scheduled_evaluations': bits * bank.logical_pbits,
        'max_temporary_weight_count_bytes': bank.logical_pbits * 4,
        'max_simultaneous_sampler_state_slots': bank.logical_pbits,
        'logical_word_pbits': bits * bank.logical_pbits,
    }
