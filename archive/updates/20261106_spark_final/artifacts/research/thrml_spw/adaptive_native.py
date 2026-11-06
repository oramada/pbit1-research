"""Online output-layer stopping with actual THRML calls and case compaction.

Each input has an independent p-bit chain. The first layer is fixed at T=64;
the second layer pools fresh independent native chunks at T=16/32/64/128/256.
Only uncoupled unary banks are supported: one warmup sweep produces stationarity
and each observation costs exactly one sweep. Resetting a coupled chain this way
would not preserve its stationary law or its correlation structure.

Confidence uses conditional output uncertainty, not uncertainty in the first
layer. The sequential margin rule is validation-calibrated, not an anytime-valid
confidence guarantee. All arithmetic remains modeled external JAX operations.
"""
from __future__ import annotations

from functools import lru_cache
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from thrml import AbstractObserver, SamplingSchedule, block_state_to_global, from_global_state, sample_with_observation

from .native import NativeProbabilityBank
from .positive_inference import NativeLinear


CHECKPOINTS = (16, 32, 64, 128, 256)


class OutputMoments(AbstractObserver):
    readout: object
    inputs: jax.Array
    weight_shape: tuple

    def __call__(self, program, state_free, state_clamped, carry, iteration):
        global_state = block_state_to_global(state_free+state_clamped, program.gibbs_spec)
        z = from_global_state(global_state, program.gibbs_spec, [self.readout])[0]
        value = self.inputs@z.reshape(self.weight_shape).astype(jnp.float32)
        n, mean, m2 = carry
        n = n+1
        delta = value-mean
        mean = mean+delta/n
        m2 = m2+delta*(value-mean)
        return (n, mean, m2), None


def key_for(domain, model, seed, example, tag):
    key = jax.random.key(domain)
    for value in (model, seed, example, tag):
        key = jax.random.fold_in(key, value)
    return key


def should_stop(logits, m2, n, gamma):
    """Paired-margin heuristic under independent output-column weight banks."""
    order = np.argsort(logits, axis=-1)
    a, b = order[:, -1], order[:, -2]
    ids = np.arange(len(logits))
    margin = logits[ids, a]-logits[ids, b]
    # Unbiased sample variance / n = M2/[n(n-1)]. Different output
    # columns have independent weights, conditional on the frozen hidden input.
    se = np.sqrt(np.maximum(m2[ids, a]+m2[ids, b], 0)/(n*(n-1)))
    if np.isinf(gamma):
        return np.zeros(len(logits), dtype=bool)
    return margin > gamma*se


class AdaptiveNetwork:
    def __init__(self, weights, biases, batch_size=128):
        self.first = NativeLinear(weights[0], biases[0])
        self.bank = NativeProbabilityBank(weights[1])
        self.bias = np.asarray(biases[1], dtype=np.float32)
        self.batch_size = batch_size

    @lru_cache(maxsize=32)
    def first_function(self):
        one = self.first.compiled(64)
        return jax.jit(jax.vmap(lambda key, row: jax.nn.relu(one(key, row[None, :]))[0]))

    @lru_cache(maxsize=64)
    def chunk_function(self, delta):
        bank = self.bank
        initial = [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)] if len(bank.free_indices) else []
        schedule = SamplingSchedule(1, delta, 1)

        def one(key, hidden, previous_n, previous_mean, previous_m2):
            observer = OutputMoments(bank.readout, hidden, bank.shape)
            carry, _ = sample_with_observation(key, bank.program, schedule, initial,
                         bank.clamp_state, (previous_n, previous_mean, previous_m2), observer)
            return carry[1], carry[2]
        return jax.jit(jax.vmap(one, in_axes=(0, 0, None, 0, 0)))

    def hidden(self, x, domain, model, seed):
        values = []
        fn = self.first_function()
        for start in range(0, len(x), self.batch_size):
            ids = np.arange(start, min(start+self.batch_size, len(x)))
            ids_pad = np.pad(ids, (0, self.batch_size-len(ids)), mode="edge")
            keys = jax.vmap(lambda i: key_for(domain, model, seed, i, 0))(jnp.asarray(ids_pad))
            values.append(np.asarray(fn(keys, jnp.asarray(x[ids_pad])))[:len(ids)])
        padded_cases = ((len(x)+self.batch_size-1)//self.batch_size)*self.batch_size
        return np.concatenate(values), padded_cases*64

    def advance(self, hidden, means, m2, active, previous, target, domain, model, seed):
        """Actually sample only active cases plus explicitly counted padding."""
        delta = target-previous
        fn = self.chunk_function(delta)
        new_means, new_m2 = means.copy(), m2.copy()
        for start in range(0, len(active), self.batch_size):
            ids = active[start:start+self.batch_size]
            ids_pad = np.pad(ids, (0, self.batch_size-len(ids)), mode="edge")
            keys = jax.vmap(lambda i: key_for(domain, model, seed, i, target))(jnp.asarray(ids_pad))
            mu, moment = fn(keys, jnp.asarray(hidden[ids_pad]), jnp.int32(previous),
                            jnp.asarray(means[ids_pad]), jnp.asarray(m2[ids_pad]))
            new_means[ids], new_m2[ids] = np.asarray(mu)[:len(ids)], np.asarray(moment)[:len(ids)]
        padded = ((len(active)+self.batch_size-1)//self.batch_size)*self.batch_size if len(active) else 0
        return new_means, new_m2, {"previous": previous, "target": target, "active_cases": len(active),
                                  "padded_cases": padded, "chain_observations": padded*delta}

    def collect(self, hidden, domain, model, seed, maximum=256):
        means = np.zeros((len(hidden), len(self.bias)), dtype=np.float32)
        m2 = np.zeros_like(means)
        history, moments, counters = [], [], []
        previous = 0
        stages = sorted({t for t in CHECKPOINTS if t <= maximum} | {maximum})
        for target in stages:
            means, m2, count = self.advance(hidden, means, m2, np.arange(len(hidden)), previous,
                                           target, domain, model, seed)
            history.append(means+self.bias)
            moments.append(m2.copy())
            counters.append(count)
            previous = target
        return np.stack(history), np.stack(moments), counters

    def online(self, hidden, domain, model, seed, gamma):
        means = np.zeros((len(hidden), len(self.bias)), dtype=np.float32)
        m2 = np.zeros_like(means)
        outputs = np.zeros_like(means)
        stop_t = np.zeros(len(hidden), dtype=np.int32)
        active, previous, counters = np.arange(len(hidden)), 0, []
        for target in CHECKPOINTS:
            if not len(active):
                break
            means, m2, count = self.advance(hidden, means, m2, active, previous, target, domain, model, seed)
            counters.append(count)
            logits = means[active]+self.bias
            stopped = should_stop(logits, m2[active], target, gamma)
            if target == CHECKPOINTS[-1]:
                stopped[:] = True
            ids = active[stopped]
            outputs[ids], stop_t[ids] = logits[stopped], target
            active = active[~stopped]
            previous = target
        assert np.all(stop_t > 0)
        return outputs, stop_t, counters


def replay(history, moments, gamma):
    output = np.zeros_like(history[0])
    stop = np.zeros(len(output), dtype=np.int32)
    for index, t in enumerate(CHECKPOINTS):
        selected = (stop == 0) & should_stop(history[index], moments[index], t, gamma)
        if t == CHECKPOINTS[-1]:
            selected = stop == 0
        output[selected], stop[selected] = history[index, selected], t
    return output, stop


def compact_observations(stop, batch_size):
    previous, total = 0, 0
    for target in CHECKPOINTS:
        active = int(np.sum(stop > previous))
        padded = ((active+batch_size-1)//batch_size)*batch_size if active else 0
        total += padded*(target-previous)
        previous = target
    return total
