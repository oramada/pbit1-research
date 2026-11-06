"""Batch-shared weight states with real online case compaction.

One THRML bank realization is broadcast to every active input in a batch, as in
the earlier fixed-budget study. Compaction changes later batch membership and
therefore future shared random paths. Online decisions are recorded on their
actual paths rather than represented as replay of a different fixed schedule.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule, sample_with_observation

from .adaptive_native import AdaptiveNetwork, CHECKPOINTS, OutputMoments, key_for, should_stop


class SharedAdaptiveNetwork(AdaptiveNetwork):
    @lru_cache(maxsize=32)
    def first_function(self):
        fn = self.first.compiled(64)
        return jax.jit(lambda key, x: jax.nn.relu(fn(key, x)))

    @lru_cache(maxsize=64)
    def chunk_function(self, delta):
        bank = self.bank
        initial = [jnp.zeros(len(bank.free_indices), dtype=jnp.bool_)] if len(bank.free_indices) else []
        schedule = SamplingSchedule(1, delta, 1)

        def run(key, hidden, previous_n, mean, m2):
            observer = OutputMoments(bank.readout, hidden, bank.shape)
            carry, _ = sample_with_observation(key, bank.program, schedule, initial,
                                               bank.clamp_state, (previous_n, mean, m2), observer)
            return carry[1], carry[2]
        return jax.jit(run)

    def hidden(self, x, domain, model, seed):
        values = []
        fn = self.first_function()
        for batch, start in enumerate(range(0, len(x), self.batch_size)):
            ids = np.arange(start, min(start+self.batch_size, len(x)))
            padded = np.pad(ids, (0, self.batch_size-len(ids)), mode="edge")
            key = key_for(domain, model, seed, batch, 0)
            values.append(np.asarray(fn(key, jnp.asarray(x[padded])))[:len(ids)])
        return np.concatenate(values), ((len(x)+self.batch_size-1)//self.batch_size)*64

    def advance(self, hidden, means, m2, active, previous, target, domain, model, seed):
        fn = self.chunk_function(target-previous)
        new_means, new_m2 = means.copy(), m2.copy()
        for batch, start in enumerate(range(0, len(active), self.batch_size)):
            ids = active[start:start+self.batch_size]
            padded = np.pad(ids, (0, self.batch_size-len(ids)), mode="edge")
            key = key_for(domain, model, seed, batch, target)
            mean, moment = fn(key, jnp.asarray(hidden[padded]), jnp.int32(previous),
                              jnp.asarray(means[padded]), jnp.asarray(m2[padded]))
            new_means[ids], new_m2[ids] = np.asarray(mean)[:len(ids)], np.asarray(moment)[:len(ids)]
        batches = (len(active)+self.batch_size-1)//self.batch_size
        return new_means, new_m2, {"previous": previous, "target": target, "active_cases": len(active),
                                  "batch_count": batches, "padded_cases": batches*self.batch_size,
                                  "bank_observations": batches*(target-previous)}

    def online(self, hidden, domain, model, seed, gamma):
        means = np.zeros((len(hidden), len(self.bias)), dtype=np.float32)
        m2, outputs = np.zeros_like(means), np.zeros_like(means)
        stop_t = np.zeros(len(hidden), dtype=np.int32)
        active, previous, counters = np.arange(len(hidden)), 0, []
        history = np.full((len(CHECKPOINTS), *means.shape), np.nan, dtype=np.float32)
        moments = np.full_like(history, np.nan)
        for index, target in enumerate(CHECKPOINTS):
            if not len(active):
                break
            means, m2, count = self.advance(hidden, means, m2, active, previous, target, domain, model, seed)
            counters.append(count)
            logits = means[active]+self.bias
            history[index, active], moments[index, active] = logits, m2[active]
            stopped = should_stop(logits, m2[active], target, gamma)
            if target == CHECKPOINTS[-1]:
                stopped[:] = True
            ids = active[stopped]
            outputs[ids], stop_t[ids] = logits[stopped], target
            active, previous = active[~stopped], target
        assert np.all(stop_t > 0)
        return outputs, stop_t, counters, history, moments
