"""Reproduce the official README Ising chain, then check all 32 states."""
from __future__ import annotations

import csv
import itertools
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

from .provenance import manifest


def main():
    out = Path("results/thrml_spw/official_example")
    out.mkdir(parents=True, exist_ok=True)
    nodes = [SpinNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i+1]) for i in range(4)]
    model = IsingEBM(nodes, edges, jnp.zeros(5), jnp.ones(4)*0.5, jnp.array(1.0))
    free = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free, clamped_blocks=[])
    states = np.array(list(itertools.product([False, True], repeat=5)))
    spins = 2*states.astype(float)-1
    analytical_energy = -0.5*np.sum(spins[:, :-1]*spins[:, 1:], axis=1)
    native_energy = np.asarray(jax.vmap(lambda x: model.energy([x], [Block(nodes)]))(jnp.asarray(states)))
    np.testing.assert_allclose(native_energy, analytical_energy, atol=1e-6)
    exact = np.exp(-analytical_energy)
    exact /= exact.sum()
    rows, tv = [], []
    seeds = list(range(10))
    schedule = SamplingSchedule(n_warmup=100, n_samples=20000, steps_per_sample=2)
    for seed in seeds:
        ki, ks = jax.random.split(jax.random.key(seed))
        initial = hinton_init(ki, model, free, ())
        samples = np.asarray(sample_states(ks, program, schedule, initial, [], [Block(nodes)])[0])
        indexes = samples.astype(np.uint8) @ (2**np.arange(4, -1, -1))
        frequency = np.bincount(indexes, minlength=32)/len(samples)
        tv.append(float(0.5*np.abs(frequency-exact).sum()))
        for index in range(32):
            rows.append({"seed":seed, "state":format(index,"05b"),
                         "exact_probability":float(exact[index]),
                         "measured_probability":float(frequency[index])})
    with (out/"state_probabilities.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"max_native_energy_error":float(np.max(np.abs(native_energy-analytical_energy))),
              "total_variation_by_seed":tv, "mean_total_variation":float(np.mean(tv)),
              "validation_threshold_tv":0.03, "passed":bool(max(tv)<0.03)}
    manifest(out/"manifest.json", {"reference":"vendor/thrml/README.md quick example",
             "nodes":5,"coupling":0.5,"beta":1.0,"seeds":seeds,
             "n_samples":20000,"warmup":100,"steps_per_sample":2}, result)
    print(result, flush=True)
    if not result["passed"]:
        raise RuntimeError("Official example distribution check failed")


if __name__ == "__main__":
    main()
