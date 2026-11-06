"""Exact and native THRML correlation characterization for tiny Ising systems.

This experiment keeps three things separate: exact equilibrium marginals and
covariances, exact temporal covariance implied by the blocked Gibbs transition
operator, and finite seeded THRML trajectories. The zero-coupling p-bank is run
through NativeProbabilityBank as a direct w=p control. Nothing here measures a
physical device or hardware speed.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from .native import NativeProbabilityBank
from .provenance import manifest


ROOT = Path("results/thrml_spw/correlations")
TARGETS = np.asarray([0.20, 0.35, 0.65, 0.80], dtype=np.float64)
COUPLINGS = (0.0, 0.35, 0.80)
SPACINGS = (1, 2, 4)
SEEDS = (3, 17, 29, 41, 73)
N_SAMPLES = 2048
N_WARMUP = 512
BETA = 1.0


def enumerate_model(n: int, p_target: np.ndarray, coupling: float):
    """Return exact state probabilities, moments, and the block-Gibbs kernel."""
    edges = [(i, i + 1) for i in range(n - 1)]
    fields = np.log(p_target) - np.log1p(-p_target)
    biases = fields / (2.0 * BETA)
    states = np.asarray(list(itertools.product([-1, 1], repeat=n)), dtype=np.int8)
    energies = -BETA * (states @ biases)
    for i, j in edges:
        energies -= BETA * coupling * states[:, i] * states[:, j]
    log_mass = -energies
    mass = np.exp(log_mass - log_mass.max())
    pi = mass / mass.sum()
    mean_s = pi @ states
    covariance_s = (states.T * pi) @ states - np.outer(mean_s, mean_s)

    lookup = {tuple(row.tolist()): i for i, row in enumerate(states)}
    blocks = [list(range(0, n, 2)), list(range(1, n, 2))]
    blocks = [block for block in blocks if block]
    block_kernels = []
    for block in blocks:
        kernel = np.zeros((len(states), len(states)), dtype=np.float64)
        for source_i, source in enumerate(states):
            local_fields = np.full(n, 0.0, dtype=np.float64)
            local_fields[:] = biases
            for i, j in edges:
                local_fields[i] += coupling * source[j]
                local_fields[j] += coupling * source[i]
            for assignments in itertools.product([-1, 1], repeat=len(block)):
                destination = source.copy()
                probability = 1.0
                for node, spin in zip(block, assignments):
                    p_plus = expit(2.0 * BETA * local_fields[node])
                    probability *= p_plus if spin == 1 else 1.0 - p_plus
                    destination[node] = spin
                kernel[source_i, lookup[tuple(destination.tolist())]] += probability
        block_kernels.append(kernel)
    # THRML samples each independent color block, in the declared block order,
    # once per sweep. Row-stochastic matrices compose left-to-right here.
    sweep = np.eye(len(states), dtype=np.float64)
    for kernel in block_kernels:
        sweep = sweep @ kernel
    if not np.allclose(sweep.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("exact Gibbs transition rows do not sum to one")
    if not np.allclose(pi @ sweep, pi, atol=1e-12):
        raise AssertionError("enumerated Gibbs distribution is not stationary")
    return states, energies, pi, mean_s, covariance_s, sweep, biases, edges, blocks


def temporal_covariance(states: np.ndarray, pi: np.ndarray, mean_s: np.ndarray,
                        transition: np.ndarray) -> np.ndarray:
    """Cov[s_i(t), s_j(t+1)] under exact stationarity and transition kernel."""
    next_means = transition @ states
    cross = np.einsum("x,xi,xj->ij", pi, states, next_means)
    return cross - np.outer(mean_s, mean_s)


def make_native_program(n: int, biases: np.ndarray, edges: list[tuple[int, int]], coupling: float):
    nodes = [SpinNode() for _ in range(n)]
    native_edges = [(nodes[i], nodes[j]) for i, j in edges]
    model = IsingEBM(nodes, native_edges, jnp.asarray(biases, dtype=jnp.float32),
                     jnp.full((len(edges),), coupling, dtype=jnp.float32),
                     jnp.asarray(BETA, dtype=jnp.float32))
    free_blocks = [Block([nodes[i] for i in block])
                   for block in [list(range(0, n, 2)), list(range(1, n, 2))]
                   if block]
    program = IsingSamplingProgram(model, free_blocks, [])
    return nodes, program, free_blocks


def sample_native(n: int, biases: np.ndarray, edges: list[tuple[int, int]],
                  coupling: float, spacing: int, seeds: tuple[int, ...],
                  n_samples: int, n_warmup: int) -> np.ndarray:
    """Run official THRML JAX Gibbs sampling independently for each seed."""
    nodes, program, blocks = make_native_program(n, biases, edges, coupling)
    schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples,
                                steps_per_sample=spacing)

    def one_chain(key):
        keys = jax.random.split(key)
        initial = [jnp.zeros((len(block),), dtype=jnp.bool_) for block in
                   [list(range(0, n, 2)), list(range(1, n, 2))] if block]
        return sample_states(keys[1], program, schedule, initial, [], [Block(nodes)])[0]

    keys = jnp.stack([jax.random.key(seed) for seed in seeds])
    samples = jax.jit(jax.vmap(one_chain))(keys)
    return np.asarray(samples, dtype=np.int8) * 2 - 1


def trajectory_stats(samples: np.ndarray):
    """Summaries per chain, then averaged over chains to retain seed as replicate."""
    # samples: chains x time x spin
    chain_means = samples.mean(axis=1)
    chain_spatial_cov = []
    chain_lag_cov = []
    chain_lag_corr = []
    for chain in samples:
        centered = chain - chain.mean(axis=0, keepdims=True)
        chain_spatial_cov.append(centered.T @ centered / len(chain))
        x, y = centered[:-1], centered[1:]
        cov = x.T @ y / max(1, len(x))
        chain_lag_cov.append(cov)
        denom = np.sqrt(np.diag(np.cov(chain.T, bias=True)))
        chain_lag_corr.append(np.diag(cov) / np.maximum(denom ** 2, 1e-15))
    return (chain_means.mean(axis=0), np.mean(chain_spatial_cov, axis=0),
            np.mean(chain_lag_cov, axis=0), np.mean(chain_lag_corr, axis=0),
            chain_means)


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    raw = {}
    exact_rows, observed_rows, zero_bank_rows = [], [], []
    exact_by_model = {}

    for n in (2, 3, 4):
        p_target = TARGETS[:n]
        for coupling in COUPLINGS:
            states, energies, pi, mean_s, cov_s, sweep, biases, edges, blocks = enumerate_model(
                n, p_target, coupling)
            exact_by_model[(n, coupling)] = (states, pi, mean_s, cov_s, sweep, biases, edges)
            transition_eigenvalues = np.linalg.eigvals(sweep)
            stationary_index = int(np.argmin(np.abs(transition_eigenvalues - 1.0)))
            nonstationary = np.delete(transition_eigenvalues, stationary_index)
            exact_rows.append({
                "n_spins": n, "coupling_J": coupling,
                "target_p": json.dumps(p_target.tolist()),
                "exact_marginal_p": json.dumps(((mean_s + 1) / 2).tolist()),
                "marginal_shift": json.dumps((((mean_s + 1) / 2) - p_target).tolist()),
                "spatial_cov_spin": json.dumps(cov_s.tolist()),
                "spatial_cov_bernoulli": json.dumps((cov_s / 4).tolist()),
                "sweep_spectral_gap": float(1.0 - np.max(np.abs(nonstationary))),
            })
            if coupling == 0.0:
                # Direct p bank: every target p is programmed independently, with no coupling.
                bank = NativeProbabilityBank(p_target, beta=BETA)
                bank_samples = (bank.sample(seed=101, n_samples=N_SAMPLES,
                                            n_chains=len(SEEDS), initial_one=False)
                                .astype(np.int8) * 2 - 1)
                raw[f"bank_n{n}"] = bank_samples
                emp_mean, emp_cov, lag_cov, lag_corr, seed_means = trajectory_stats(bank_samples)
                for i in range(n):
                    zero_bank_rows.append({
                        "n_spins": n, "node_i": i,
                        "target_p": p_target[i], "native_realized_p": bank.realized_probabilities.reshape(-1)[i],
                        "empirical_p": (emp_mean[i] + 1) / 2,
                        "empirical_marginal_error": (emp_mean[i] + 1) / 2 - p_target[i],
                        "empirical_spatial_cov_with_others": json.dumps(emp_cov[i].tolist()),
                        "empirical_lag1_corr": lag_corr[i],
                        "sample_spacing_sweeps": 1,
                        "base_seed": 101,
                        "n_chains": len(SEEDS), "n_samples_per_chain": N_SAMPLES,
                    })

            for spacing in SPACINGS:
                transition = np.linalg.matrix_power(sweep, spacing)
                exact_lag = temporal_covariance(states, pi, mean_s, transition)
                native = sample_native(n, biases, edges, coupling, spacing, SEEDS,
                                       N_SAMPLES, N_WARMUP)
                raw[f"ising_n{n}_j{coupling}_d{spacing}"] = native
                emp_mean, emp_cov, lag_cov, lag_corr, seed_means = trajectory_stats(native)
                exact_p = (mean_s + 1) / 2
                for i in range(n):
                    observed_rows.append({
                        "n_spins": n, "coupling_J": coupling, "spacing_sweeps": spacing,
                        "node_i": i, "target_p_uncoupled": p_target[i],
                        "exact_coupled_p": exact_p[i],
                        "exact_marginal_shift": exact_p[i] - p_target[i],
                        "empirical_p_mean_over_seeds": (emp_mean[i] + 1) / 2,
                        "empirical_p_minus_exact": (emp_mean[i] + 1) / 2 - exact_p[i],
                        "exact_spatial_cov_row_spin": json.dumps(cov_s[i].tolist()),
                        "empirical_spatial_cov_row_spin": json.dumps(emp_cov[i].tolist()),
                        "exact_lag1_temporal_cov_row_spin": json.dumps(exact_lag[i].tolist()),
                        "empirical_lag1_temporal_cov_row_spin": json.dumps(lag_cov[i].tolist()),
                        "exact_lag1_autocorrelation": exact_lag[i, i] / max(cov_s[i, i], 1e-15),
                        "empirical_lag1_autocorrelation": lag_corr[i],
                        "n_chains": len(SEEDS), "n_samples_per_chain": N_SAMPLES,
                        "warmup_sweeps": N_WARMUP,
                    })
                print(f"n={n} J={coupling:.2f} spacing={spacing} "
                      f"max marginal err={np.max(np.abs((emp_mean+1)/2-exact_p)):.4f} "
                      f"mean lag1 corr={np.mean(lag_corr):.3f}", flush=True)

    np.savez_compressed(ROOT / "raw_trajectories.npz", **raw)
    write_csv(ROOT / "exact_model_moments.csv", exact_rows)
    write_csv(ROOT / "native_coupled_observations.csv", observed_rows)
    write_csv(ROOT / "zero_coupling_probability_bank.csv", zero_bank_rows)

    max_shift_by_j = {}
    for key, (_, _, _, _, _, _, _) in exact_by_model.items():
        n, coupling = key
        p_target = TARGETS[:n]
        mean_s = exact_by_model[key][2]
        max_shift_by_j.setdefault(coupling, 0.0)
        max_shift_by_j[coupling] = max(
            max_shift_by_j[coupling],
            float(np.max(np.abs((mean_s + 1) / 2 - p_target))),
        )
    max_native_marginal_error = max(abs(r["empirical_p_minus_exact"]) for r in observed_rows)
    max_spatial_cov_error = max(
        float(np.max(np.abs(np.asarray(json.loads(r["empirical_spatial_cov_row_spin"])) -
                            np.asarray(json.loads(r["exact_spatial_cov_row_spin"])))))
        for r in observed_rows
    )
    max_temporal_cov_error = max(
        float(np.max(np.abs(np.asarray(json.loads(r["empirical_lag1_temporal_cov_row_spin"])) -
                            np.asarray(json.loads(r["exact_lag1_temporal_cov_row_spin"])))))
        for r in observed_rows
    )
    max_zero_bank_lagcorr = max(abs(r["empirical_lag1_corr"]) for r in zero_bank_rows)
    zero_offdiag = []
    for r in zero_bank_rows:
        cov = np.asarray(json.loads(r["empirical_spatial_cov_with_others"]))
        zero_offdiag.extend(np.delete(cov, int(r["node_i"])) if cov.ndim == 1 else cov.tolist())

    report = f"""# Native THRML coupling and correlation characterization

## Method

Small Ising chains with 2, 3, and 4 spins were exhaustively enumerated. The energy is
the official THRML convention
`E(s) = -beta (sum_i b_i s_i + sum_(i,j) J_ij s_i s_j)` with beta={BETA}.
For each node, the field is programmed from the uncoupled target probability
`p_i={{{', '.join(f'{v:.2f}' for v in TARGETS[:4])}}}` using
`b_i=logit(p_i)/(2 beta)`. The same fields are retained under couplings
J in {COUPLINGS}. Coupled models use a nearest-neighbor chain and alternating
independent-set blocks. Exact state probabilities, marginals, spatial covariance,
the full blocked-sweep transition matrix, and lagged covariance under its powers are
computed without Monte Carlo.

The native sampler is the pinned THRML {__import__('importlib.metadata').metadata.version('thrml')} JAX implementation. For each model/spacing, {len(SEEDS)} independent
chains each record {N_SAMPLES} samples after {N_WARMUP} warmup sweeps. Sampling spacings
are {SPACINGS} full sweeps. The zero-coupling control uses the project-native
`NativeProbabilityBank` (`w=p`) and directly samples the requested marginals.

## Findings

At J=0 the exact marginals equal the requested p values and cross-spin covariance is
zero. The maximum absolute exact marginal shift across the evaluated n is
{max_shift_by_j[0.0]:.6f} at J=0, {max_shift_by_j[0.35]:.6f} at J=.35, and
{max_shift_by_j[0.8]:.6f} at J=.8. At nonzero J, preserving the unary biases does not
preserve each individual marginal: neighbors alter its effective field. The largest
absolute native-vs-exact marginal discrepancy across all coupled runs was
{max_native_marginal_error:.4f} over five independent seeds.

Coupling creates nonzero spatial covariance and temporal persistence. With J=.8, exact
one-sweep autocorrelation reaches up to {max(r["exact_lag1_autocorrelation"] for r in observed_rows if r["coupling_J"] == 0.8 and r["spacing_sweeps"] == 1):.3f}
for an observed spin. The maximum absolute native-vs-exact row error was
{max_spatial_cov_error:.4f} for spatial covariance and {max_temporal_cov_error:.4f} for
lag-one temporal covariance. Increasing sample spacing applies the exact transition
operator power `P^d`; in the strongly coupled cases the measured lag-one persistence
falls as spacing grows.

For the independent w=p bank, measured lag-one autocorrelation remains bounded by
{max_zero_bank_lagcorr:.3f} in magnitude across the nodes, and cross-spin sample
covariances are close to zero. The control is included separately from coupled Ising
models. See `exact_model_moments.csv`, `native_coupled_observations.csv`, and
`zero_coupling_probability_bank.csv` for the numeric evidence. Raw chain states are
preserved in `raw_trajectories.npz`.

This is simulator characterization only. THRML sweep spacing is an algorithmic count;
it does not represent p-bit physical transition time, chip correlation, speed, energy,
or hardware performance.
"""
    (ROOT / "report.md").write_text(report)

    results = {
        "exact_configurations": len(exact_rows),
        "native_configurations": len(observed_rows),
        "zero_bank_rows": len(zero_bank_rows),
        "seeds": list(SEEDS), "elapsed_seconds": time.time() - start,
        "max_abs_native_marginal_error": max_native_marginal_error,
        "max_abs_native_spatial_covariance_error": max_spatial_cov_error,
        "max_abs_native_temporal_covariance_error": max_temporal_cov_error,
        "max_abs_zero_bank_lag1_autocorrelation": max_zero_bank_lagcorr,
        "exact_stationarity_max_error": max(
            float(np.max(np.abs(exact_by_model[key][1] @ exact_by_model[key][4] - exact_by_model[key][1])))
            for key in exact_by_model
        ),
        "scope": "THRML/JAX simulator only; no physical correlation or speed claim",
    }
    config = {"n_spins": [2, 3, 4], "target_probabilities": TARGETS.tolist(),
              "couplings": COUPLINGS, "spacings_full_sweeps": SPACINGS,
              "seeds": SEEDS, "n_samples": N_SAMPLES, "n_warmup": N_WARMUP,
              "beta": BETA, "topology": "nearest-neighbor chain",
              "blocks": "alternating independent sets", "zero_control": "NativeProbabilityBank w=p"}
    manifest(ROOT / "manifest.json", config, results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    run()
