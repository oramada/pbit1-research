"""Matched observation budgets on Extropic THRML's native Gibbs sampler.

For each target probability p, compare one logical p-bit read K*T times in
time with K same-p replicas read T times in parallel. The parallel replicas
are deliberately not described as a positional K-bit representation; that
deterministic code is measured independently below.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from research.thrml_spw.native import NativeProbabilityBank
from research.thrml_spw.provenance import manifest


def positional_code(probabilities: np.ndarray, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Unsigned positive positional code with 2**K evenly-spaced levels incl endpoints."""
    qmax = (1 << n_bits) - 1
    integers = np.rint(np.asarray(probabilities) * qmax).astype(np.uint64)
    bit_positions = np.arange(n_bits, dtype=np.uint64)
    bits = ((integers[..., None] >> bit_positions) & 1).astype(np.uint8)
    weights = (2.0 ** bit_positions.astype(np.float64)) / qmax
    decoded = bits @ weights
    return bits, decoded


def _trial_counts_temporal(bank: NativeProbabilityBank, K: int, T: int,
                           n_trials: int, batch_chains: int, seed_base: int):
    counts = np.empty((n_trials, bank.probabilities.size), dtype=np.uint16)
    for start in range(0, n_trials, batch_chains):
        n_chain = min(batch_chains, n_trials - start)
        states = bank.sample(seed_base + start, n_samples=K * T,
                             n_chains=n_chain)
        # [chain, temporal read, scalar] -> successes among K*T observations.
        counts[start:start+n_chain] = states.sum(axis=1, dtype=np.uint16)
    return counts


def _trial_counts_spatial(bank: NativeProbabilityBank, K: int, T: int,
                          n_trials: int, batch_chains: int, seed_base: int):
    counts = np.empty((n_trials, bank.shape[-1]), dtype=np.uint16)
    for start in range(0, n_trials, batch_chains):
        n_chain = min(batch_chains, n_trials - start)
        states = bank.sample(seed_base + start, n_samples=T,
                             n_chains=n_chain)
        # [chain, time, replica, scalar] -> all K*T replica observations.
        counts[start:start+n_chain] = states.sum(axis=(1, 2), dtype=np.uint16)
    return counts


def _metrics(counts, n_obs, target, realized, mode, K, T, logical_nodes,
             free_nodes, clamped_nodes, sample_steps,
             bias_fields, bias_bytes, parallel_step_read_width, acc_bits,
             local_counter_bits):
    estimate = counts.astype(np.float64) / n_obs
    err = estimate - target[None, :]
    analytic_variance = realized * (1.0 - realized) / n_obs
    analytic_mse = analytic_variance + (realized - target) ** 2
    empirical_mse = float(np.mean(err ** 2))
    target_mse = float(np.mean(analytic_mse))
    interior = (target > 0) & (target < 1)
    return {
        "mode": mode, "K_replicas": K, "T_parallel_samples": T,
        "observations_per_scalar": n_obs,
        "logical_pbits_total": logical_nodes,
        "logical_pbits_free": free_nodes,
        "logical_pbits_clamped_endpoints": clamped_nodes,
        # The pinned JAX backend observes state after the warmup, then performs
        # n_samples-1 further sweeps. With n_warmup=1 and steps_per_sample=1,
        # the first observation is emitted by the only warmup sweep: total sweeps
        # equal n_samples, not n_samples+1.
        "sample_emitting_parallel_steps": sample_steps,
        "first_observation_after_warmup_sweeps": 1,
        "additional_nonemitting_warmup_parallel_steps": 0,
        "parallel_steps_total": sample_steps,
        "logical_node_update_events_first_observation_sweep": free_nodes,
        "logical_node_update_events_after_first_observation": free_nodes * max(0, sample_steps - 1),
        "logical_node_update_events_total": free_nodes * sample_steps,
        "read_values_total_including_clamps": logical_nodes * sample_steps,
        "read_values_per_parallel_step": parallel_step_read_width,
        "float32_bias_fields": bias_fields,
        "float32_bias_storage_bytes_simulator": bias_bytes,
        "unique_bias_values_for_fanout": int(np.unique(realized[interior]).size),
        "target_grid_points_including_endpoints": int(target.size),
        "interior_target_points": int(interior.sum()),
        "float32_realized_probability_rmse": float(np.sqrt(np.mean((realized-target)**2))),
        "mean_bias_vs_target": float(np.mean(err)),
        "mae_vs_target": float(np.mean(np.abs(err))),
        "rmse_vs_target": float(np.sqrt(empirical_mse)),
        "analytic_rmse_including_float32_bias": float(np.sqrt(target_mse)),
        "empirical_mse_over_analytic": empirical_mse / target_mse if target_mse else 1.0,
        "exact_final_accumulator_bits_per_scalar": acc_bits,
        "parallel_local_counter_bits_per_scalar": local_counter_bits,
        "beta": 1.0,
        "couplings": 0,
        "endpoint_policy": "THRML clamped SpinNodes at p=0 and p=1",
        "field_dtype": "float32",
    }


def run(out_dir: Path, n_trials: int = 1000, batch_chains: int = 100,
        n_targets: int = 257, parallel_T=(1, 4, 16), seed: int = 41027):
    if n_targets < 3 or n_trials < 1 or batch_chains < 1:
        raise ValueError("need >=3 targets and positive trial and chain batch counts")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = np.linspace(0.0, 1.0, n_targets, dtype=np.float64)
    base = NativeProbabilityBank(target, beta=1.0)
    realized_temporal = np.asarray(base.realized_probabilities).reshape(-1)
    rows = []
    controls = []
    for K in (2, 4, 8):
        repeated = np.broadcast_to(target[None, :], (K, target.size)).copy()
        spatial = NativeProbabilityBank(repeated, beta=1.0)
        realized_spatial = np.asarray(spatial.realized_probabilities)
        # All replicas for a scalar receive the same float32 logit field.
        spatial_realized_per_scalar = realized_spatial[0]
        for T in parallel_T:
            if T < 1:
                raise ValueError("parallel T must be positive")
            n_obs = K * T
            raw_t = _trial_counts_temporal(base, K, T, n_trials, batch_chains,
                                           seed + 10000*K + 100*T)
            raw_s = _trial_counts_spatial(spatial, K, T, n_trials, batch_chains,
                                          seed + 20000*K + 100*T)
            raw_path = out_dir / f"raw_K{K}_T{T}.npz"
            np.savez_compressed(raw_path, targets=target, temporal_success_counts=raw_t,
                                spatial_success_counts=raw_s, observations_per_scalar=n_obs)
            accum_bits = int(math.ceil(math.log2(n_obs + 1)))
            t_sample_steps = n_obs
            s_sample_steps = T
            rows.append(_metrics(
                raw_t, n_obs, target, realized_temporal,
                "one_temporal_pbit", K, T,
                logical_nodes=target.size,
                free_nodes=base.free_indices.size,
                clamped_nodes=base.clamped_indices.size,
                sample_steps=t_sample_steps,
                bias_fields=base.free_indices.size,
                bias_bytes=base.biases.nbytes,
                parallel_step_read_width=target.size,
                acc_bits=accum_bits,
                local_counter_bits=accum_bits,
            ))
            rows.append(_metrics(
                raw_s, n_obs, target, spatial_realized_per_scalar,
                "K_spatial_same_probability_replicas", K, T,
                logical_nodes=K*target.size,
                free_nodes=spatial.free_indices.size,
                clamped_nodes=spatial.clamped_indices.size,
                sample_steps=s_sample_steps,
                bias_fields=spatial.free_indices.size,
                bias_bytes=spatial.biases.nbytes,
                parallel_step_read_width=K*target.size,
                acc_bits=accum_bits,
                local_counter_bits=K*int(math.ceil(math.log2(T + 1))) + accum_bits,
            ))
            # A zero-coupling block update sweeps every free logical node once.
            # At the matched K*T observation budget, node-update events must agree.
            temporal_update_events = base.free_indices.size * t_sample_steps
            spatial_update_events = spatial.free_indices.size * s_sample_steps
            assert temporal_update_events == spatial_update_events, (
                f"matched update-count mismatch: temporal={temporal_update_events}, "
                f"spatial={spatial_update_events}, K={K}, T={T}")
            controls.extend([
                {"K": K, "T": T, "mode": "one_temporal_pbit",
                 **base.controls(), "observation_budget_per_scalar": n_obs,
                 "sampler_warmup_steps": 1, "sampler_steps_per_sample": 1,
                 "total_parallel_sweeps_for_n_samples": "n_samples",
                 "first_observation_is_emitted_after_the_warmup_sweep": True,
                 "parallel_read_width": target.size},
                {"K": K, "T": T, "mode": "K_spatial_same_probability_replicas",
                 **spatial.controls(), "observation_budget_per_scalar": n_obs,
                 "sampler_warmup_steps": 1, "sampler_steps_per_sample": 1,
                 "total_parallel_sweeps_for_n_samples": "n_samples",
                 "first_observation_is_emitted_after_the_warmup_sweep": True,
                 "parallel_read_width": K*target.size},
            ])

    det_rows = []
    for K in (2, 4, 8):
        bits, decoded = positional_code(target, K)
        error = decoded - target
        det_rows.append({
            "K_bits": K, "deterministic_bit_cells_per_scalar": K,
            "stochastic_logical_pbits": 0, "sampling_transitions": 0,
            "positive_binary_positional_coefficients": json.dumps(
                [(2**j) / ((1 << K)-1) for j in range(K)]),
            "mean_bias": float(np.mean(error)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "endpoint_errors_zero": bool(decoded[0] == 0.0 and decoded[-1] == 1.0),
            "interpretation": "deterministic positive positional binary code; not p-bit replicas",
        })
        np.savez_compressed(out_dir / f"positional_K{K}.npz", targets=target,
                            bits=bits, decoded=decoded)

    stats_path = out_dir / "matched_statistics.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    controls_path = out_dir / "simulator_controls.json"
    controls_path.write_text(json.dumps(controls, indent=2) + "\n")
    with (out_dir / "positional_deterministic.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(det_rows[0]))
        writer.writeheader()
        writer.writerows(det_rows)
    summary = {
        "config": {"n_trials": n_trials, "batch_chains": batch_chains,
                   "n_targets": n_targets, "parallel_T": list(parallel_T),
                   "K_replicas": [2, 4, 8], "seed": seed},
        "result": {"matched_rows": len(rows), "deterministic_rows": len(det_rows),
                   "observation_budgets": sorted({r["observations_per_scalar"] for r in rows}),
                   "max_abs_empirical_mse_ratio_error": max(
                       abs(r["empirical_mse_over_analytic"] - 1) for r in rows),
                   "energy": "not estimated: physical THRML device energy is unavailable",
                   "latency": "parallel-step counts reported; no physical sampling frequency available"},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest(out_dir / "manifest.json", summary["config"], summary["result"])
    return rows, det_rows, summary


def write_report(path: Path, rows, det_rows, summary):
    max_ratio_error = summary["result"]["max_abs_empirical_mse_ratio_error"]
    lines = [
        "# Native THRML matched-resource experiment",
        "",
        "## Protocol",
        "",
        "This experiment uses `NativeProbabilityBank` and the official THRML `SpinGibbsConditional` / `FactorSamplingProgram` sampler. For each target p in an endpoint-inclusive uniform grid, one temporal node is sampled K*T times, while K independent nodes programmed to the same p are sampled T parallel steps. Both estimate p from K*T observations per scalar. This is an identical-observation-budget comparison, not a deterministic positional code comparison.",
        "",
        "The positional positive-binary baseline is reported in `positional_deterministic.csv`: K bits encode an integer in [0,2^K-1] with normalized powers-of-two coefficients. It uses no stochastic THRML nodes or inference sampling transitions. It is kept separate because its mechanism and cost differ from replica sampling.",
        "",
        "## Simulator accounting",
        "",
        "- Each THRML logical p-bit is one `SpinNode`; all p=0 and p=1 values use official clamped nodes. Endpoints are counted in logical nodes and read outputs, but excluded from free-node updates and float32 bias storage.",
        "- Interior control fields are float32 (4 bytes each) in this simulator. Spatial replicas therefore store K times as many logical control fields; `unique_bias_values_for_fanout` notes that equal values could be fanned out, but no physical sharing circuit is modeled.",
        "- A parallel step updates every free node in its block. Read counts include clamped endpoint states returned by the simulator.",
        "- Verified in pinned source `vendor/thrml/thrml/block_sampling.py`, `_sample_with_observation_jax`: the first record is observed immediately after warmup; only `n_samples-1` further iterations run. With `n_warmup=1`, `n_samples=S`, and `steps_per_sample=1`, one warmup sweep emits the first observation, followed by S-1 sweeps. Total Gibbs sweeps are exactly S, not S+1. First-sweep update events are included in the total, not added again.",
        "- Exact p-estimator accumulators need ceil(log2(K*T+1)) bits per scalar. Spatial local counters plus final combine width are reported separately as a logical implementation estimate, not a hardware measurement.",
        "- THRML stores floating-point control parameters, so probability/control precision is not compressed by this result. No physical switching rate or energy-per-transition data are available; energy is not estimated.",
        "",
        "## Result",
        "",
        f"Across {len(rows)} native simulator conditions, the largest absolute deviation of empirical MSE/theory from one was {max_ratio_error:.4f}. Spatial replication and temporal sampling therefore follow the same Bernoulli estimation law at equal K*T observation budgets. The expected distinction is implementation accounting: spatial replication has K times the logical p-bit population and K times the read width per parallel step, while temporal sampling has K times the sample-step depth.",
        "",
        "The K=2/4/8 deterministic positional codes have separate quantization errors in `positional_deterministic.csv`; they should be compared with storage and latency criteria, not presented as matched stochastic-transition methods.",
        "",
        "## Files",
        "",
        "- `raw_K*_T*.npz`: raw success counts by Monte Carlo chain and probability target for both native schedules.",
        "- `matched_statistics.csv`: error and logical control/update/read/accumulator accounting.",
        "- `simulator_controls.json`: THRML bank counts, beta, endpoint policy, dtype, and backend.",
        "- `positional_K*.npz`, `positional_deterministic.csv`: exact deterministic code bits, decode, and quantization error.",
        "- `manifest.json`: source commit, package versions, platform, hashes, config, and scope.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/thrml_spw/matched_resources"))
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--batch-chains", type=int, default=100)
    parser.add_argument("--targets", type=int, default=257)
    parser.add_argument("--parallel-T", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--seed", type=int, default=41027)
    args = parser.parse_args()
    rows, det_rows, summary = run(args.out, args.trials, args.batch_chains,
                                  args.targets, tuple(args.parallel_T), args.seed)
    write_report(args.out / "REPORT.md", rows, det_rows, summary)
    print(f"wrote native THRML matched-resource results under {args.out}")
    print(f"conditions={len(rows)}; max |MSE ratio - 1|="
          f"{summary['result']['max_abs_empirical_mse_ratio_error']:.5f}")


if __name__ == "__main__":
    main()
