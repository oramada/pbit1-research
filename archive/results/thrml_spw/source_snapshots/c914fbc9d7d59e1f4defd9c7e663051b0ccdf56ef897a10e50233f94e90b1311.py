"""Exact/native negative control for confidence-selected sample-mean bias."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import binom

from .layer_budgets import write_csv
from .native import NativeProbabilityBank
from .provenance import manifest


def main():
    out = Path("results/thrml_spw/adaptive_bias")
    out.mkdir(parents=True, exist_ok=True)
    requested, n, maximum, trials = .6, 16, 256, 10000
    bank = NativeProbabilityBank(np.array([requested]))
    p = float(bank.realized_probabilities[0])
    samples = bank.sample(830730, maximum, trials)[..., 0]
    initial = samples[:, :n].sum(axis=1)
    final = samples.sum(axis=1)
    k = np.arange(n+1)
    mass = binom.pmf(k, n, p)
    prefix_mean, se = k/n, np.sqrt((k/n)*(1-k/n)/(n-1))
    rows = []
    for name in ("1", "2", "3", "4", "never"):
        stop = np.abs(prefix_mean-.5) > float(name)*se if name != "never" else np.zeros(n+1, dtype=bool)
        conditional_mean = np.where(stop, prefix_mean, (k+(maximum-n)*p)/maximum)
        conditional_variance = np.where(stop, 0, (maximum-n)*p*(1-p)/maximum**2)
        exact_mean = float(mass@conditional_mean)
        exact_mse = float(mass@((conditional_mean-requested)**2+conditional_variance))
        values = np.where(stop[initial], initial/n, final/maximum)
        squared = (values-requested)**2
        mean_se = float(values.std(ddof=1)/np.sqrt(trials))
        mse_se = float(squared.std(ddof=1)/np.sqrt(trials))
        assert abs(values.mean()-exact_mean) < 6*mean_se
        assert abs(squared.mean()-exact_mse) < 6*mse_se
        rows.append({"gamma": name, "requested_probability": requested, "realized_probability": p,
                     "trials": trials, "exact_returned_mean": exact_mean,
                     "exact_bias_vs_requested": exact_mean-requested,
                     "empirical_mean": float(values.mean()), "empirical_mean_se": mean_se,
                     "exact_mse": exact_mse, "empirical_mse": float(squared.mean()),
                     "empirical_mse_se": mse_se,
                     "exact_stop_at16_probability": float(mass@stop)})
    write_csv(out/"metrics.csv", rows)
    np.savez_compressed(out/"prefix_counts.npz", initial16=initial.astype(np.uint16), final256=final.astype(np.uint16))
    manifest(out/"manifest.json", {"n": n, "maximum": maximum, "trials": trials,
             "requested_probability": requested, "seed": 830730,
             "rule": "Compare logits (sample mean, deterministic 0.5); stop at16 if top-two margin exceeds gamma*empirical SE, otherwise return fixed256 mean",
             "scope": "Full native trajectories are generated for exact-law validation; this negative-control run claims no online-work saving"},
             {"rows": rows, "interpretation": "Finite-horizon confidence stopping generally changes the expectation of a returned sample mean. Task calibration does not guarantee unbiased logits or anytime confidence coverage."})
    print(rows, flush=True)


if __name__ == "__main__":
    main()
