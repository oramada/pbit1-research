"""Standalone native adaptive-inference plots and research report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/thrml_spw")


def read(path):
    with (ROOT/path).open() as f:
        return list(csv.DictReader(f))


def main():
    shared = read("adaptive_shared/summary.csv")
    independent = read("adaptive_native/summary.csv")
    bias = read("adaptive_bias/metrics.csv")
    selection = json.loads((ROOT/"adaptive_shared/selection.json").read_text())
    profiles, chosen = selection["profiles"], selection["selected"]
    reference = selection["reference"]
    adaptive = next(r for r in shared if r["condition"] == "adaptive")
    matched = next(r for r in shared if r["condition"] == "matched_ceil")
    raw = np.load(ROOT/"adaptive_shared/test_outputs.npz")
    checkpoints = [16, 32, 64, 128, 256]
    stops = np.concatenate([raw[f"model{model}_seed{seed}_stop"] for model in range(5) for seed in range(5)])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for r in profiles:
        fraction = r["executed_free_updates_mean"]/reference["executed_free_updates_mean"]
        selected = r["gamma"] == chosen["gamma"]
        axes[0, 0].scatter(fraction, r["accuracy_mean"]*100, color="tab:red" if selected else "tab:blue", s=55 if selected else 25)
        if r["gamma"] in ("0", "1", "2", "3", "4", "never"):
            offset = (-3, -11) if r["gamma"] == "never" else (3, 7 if r["gamma"] == "0" else -11)
            axes[0, 0].annotate(r["gamma"], (fraction, r["accuracy_mean"]*100), xytext=offset,
                                 ha="right" if r["gamma"] == "never" else "left", textcoords="offset points", fontsize=9)
    axes[0, 0].set(xlabel="Fraction of fixed-reference updates", ylabel="Validation accuracy (%)",
                    title="Validation-only threshold choice")
    for i, r in enumerate(shared):
        x, y = float(r["mean_fraction_of_full256_updates"]), float(r["accuracy_mean"])*100
        axes[0, 1].errorbar(x, y, yerr=float(r["accuracy_ci95_margin"])*100, fmt="o",
                           color="tab:red" if r["condition"] == "adaptive" else "tab:blue")
        label = {"full256": "fixed256", "adaptive": "adaptive", "matched_floor": "fixed floor", "matched_ceil": "fixed ceiling"}[r["condition"]]
        offset = (-5, 10) if r["condition"] == "full256" else (6, 8 if r["condition"] == "adaptive" else -12-12*(i-2))
        axes[0, 1].annotate(label, (x, y), xytext=offset, textcoords="offset points", ha="right" if r["condition"] == "full256" else "left", fontsize=9)
    axes[0, 1].set(xlabel="Fraction of fixed-reference updates", ylabel="Test accuracy (%)",
                    title="Shared bank: five models × five seeds")
    axes[0, 1].set_xlim(.81, 1.015)
    axes[1, 0].bar(range(5), [100*np.mean(stops == t) for t in checkpoints], color="tab:purple")
    axes[1, 0].set(xticks=range(5), xticklabels=checkpoints, xlabel="Output-layer samples at stopping",
                    ylabel="Test examples (%)", title="Actual online stopping distribution")
    axes[1, 1].plot(range(len(bias)), [float(r["exact_returned_mean"]) for r in bias], "o-", label="Exact stopped expectation")
    axes[1, 1].errorbar(range(len(bias)), [float(r["empirical_mean"]) for r in bias],
                      yerr=[1.96*float(r["empirical_mean_se"]) for r in bias], fmt="x", label="THRML: 10,000 trials")
    axes[1, 1].axhline(.6, color="black", ls="--", label="Programmed p = 0.6")
    axes[1, 1].set(xticks=range(len(bias)), xticklabels=[r["gamma"] for r in bias], xlabel="Margin / SE threshold",
                    ylabel="Returned sample mean", title="Stopping can bias a numerical estimate")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Online inference driven by official THRML sampling; physical costs unmeasured")
    fig.savefig(ROOT/"adaptive_evidence.png", dpi=180)
    fig.savefig(ROOT/"adaptive_evidence.pdf")
    plt.close(fig)

    rows = read("adaptive_shared/test_metrics.csv")
    mean_second = lambda condition: float(np.mean([int(r["second_bank_observations"]) for r in rows if r["condition"] == condition]))
    moment_ratio = 6*mean_second("adaptive")/mean_second("full256")
    extra_m2_bits = 10*32
    lines = ["# Online single-p-bit inference in official THRML\n",
        "This experiment adds genuine sampler stopping and compaction. The first layer "
        "uses 64 samples before ReLU. The output layer checks its running mean and empirical "
        "variance at 16,32,64,128,256 samples. A case stops when the top-two margin exceeds "
        "gamma times its conditional output SE, or at the final cap. All weights remain "
        "direct w=p probabilities; exact endpoints are clamped. Controls and external "
        "arithmetic use float32. These are the same five positive-weight MNIST models.\n",
        "## Main result: a shared weight bank\n",
        "One sampled bank state serves up to 128 input rows. Unfinished cases are compacted "
        "between actual THRML chunk calls; stopped cases are not sampled further. Partial "
        "batches and their padded arithmetic are counted. Each bank is uncoupled, so resetting "
        "it and running warmup1 gives a new stationary sample; every chunk observation costs "
        "one sweep. This reset protocol must not be generalized to coupled banks.\n",
        f"Validation over 5,000 images and five trained models selected gamma={chosen['gamma']} "
        "from ten candidates. The rule minimized executed free-node updates while keeping "
        "accuracy within 0.2 pp and CE within 0.02 of the fixed reference on validation only. "
        "Selection was frozen before full test evaluation.\n",
        "All 10,000 test images were evaluated for five trained models and five sampling "
        "seeds per model. Intervals use 1.96*SD/sqrt(5) across trained-model means after "
        "averaging sampling seeds; paired changes use paired model means.\n",
        "| Condition | Test accuracy | Updates / fixed reference | Mean output samples |\n|---|---:|---:|---:|"]
    for r in shared:
        lines.append(f"| {r['condition']} | {100*float(r['accuracy_mean']):.3f}% +/-{100*float(r['accuracy_ci95_margin']):.3f} pp | "
                     f"{100*float(r['mean_fraction_of_full256_updates']):.2f}% | {float(r['mean_output_samples']):.2f} |")
    lines += [f"\nAdaptive inference reduced total stochastic free-node updates by "
        f"{100*(1-float(adaptive['mean_fraction_of_full256_updates'])):.2f}%, including its fixed "
        f"first layer and padding. Its paired accuracy change was "
        f"{100*float(adaptive['paired_accuracy_delta_vs_full256']):.3f} +/-"
        f"{100*float(adaptive['paired_delta_ci95_margin']):.3f} pp.\n",
        f"Uniform integer output budgets bracket each run's actual adaptive bank-observation "
        f"count, using counters only. Adaptive accuracy exceeded the upper-budget comparator "
        f"by {100*float(matched['adaptive_minus_condition_accuracy_mean']):.3f} +/-"
        f"{100*float(matched['adaptive_minus_condition_ci95_margin']):.3f} pp. The first-layer "
        "schedule and bank-sharing width were identical across these comparisons.\n",
        "Compaction changes later batch membership and therefore future random paths. Test "
        "decisions are audited against their actual online moment traces, not against a "
        "different reference trajectory. Actual stage counters, active counts and padding "
        "are stored. A no-stopping native test equals the fixed path exactly.\n",
        "## Independent-case verification\n",
        "A second implementation gives every example its own independent chain. Each retained "
        "path survives compaction exactly: online outputs and stop times equal complete-path "
        "replay bit for bit, and the first layer is recomputed on the online branch. This "
        "verification used five trained models and one sampling seed per model.\n",
        "| Condition | Test accuracy | Updates / fixed reference |\n|---|---:|---:|"]
    for r in independent:
        lines.append(f"| {r['condition']} | {100*float(r['accuracy_mean']):.3f}% | {100*float(r['mean_fraction_of_full256_updates']):.2f}% |")
    lines += ["\nIndependent chains sacrifice the weight-state reuse of batch-shared inference. "
        "Their absolute dataset update counts are not directly comparable to shared-bank "
        "counts. Each protocol has its own fixed and matched-budget references. This work "
        "does not multiply percentage savings from experiments with different batch/key rules.\n",
        "## Statistical and systems limits\n",
        "The SE is conditional on a frozen stochastic first-layer output. It excludes that "
        "layer's error. Output weight columns are independent here, but repeated looks and "
        "selection of the leading logits prevent interpreting gamma as an anytime confidence "
        "guarantee. Confidence-calibrated accuracy does not imply unbiased returned logits.\n",
        "An exact/native scalar negative control programs p=0.6 and compares its sample mean "
        "with a deterministic rival of 0.5. It stops at 16 when the same margin/SE criterion "
        "passes, otherwise at 256. At gamma=3, its exact returned expectation is 0.61389128, "
        "despite a programmed mean of 0.6. Ten thousand actual THRML trajectories agree with "
        "the exact finite-horizon law. That control generates full traces for statistical "
        "validation and claims no online-work savings. The underlying fixed-horizon "
        "probability representation remains valid; this classification policy does not "
        "supply an unbiased scalar-weight display or a guaranteed scalar error bound.\n",
        f"Confidence also adds storage and computation. Each pending case has a 10-entry "
        f"float32 M2 array ({extra_m2_bits} nominal bits) alongside its mean and frozen hidden "
        "activation; case IDs, scheduling and audit histories add other buffers. A Welford "
        "moment update uses two subtractions, two additions, one multiplication and one "
        "division per output value. Counting these as six distinct arithmetic operations, "
        f"the adaptive output-statistics work is about {moment_ratio:.2f} times the simple "
        "one-add-per-value accumulation work of an ordinary fixed sampler, before margin/SE "
        "checks. Divides and adds need not have equal hardware cost. The fixed statistical "
        "reference here also tracks moments for auditing; an ordinary deployed fixed layer "
        "would not need that variance overhead. Dot products and first-layer work are separate.\n",
        "The demonstrated gain is fewer stochastic updates at useful task quality. Physical "
        "energy, latency and density depend on controller retention, fanout/readout, multiply "
        "and moment hardware, memory and scheduling. Those advantages remain unproven. The "
        "adaptive route has not yet been jointly validated with the earlier six-bit controls "
        "and integer accumulators; separate successes are not assumed to compose.\n",
        "## Reproduction\n",
        "Use the pinned .venv-thrml environment. Run the test_adaptive_native and "
        "test_adaptive_shared unittest modules, then adaptive_study, adaptive_bias, "
        "adaptive_shared_study, verify_adaptive, verify_adaptive_shared and adaptive_report "
        "as research.thrml_spw modules. Protocols, selections, raw moment/output archives, "
        "source hashes and manifests are stored in the corresponding result directories. "
        "The long-horizon goal remains active for larger architectures, broader training "
        "and an implementable hardware/control comparison.\n"]
    (ROOT/"ADAPTIVE_REPORT.md").write_text("\n".join(lines)+"\n")
    print("Wrote ADAPTIVE_REPORT.md and adaptive_evidence.png/.pdf")


if __name__ == "__main__":
    main()
