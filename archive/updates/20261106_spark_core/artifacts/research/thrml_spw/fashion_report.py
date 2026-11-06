"""Larger-network figures and explicit representation-specific storage ledger."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .layer_budgets import write_csv


ROOT = Path("results/thrml_spw")


def read(path):
    with (ROOT/path).open() as f:
        return list(csv.DictReader(f))


def main():
    out = ROOT/"fashion_native"
    summary = read("fashion_native/summary.csv")
    rows = read("fashion_native/test_metrics.csv")
    variance = read("fashion_native/weight_variance.csv")
    selection = json.loads((out/"selection.json").read_text())["routes"]
    manifest = json.loads((out/"manifest.json").read_text())
    lookup = {(r["route"], r["condition"]): r for r in summary}
    routes = ["positive_direct", "positive_range", "signed_range"]
    ledger = []
    # Baseline metadata is derived per representation, not copied from the raw
    # route-wide native metadata column in the experimental metrics table.
    for model in range(5):
        for route in routes:
            controls = manifest["result"]["controls"][f"model{model}_{route}"]
            free = sum(c["free_pbits"] for c in controls)
            native_metadata = sum(c.get("metadata_bits", 0) for c in controls)
            for condition in sorted({r["condition"] for r in rows if r["route"] == route}):
                native = condition.endswith("_native")
                if native:
                    code_bits, metadata, bias_bits, modes = free*32, native_metadata, 74*32, 13184*2
                elif condition == "FP32":
                    code_bits, metadata, bias_bits, modes = 13184*32, 0, 74*32, 0
                elif condition == "FP16_rounded_values":
                    code_bits, metadata, bias_bits, modes = 13184*16, 0, 74*16, 0
                else:
                    bits = int(condition.split("_")[0].replace("affine", ""))
                    groups = 74 if condition.endswith("per_output") else 2
                    code_bits, metadata, bias_bits, modes = 13184*bits, groups*2*32, 74*32, 0
                group = [r for r in rows if int(r["model_seed"]) == model and r["route"] == route and r["condition"] == condition]
                ledger.append({"model_seed": model, "route": route, "condition": condition,
                    "logical_pbit_nodes": 13184 if native else 0, "free_nodes": free if native else 0,
                    "control_field_or_numeric_code_bits": code_bits, "range_or_quantizer_metadata_bits": metadata,
                    "neural_bias_bits": bias_bits, "endpoint_mode_bits_assumed": modes,
                    "declared_digital_subtotal_excludes_pbit_states": code_bits+metadata+bias_bits+modes,
                    "sequential_sampling_rounds": int(group[0]["T0"])+int(group[0]["T1"]),
                    "isolated_free_updates": int(group[0]["free_updates_per_isolated_example"]),
                    "storage_scope": "Native: field subtotal plus explicit hypothetical endpoint modes; digital: nominal packed values. Not total Python memory or silicon area."})
    write_csv(out/"storage_ledger.csv", ledger)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {"positive_direct": "tab:blue", "positive_range": "tab:orange", "signed_range": "tab:purple"}
    for route in routes:
        ax = axes[0, 1] if route == "signed_range" else axes[0, 0]
        for condition, marker in (("selected_native", "o"), ("uniform256_native", "s"), ("matched_updates_ceil_native", "x")):
            r = lookup[(route, condition)]
            ax.errorbar(float(r["free_updates_per_isolated_example_mean"]), 100*float(r["accuracy_mean"]),
                yerr=100*float(r["accuracy_ci95_margin"]), fmt=marker, color=colors[route],
                label=route if condition == "selected_native" else None)
            if condition == "selected_native":
                s = selection[route]["selected"]
                ax.annotate(f"({s['T0']},{s['T1']})", (float(r["free_updates_per_isolated_example_mean"]), 100*float(r["accuracy_mean"])),
                            xytext=(5, 9) if route == "positive_range" else (-5, 9), textcoords="offset points",
                            ha="left" if route == "positive_range" else "right", fontsize=8)
    for ax, route in ((axes[0, 0], "positive_direct"), (axes[0, 1], "signed_range")):
        ax.axhline(100*float(lookup[(route, "FP32")]["accuracy_mean"]), color="black", ls="--", label="FP32")
        ax.set(xscale="log", xlabel="Free-node updates per isolated example", ylabel="Test accuracy (%)")
        ax.legend(fontsize=8)
    axes[0, 0].set_title("Same positive weights, different encodings")
    axes[0, 1].set_title("Signed weights with affine p-bit encoding")
    for i, route in enumerate(("positive_direct", "signed_range")):
        for j, condition in enumerate(("FP32", "affine4_per_output")):
            r = lookup[(route, condition)]
            axes[1, 0].bar(i+(j-.5)*.32, 100*float(r["accuracy_mean"]), width=.32,
                          yerr=100*float(r["accuracy_ci95_margin"]), color=["tab:blue", "tab:orange"][j], label=condition if i == 0 else None)
    axes[1, 0].set(xticks=[0, 1], xticklabels=["Positive projected", "Signed control"], ylim=(70, 90), ylabel="Test accuracy (%)",
                    title="Model constraint before stochastic encoding")
    axes[1, 0].legend(fontsize=8)
    for j, route in enumerate(("positive_direct", "positive_range")):
        v = [np.mean([float(r["mean_single_observation_weight_variance"]) for r in variance if r["route"] == route and int(r["layer"]) == i]) for i in range(2)]
        axes[1, 1].bar(np.arange(2)+(j-.5)*.34, v, width=.34, color=colors[route], label=route)
    axes[1, 1].set(xticks=[0, 1], xticklabels=["First layer", "Output layer"], ylabel="Mean single-draw weight variance",
                    title="Range metadata reduces sampling variance")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Fashion-MNIST: 13,184 native weight nodes; five trained models × five sampling seeds")
    fig.supxlabel("Native points: circle = selected allocation; square = uniform256; cross = matched-update ceiling", fontsize=9)
    fig.savefig(ROOT/"fashion_evidence.png", dpi=180)
    fig.savefig(ROOT/"fashion_evidence.pdf")
    plt.close(fig)
    lines = ["# Larger-network and second-dataset native THRML evidence\n",
        "Five 196->64->10 Fashion-MNIST networks use 13,184 matrix weights (5.57 times "
        "the first MNIST experiment) and 74 deterministic neural biases. Inputs are 14x14 "
        "average-pooled images normalized using the 55,000 training examples only. A 5,000-image "
        "validation set selects checkpoints and sample budgets; all 10,000 test images are "
        "held out from selection. Each selected native configuration is tested with five "
        "sampling seeds per trained model. Intervals use 1.96*SD/sqrt(5) across trained-model "
        "means after averaging sampling seeds, with paired differences formed by model.\n",
        "## Separate model quality from representation quality\n",
        "Positive and signed-control models start from exactly the same positive weights and "
        "follow identical minibatch orders. The control releases matrix projection; both train "
        "for 20 epochs and select the lowest validation CE checkpoint. This measures the effect "
        "of projection under this training protocol, not a universal limit on positive networks.\n",
        "| Trained model | FP32 | Per-output affine4 numerical baseline |\n|---|---:|---:|"]
    for label, route in (("Positive projected", "positive_direct"), ("Signed released control", "signed_range")):
        lines.append(f"| {label} | {100*float(lookup[(route,'FP32')]['accuracy_mean']):.3f}% | {100*float(lookup[(route,'affine4_per_output')]['accuracy_mean']):.3f}% |")
    lines += ["\n## Native results and predeclared gates\n",
        "Direct encoding programs p=w for positive weights. Affine encoding instead programs "
        "p=(w-l)/(u-l) per output column, then accumulates scaled stochastic outputs and "
        "adds the deterministic offset before ReLU. Constant columns use p=0 clamps and "
        "their stored constant. It is still one logical node per weight, but p is not the "
        "original signed weight. All random states come from official THRML.\n",
        "For each route, a 16-pair validation sweep minimizes free-node updates subject to "
        "mean accuracy loss at most 1.0 pp and CE increase at most 0.05 from its deterministic "
        "baseline. Failed gates remain failures; the best-CE candidate is then tested as a "
        "diagnostic, not silently promoted to success.\n",
        "| Route | Validation gate | T first / output | Test accuracy | Paired change vs FP32 (pp) | Free updates / isolated example | Sequential sampling rounds |\n|---|---|---:|---:|---:|---:|---:|"]
    for route in routes:
        s, r = selection[route], lookup[(route, "selected_native")]
        t0, t1 = s["selected"]["T0"], s["selected"]["T1"]
        lines.append(f"| {route} | {'pass' if s['gate_passed_on_validation'] else 'FAIL; diagnostic fallback'} | {t0}/{t1} | "
            f"{100*float(r['accuracy_mean']):.3f}% +/-{100*float(r['accuracy_ci95_margin']):.3f} | "
            f"{100*float(r['paired_accuracy_delta_vs_FP32']):.3f} +/-{100*float(r['paired_delta_ci95_margin']):.3f} | "
            f"{float(r['free_updates_per_isolated_example_mean']):,.0f} | {t0+t1} |")
    lines += ["\nFixed256 and a per-model uniform ceiling of the selected actual free-update "
        "budget are also evaluated. The ceiling uses control counts only, not test labels. "
        "Its paired comparison with the selected allocation is:\n",
        "| Route | Fixed256 accuracy | Matched-update ceiling accuracy | Selected minus matched (pp) |\n|---|---:|---:|---:|"]
    for route in routes:
        a, b, c = [lookup[(route, key)] for key in ("uniform256_native", "matched_updates_ceil_native", "selected_native")]
        lines.append(f"| {route} | {100*float(a['accuracy_mean']):.3f}% | {100*float(b['accuracy_mean']):.3f}% | "
            f"{100*float(c['paired_accuracy_delta_vs_matched_ceil']):.3f} +/-{100*float(c['paired_matched_ci95_margin']):.3f} |")
    lines += ["\n## Precision and resource accounting\n",
        "For independent draws, affine weight variance is (w-l)(u-w)/T, compared with "
        "w(1-w)/T for direct [0,1] encoding. For 0<=l<=w<=u<=1, affine variance cannot exceed "
        "direct variance; neural accuracy still depends on input structure and nonlinearities. "
        "weight_variance.csv records expected variance and finite-control rounding separately.\n",
        "Affine routes store two float32 values per output column: 148 values, 4736 bits or 592 "
        "bytes, plus 74 deterministic biases. Their input-sum correction and output scaling are "
        "external float32 arithmetic. The 1,024-sample output layer is a sequential bottleneck; "
        "a large reduction in total node evaluations does not imply the same latency reduction.\n",
        "storage_ledger.csv gives representation-specific numeric/controller subtotals. FP32 "
        "has no affine-range metadata; quantized baselines count their implemented min/span "
        "metadata. The raw metrics column metadata_bits describes the fitted native route, "
        "including when that route is numerically compared with a conventional baseline; "
        "it is not used to charge native metadata to FP32 or FP16 in the storage ledger.\n",
        "Native fields remain float32, with separate assumed two-bit endpoint modes, affine "
        "metadata and neural biases. These are declared subtotals, excluding requested "
        "probability arrays, program copies, Python objects, sample states and all other buffers. "
        "They are not total simulator memory, physical controller area or achieved compression. "
        "P-bit nodes and digital storage bits are separate resources. Conventional low-bit "
        "baselines are numerical simulations, not optimized low-bit kernels.\n",
        "One bank state is shared across each 128-input batch. Actual dataset free updates "
        "equal isolated updates times 79 batches. Fields/ranges are programmed once per model "
        "and reused in software; physical programming/retention, readout, accumulator and "
        "energy costs remain unmeasured. These frozen-model routes have not yet been jointly "
        "validated with finite-width field controllers or integer accumulators.\n",
        "## Reproduction\n",
        "Train/export with .venv/bin/python -m research.thrml_spw.train_fashion_positive. "
        "Use .venv-thrml for test_range_native, fashion_native, verify_fashion and fashion_report. "
        "Protocols, exact input hashes, training history, raw validation/test logits, controls, "
        "selection gates and the pinned official simulator manifest accompany this report. "
        "The broader goal remains active for native Transformer work, stronger training and "
        "physical-system characterization.\n"]
    (ROOT/"FASHION_REPORT.md").write_text("\n".join(lines)+"\n")
    print("Wrote FASHION_REPORT.md, storage_ledger.csv and fashion_evidence.png/.pdf")


if __name__ == "__main__":
    main()
