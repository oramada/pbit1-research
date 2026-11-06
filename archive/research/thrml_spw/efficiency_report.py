"""Build standalone figures and readable reports from native efficiency results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/thrml_spw")


def rows(name):
    with (ROOT/name).open() as f:
        return list(csv.DictReader(f))


def main():
    allocation = json.loads((ROOT/"layer_budgets/selection.json").read_text())
    budget = rows("layer_budgets/summary.csv")
    matched = rows("layer_budget_matched/summary.csv")
    update_matched = rows("layer_update_matched/summary.csv")
    acc_selection = json.loads((ROOT/"accumulator_precision/selection.json").read_text())
    acc = rows("accumulator_precision/summary.csv")
    joint = rows("joint_efficiency/summary.csv")
    t0, t1 = allocation["selected"]["T0"], allocation["selected"]["T1"]
    chosen = next(r for r in budget if int(r["T0"]) == t0 and int(r["T1"]) == t1)
    reference = next(r for r in budget if int(r["T0"]) == 256 and int(r["T1"]) == 256)
    selected_name = acc_selection["selected"]["configuration"]
    values = [16, 32, 64, 128, 256]
    heat = np.array([[next(r["accuracy_mean"] for r in allocation["profiles"] if r["T0"] == a and r["T1"] == b)
                     for b in values] for a in values])*100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    im = axes[0].imshow(heat, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set(xticks=range(5), xticklabels=values, yticks=range(5), yticklabels=values,
                xlabel="Second-layer samples", ylabel="First-layer samples", title="Validation accuracy (%)")
    axes[0].scatter(values.index(t1), values.index(t0), s=180, facecolors="none", edgecolors="red", linewidths=2)
    fig.colorbar(im, ax=axes[0], shrink=.8)
    for r in budget:
        x = float(r["logical_reads_per_isolated_example"])/1000
        y = float(r["accuracy_mean"])*100
        is_chosen = (int(r["T0"]), int(r["T1"])) == (t0, t1)
        axes[1].errorbar(x, y, yerr=float(r["accuracy_ci95_margin"])*100, fmt="o",
                         color="tab:red" if is_chosen else "tab:blue")
        offset = (-8, 7) if int(r["T0"]) == 256 else (5, -14 if is_chosen else 7)
        axes[1].annotate(f"({r['T0']}, {r['T1']})", (x, y), xytext=offset,
                         ha="right" if int(r["T0"]) == 256 else "left", textcoords="offset points")
    for i, r in enumerate(matched):
        x, y = float(r["logical_reads_per_example"])/1000, float(r["uniform_accuracy_mean"])*100
        axes[1].errorbar(x, y, yerr=float(r["uniform_accuracy_ci95_margin"])*100, fmt="s", color="tab:green")
        axes[1].annotate(f"uniform {r['uniform_samples']}", (x, y), xytext=(10, -12-14*i), textcoords="offset points")
    axes[1].set(xlabel="Logical weight reads per isolated example (thousands)", ylabel="Test accuracy (%)",
                title="Five models × five sampling seeds")
    axes[1].grid(alpha=.2)
    fig.suptitle("Native THRML layer allocation; counts are modeled work, not chip energy")
    fig.savefig(ROOT/"layer_budget_evidence.png", dpi=180)
    fig.savefig(ROOT/"layer_budget_evidence.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    names = list(acc_selection["profiles"])
    axes[0].bar(range(len(names)), [acc_selection["profiles"][n]["accuracy"]*100 for n in names],
                color=["tab:red" if n == selected_name else "tab:blue" for n in names])
    axes[0].set(xticks=range(len(names)), xticklabels=names, ylabel="Validation accuracy (%)", ylim=(0, 100),
                title="Fractional precision competes with sum headroom")
    axes[0].tick_params(axis="x", labelrotation=55)
    for i, r in enumerate(joint):
        axes[1].errorbar(i, float(r["accuracy_mean"])*100, yerr=float(r["accuracy_ci95_margin"])*100, fmt="o", color="tab:purple")
    labels = ["FP fields\nFP sums", "6-bit fields\nFP sums", "FP fields\n16-bit sums", "6-bit fields\n16-bit sums"]
    axes[1].axhline(float(reference["accuracy_mean"])*100, color="black", ls="--", label="Uniform256, float32")
    axes[1].set(xticks=range(len(joint)), xticklabels=labels, ylabel="Test accuracy (%)",
                title=f"Joint confirmation at layer samples ({t0}, {t1})")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("THRML-driven arithmetic models; physical hardware costs remain unmeasured")
    fig.savefig(ROOT/"accumulator_joint_evidence.png", dpi=180)
    fig.savefig(ROOT/"accumulator_joint_evidence.pdf")
    plt.close(fig)

    report = ["# Native layer allocation and accumulator research\n",
              "All selections used validation labels only. Test results use all 10,000 images, "
              "five trained models and five independent sampling seeds per model. Intervals "
              "are approximate 1.96*SD/sqrt(5) across trained-model means; paired intervals "
              "use paired model means. Raw logits, frozen rules and manifests are preserved.\n",
              "## Layer allocation\n",
              f"The validation-selected pair is ({t0},{t1}). Test accuracy was "
              f"{100*float(chosen['accuracy_mean']):.3f}% +/-{100*float(chosen['accuracy_ci95_margin']):.3f} pp, "
              f"versus {100*float(reference['accuracy_mean']):.3f}% for uniform256 on the same keys. "
              f"The paired accuracy change was {100*float(chosen['paired_accuracy_delta_vs_uniform256']):.3f} "
              f"+/-{100*float(chosen['paired_delta_ci95_margin']):.3f} pp.\n",
              f"Logical reads per isolated example fell from 606,208 to {int(chosen['logical_reads_per_isolated_example']):,} "
              f"({100*(1-float(chosen['reads_fraction_of_uniform256'])):.2f}% fewer). Serial layer sweeps "
              f"fell from 512 to {t0+t1}. These are distinct modeled resources. Endpoint reads include "
              "clamps; free-update counts are reported separately. Actual dataset sampling shares "
              "one bank state across each batch of 32, and those counts are not per-input energy.\n",
              "The coarse uniform grid leaves unused budget. Exact integer uniform budgets "
              "bracket the selected read count as follows:\n",
              "| Uniform samples | Logical reads | Test accuracy | Selected minus uniform (paired pp) |\n|---|---:|---:|---:|"]
    for r in matched:
        report.append(f"| {r['uniform_samples']} | {int(r['logical_reads_per_example']):,} | "
                      f"{100*float(r['uniform_accuracy_mean']):.3f}% | "
                      f"{100*float(r['selected_minus_uniform_mean']):.3f} +/-{100*float(r['paired_delta_ci95_margin']):.3f} |")
    report += ["\nFree stochastic updates exclude clamped weights, unlike the read count. "
               "The selected allocation averages 184,691.2 free-node updates per isolated "
               "example versus 509,747.2 for uniform256 (63.77% fewer). A separate uniform "
               "comparison brackets this update budget for each model using weights alone.\n",
               "| Uniform update-budget bracket | Test accuracy | Selected minus uniform (paired pp) |\n|---|---:|---:|"]
    for r in update_matched:
        report.append(f"| {r['bracket']} | {100*float(r['uniform_accuracy_mean']):.3f}% | "
                      f"{100*float(r['selected_minus_uniform_accuracy_mean']):.3f} +/-{100*float(r['paired_delta_ci95_margin']):.3f} |")
    report += ["\n## Accumulator precision\n",
               f"Validation selected {selected_name}. Dot products, controls and final activations "
               "remain float32; each sampled dot-product increment is rounded to a fixed-point "
               "lattice before a saturating integer sum. The software containers are int32, "
               "so the nominal 16-bit register count is not measured Python/JAX memory savings.\n",
               "| Format | Test accuracy | Paired change vs float32 (pp) | Sum saturations across test runs |\n|---|---:|---:|---:|"]
    for r in acc:
        report.append(f"| {r['configuration']} | {100*float(r['accuracy_mean']):.3f}% | "
                      f"{100*float(r['paired_accuracy_delta_vs_float32']):.3f} +/-{100*float(r['paired_delta_ci95_margin']):.3f} | "
                      f"{int(r['total_sum_clips']):,} |")
    report += ["\nA sum must retain headroom proportional to T. With B signed bits and F fractional "
               "bits, its positive averaged-output limit before bias is (2^(B-1)-1)/(T*2^F). "
               "At T=256, int16_frac0 supports approximately 128, while int16_frac8 supports "
               "only 0.5. Increasing fractional precision at fixed width can therefore destroy "
               "accuracy through saturation. Register-only peak footprint for 32 simultaneous "
               "output sums is 512 nominal bits at B=16 versus 1024 at B=32, excluding hidden "
               "activations, dot-product temporaries, control fields and all other buffers.\n",
               "Rounding increments can also change their expectation even without saturation. "
               "The native two-p-bit negative control in accumulator_bias/ demonstrates a "
               "limiting output of 0.1875 instead of 0.30 for integer increments. More temporal "
               "samples do not remove that bias. The unsaturated per-layer mean shift is bounded "
               "by half a lattice unit; useful task accuracy is an empirical property.\n",
               "## Joint confirmation\n",
               "The independently chosen six-bit [-4,4] controls, integer accumulator and nonuniform "
               "layer counts were combined without further test-based selection.\n",
               "| Control / sum configuration | Test accuracy | Paired change vs original uniform256 (pp) |\n|---|---:|---:|"]
    for r in joint:
        report.append(f"| {r['configuration']} | {100*float(r['accuracy_mean']):.3f}% | "
                      f"{100*float(r['paired_accuracy_delta_vs_uniform256']):.3f} +/-{100*float(r['paired_delta_ci95_margin']):.3f} |")
    report += ["\nThis supports useful native inference under explicit reduced control/accumulator "
               "precision and fewer logical readouts. It does not establish a physical density, "
               "energy, or latency advantage over digital quantization or an Extropic architecture. "
               "Control retention, physical gate costs, transfer functions and a relevant application "
               "baseline remain required. Confidence stopping and larger architectures remain open.\n"]
    (ROOT/"EFFICIENCY_REPORT.md").write_text("\n".join(report)+"\n")
    print("Wrote EFFICIENCY_REPORT.md and two standalone figure pairs")


if __name__ == "__main__":
    main()
