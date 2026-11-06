"""Report selective native Transformer execution without broadening its claim."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


ROOT = Path("results/thrml_spw")


def read(path):
    with (ROOT/path).open() as f:
        return list(csv.DictReader(f))


def main():
    prepared = json.loads((ROOT/"transformer_preparation/manifest.json").read_text())
    native = json.loads((ROOT/"transformer_projection_native/manifest.json").read_text())
    selection = json.loads((ROOT/"transformer_replay/selection.json").read_text())
    summary = read("transformer_replay/summary.csv")
    linear = read("transformer_projection_native/linear_metrics.csv")
    test = {r["condition"]: r for r in summary if r["split"] == "test"}
    chosen = selection["selected"]["samples"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for split, color in (("validation", "tab:blue"), ("test", "tab:orange")):
        values, theory = [], []
        for t in (16, 64, 256):
            group = [r for r in linear if r["split"] == split and int(r["samples"]) == t]
            values.append(np.mean([float(r["linear_mse"]) for r in group]))
            theory.append(float(group[0]["predicted_sampling_mse"]))
        axes[0].loglog([16, 64, 256], values, "o-", color=color, label=f"{split}: native")
        axes[0].loglog([16, 64, 256], theory, "--", color=color, alpha=.6, label=f"{split}: variance/T")
    axes[0].set(xlabel="Samples per projection invocation", ylabel="Projection output MSE", title="589,824 actual THRML weight nodes")
    axes[0].legend(fontsize=8)
    for t in (16, 64, 256):
        r = test[f"T{t}"]
        value = float(r["relative_perplexity_from_mean_ce"])
        lo, hi = float(r["relative_perplexity_ci95_lower"]), float(r["relative_perplexity_ci95_upper"])
        axes[1].errorbar(t, value, yerr=[[value-lo], [hi-value]], fmt="o", color="tab:red" if t == chosen else "tab:blue")
    for name, label, style in (("FP32", "FP32", "--"), ("affine4", "Affine 4-bit", ":")):
        axes[1].axhline(float(test[name]["relative_perplexity_from_mean_ce"]), color="black" if name == "FP32" else "tab:green", ls=style, label=label)
    axes[1].axvline(chosen, color="tab:red", ls=":", alpha=.4)
    axes[1].set(xscale="log", xticks=[16, 64, 256], xticklabels=[16, 64, 256], xlabel="Samples", ylabel="Test perplexity / FP32 perplexity",
                title="Downstream pretrained-model evaluation")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks([16, 64, 256])
        ax.set_xticklabels(["16", "64", "256"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.2)
    fig.suptitle("DistilGPT-2: one projection converted (0.72% of model); five sampling seeds")
    fig.savefig(ROOT/"transformer_projection_evidence.png", dpi=180)
    fig.savefig(ROOT/"transformer_projection_evidence.pdf")
    plt.close(fig)
    controls = native["result"]["controls"]
    lines = ["# Selective native THRML projection inside a pretrained Transformer\n",
        f"A genuine pretrained DistilGPT-2 model has {prepared['total_parameters']:,} unique parameters. "
        f"This experiment converts only `{prepared['config']['target_module']}`: "
        f"{prepared['converted_matrix_parameters']:,} matrix weights, or {100*prepared['converted_fraction']:.3f}% "
        "of the model. All other parameters and the selected projection's bias remain deterministic. "
        "This is a selective-operator result, not full-model p-bit conversion.\n",
        f"Model source: [pinned pretrained checkpoint](https://huggingface.co/distilbert/distilgpt2/tree/{prepared['config']['model_revision']}). "
        f"Data source: [pinned WikiText-2 raw splits](https://huggingface.co/datasets/Salesforce/wikitext/tree/{prepared['dataset_revision']}/wikitext-2-raw-v1). "
        "Checksums for the model, tokenizer, source parquet files and exported inputs accompany the run.\n",
        "## Actual forward path\n",
        "The selected projection uses signed weights encoded as per-output `lower+span*p`. "
        "Each normalized probability has one logical THRML node. Native Gibbs states drive "
        "linear output accumulation; the averaged linear output is returned before all "
        "downstream residual, activation and attention operations. No averaged weight matrix "
        "is reconstructed. Eight context windows share each sampled weight-bank state.\n",
        "Torch and THRML run in separate version-recorded environments. The deterministic prefix supplies "
        "the projection input, THRML computes the replacement output, and the real Torch model "
        "continues from that point. Every replay checks that the current input exactly equals "
        "the exported input; FP32 replacement exactly recovers per-token losses/predictions. "
        "This cache protocol is valid because no earlier module is converted. Converting "
        "multiple modules will require regenerating later inputs after earlier stochastic replacements.\n",
        "## Evaluation and selection\n",
        "Separate official validation/test splits supply 16/32 nonoverlapping 65-token windows, "
        "respectively 1,024/2,048 next-token targets. These are short-context sampled-subset "
        "perplexities, not full-corpus benchmark numbers. Five sampling seeds condition on "
        "one fixed pretrained model and the fixed token windows.\n",
        "The predeclared calibration gate accepts at most 5% relative perplexity increase and "
        "at most one percentage-point token-accuracy loss. The smallest passing sample budget "
        f"was T={chosen}; calibration gate {'passed' if selection['gate_passed_on_calibration'] else 'FAILED (diagnostic fallback)'}. "
        "The choice was saved before test evaluation. All three predeclared budget curves "
        "are retained.\n",
        "| Projection representation | Test perplexity: exp(mean CE) | Relative to FP32 | Token accuracy |\n|---|---:|---:|---:|"]
    for name in ("FP32", "zero_matrix", "affine4", "affine8", "T16", "T64", "T256"):
        r = test[name]
        ratio = f"{float(r['relative_perplexity_from_mean_ce']):.4f}"
        if name.startswith("T"):
            ratio += f" [{float(r['relative_perplexity_ci95_lower']):.4f}, {float(r['relative_perplexity_ci95_upper']):.4f}]"
        lines.append(f"| {name} | {float(r['exp_mean_cross_entropy']):.3f} | {ratio} | {100*float(r['token_accuracy_mean']):.3f}% |")
    selected = test[f"T{chosen}"]
    lines += [f"\nAt selected T={chosen}, held-out subset perplexity increased by "
        f"{100*(float(selected['relative_perplexity_from_mean_ce'])-1):.2f}%. Intervals use "
        "1.96*SD/sqrt(5) for sampling-seed CE, then exponentiate the bounds. They quantify "
        "Monte Carlo variation, not training or corpus-population uncertainty. Perplexity "
        "is exp(mean CE), not an unlabelled average of per-seed perplexities.\n",
        "The zero-matrix control keeps the projection bias and removes its weight contribution. "
        f"Its perplexity rises to {float(test['zero_matrix']['relative_perplexity_from_mean_ce']):.2f} times FP32, "
        "showing the chosen component affects the task. Affine4/8 are deterministic numerical "
        "controls on the same selected weights, not optimized quantized kernels.\n",
        "## Resources and limits\n",
        f"The selected matrix uses {controls['logical_pbits']:,} logical nodes, "
        f"{controls['free_pbits']:,} free nodes and explicit endpoint clamps. Range metadata "
        f"is {controls['metadata_bits']:,} bits ({controls['metadata_bits']//8:,} bytes), plus "
        f"768 deterministic bias values. The simulator retains {controls['bias_storage_bytes_simulator']:,} "
        "bytes of float32 field controls as a subtotal, before probability arrays, Python "
        "objects, program copies, output accumulators and other buffers. There is no achieved "
        "software-memory compression or demonstrated physical density advantage.\n",
        "For every native condition, linear_metrics.csv counts actual free-node evaluations "
        "and bank readouts, and compares projection MSE with its analytical sampling variance/T. "
        "Weights/ranges are static and reused; physical bias retention/programming, switching, "
        "fanout, output accumulation, energy and latency remain unmeasured. Controls, metadata "
        "and external arithmetic are float32. Finite-controller and correlated-device effects "
        "have not been validated for this large operator.\n",
        "One useful selective projection does not prove whole-Transformer viability, a "
        "25–100% converted-model regime, language-model compression, or a system advantage "
        "over Extropic's own architecture. Those questions remain active.\n",
        "## Reproduce\n",
        "Run transformer_export in .venv, transformer_projection_native in .venv-thrml, "
        "transformer_replay in .venv, then verify_transformer and transformer_report in "
        ".venv-thrml, all as research.thrml_spw modules. Preserve the recorded dataset/model "
        "revisions and checksums. Raw token losses/predictions, native projection outputs, "
        "fields/metadata counts and code hashes accompany the run.\n"]
    (ROOT/"TRANSFORMER_PROJECTION_REPORT.md").write_text("\n".join(lines)+"\n")
    print("Wrote TRANSFORMER_PROJECTION_REPORT.md and transformer_projection_evidence.png/.pdf")


if __name__ == "__main__":
    main()
