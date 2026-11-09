"""Build the product comparison from audited scores; no new model runs."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/thrml_spw/spark_depth_20261106/cross_model_summary_v2/results.json"
OUT = ROOT / "docs/assets"


def main():
    data = json.loads(SOURCE.read_text())
    models = {row["model"]: row for row in data["models"]}
    rows = []
    for name, label, task in [
        ("GPT2", "GPT-2", "Language modeling"),
        ("DistilBERT", "DistilBERT", "Sentiment classification"),
        ("LFM3-1.7B", "LFM3-1.7B", "Language modeling"),
    ]:
        row = models[name]
        n = row["unique_parameters"]
        size = (f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.0f}M") + " parameters"
        assert row["seed_count"] == 5
        assert row["logical_pbits_per_learned_parameter"] == 1
        if name == "DistilBERT":
            ref, sampled = 100 * row["BF16_accuracy"], 100 * row["mean_selected_accuracy"]
            change = sampled - ref
            score_labels = [f"{ref:.3f}%", f"{sampled:.4f}%"]
            change_label = f"{change:+.4f} pp".replace("-", "−")
            unit, direction, limit, ticks = "Accuracy", "higher is better", 100, [0, 25, 50, 75, 100]
            change_caption = "accuracy change"
        else:
            ref, sampled = row["BF16_PPL"], row["selected_PPL_at_mean_NLL"]
            change = 100 * (row["perplexity_ratio"] - 1)
            score_labels = [f"{ref:.5f}", f"{sampled:.5f}"]
            change_label = f"{change:+.2f}%"
            unit, direction, limit, ticks = "Perplexity", "lower is better", 30, [0, 10, 20, 30]
            change_caption = "perplexity change"
        rows.append(dict(model=label, size=size, task=task, bf16=ref, pbit1=sampled,
                         score_labels=score_labels, change=change, change_label=change_label,
                         change_caption=change_caption, unit=unit, direction=direction,
                         limit=limit, ticks=ticks))

    colors = dict(bg="#fcf8f5", ink="#321c24", muted="#75656a", edge="#e8ddd8",
                  bf16="#dcd3d5", pbit1="#7b243d", grid="#f0e8e4")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "pbit1-product-benchmarks",
                         "savefig.facecolor": colors["bg"]})
    fig = plt.figure(figsize=(12, 8.0), facecolor=colors["bg"])
    fig.text(.045, .945, "One p-bit per weight. Three tested models.", fontsize=23,
             fontweight="bold", color=colors["ink"])
    fig.text(.045, .901, "BF16 reference vs. PBit1 reconstruction · five independent weight draws per model",
             fontsize=11.5, color=colors["muted"])
    for x, c, label in [(.045, colors["bf16"], "BF16"), (.15, colors["pbit1"], "PBit1")]:
        fig.add_artist(Rectangle((x, .851), .016, .016, transform=fig.transFigure, color=c))
        fig.text(x + .024, .851, label, fontsize=10.5, color=colors["ink"], va="bottom")
    for row, bottom in zip(rows, [.615, .365, .115]):
        fig.add_artist(FancyBboxPatch((.03, bottom), .94, .215,
                       boxstyle="round,pad=0.008,rounding_size=0.012",
                       facecolor="white", edgecolor=colors["edge"], linewidth=.9,
                       transform=fig.transFigure, zorder=0))
        fig.text(.055, bottom + .151, row["model"], fontsize=16.5,
                 fontweight="bold", color=colors["ink"])
        fig.text(.055, bottom + .112, row["size"], fontsize=10.8, color=colors["muted"])
        fig.text(.055, bottom + .078, row["task"], fontsize=9, color=colors["muted"])
        ax = fig.add_axes([.275, bottom + .047, .43, .142], facecolor="white", zorder=1)
        vals = [row["bf16"], row["pbit1"]]
        ax.barh([.72, .22], vals, height=.30, color=[colors["bf16"], colors["pbit1"]], zorder=3)
        for y, value, label, textcolor in zip([.72, .22], vals, row["score_labels"],
                                             [colors["ink"], "white"]):
            ax.text(value - row["limit"] * .022, y, label, va="center", ha="right",
                    fontsize=11, color=textcolor, fontweight="bold", zorder=4)
        ax.set(xlim=(0, row["limit"]), ylim=(-.10, 1.04), yticks=[], xticks=row["ticks"])
        ax.grid(axis="x", color=colors["grid"], zorder=0)
        ax.tick_params(axis="x", length=0, labelsize=8.5, colors=colors["muted"], pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(f"{row['unit']} · {row['direction']}", fontsize=8.5,
                      color=colors["muted"], labelpad=3)
        fig.text(.755, bottom + .122, row["change_label"], fontsize=21,
                 color=colors["pbit1"], fontweight="bold")
        fig.text(.755, bottom + .087, row["change_caption"], fontsize=10.5, color=colors["muted"])
        fig.text(.755, bottom + .050, "vs. BF16 reference", fontsize=9.5, color=colors["muted"])
    fig.text(.045, .055, "32,768 samples/weight · 12-bit controls + shared ranges · fixed benchmark datasets",
             fontsize=10, color=colors["muted"])
    fig.text(.045, .025, "Bars show model scores, not memory or speed. Confidence intervals and protocols: docs/EVIDENCE.md",
             fontsize=9, color=colors["muted"])
    OUT.mkdir(exist_ok=True)
    for suffix in ["svg", "png"]:
        fig.savefig(OUT / f"product_benchmarks.{suffix}", dpi=180,
                    metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    # A narrower layout keeps score labels readable in the mobile README.
    fig = plt.figure(figsize=(6, 10.8), facecolor=colors["bg"])
    fig.text(.06, .931, "One p-bit per weight.\nThree tested models.", fontsize=20,
             fontweight="bold", color=colors["ink"], linespacing=1.25)
    fig.text(.06, .861, "BF16 reference vs. PBit1 reconstruction\nFive independent weight draws per model",
             fontsize=10, color=colors["muted"], linespacing=1.5)
    for x, c, label in [(.06, colors["bf16"], "BF16"), (.29, colors["pbit1"], "PBit1")]:
        fig.add_artist(Rectangle((x, .810), .028, .012, transform=fig.transFigure, color=c))
        fig.text(x + .04, .810, label, fontsize=10, color=colors["ink"], va="bottom")
    for row, bottom in zip(rows, [.555, .315, .075]):
        fig.add_artist(FancyBboxPatch((.04, bottom), .92, .215,
                       boxstyle="round,pad=0.008,rounding_size=0.012",
                       facecolor="white", edgecolor=colors["edge"], linewidth=.9,
                       transform=fig.transFigure, zorder=0))
        fig.text(.075, bottom + .181, row["model"], fontsize=15,
                 fontweight="bold", color=colors["ink"])
        fig.text(.075, bottom + .154, row["size"], fontsize=9.5, color=colors["muted"])
        fig.text(.925, bottom + .181, row["change_label"], fontsize=15,
                 color=colors["pbit1"], fontweight="bold", ha="right")
        fig.text(.925, bottom + .154, row["change_caption"], fontsize=8.5,
                 color=colors["muted"], ha="right")
        ax = fig.add_axes([.085, bottom + .035, .83, .105], facecolor="white", zorder=1)
        vals = [row["bf16"], row["pbit1"]]
        ax.barh([.72, .22], vals, height=.30, color=[colors["bf16"], colors["pbit1"]], zorder=3)
        for y, value, label, textcolor in zip([.72, .22], vals, row["score_labels"],
                                             [colors["ink"], "white"]):
            ax.text(value - row["limit"] * .022, y, label, va="center", ha="right",
                    fontsize=11, color=textcolor, fontweight="bold", zorder=4)
        ax.set(xlim=(0, row["limit"]), ylim=(-.10, 1.04), yticks=[], xticks=row["ticks"])
        ax.grid(axis="x", color=colors["grid"], zorder=0)
        ax.tick_params(axis="x", length=0, labelsize=8, colors=colors["muted"], pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(f"{row['unit']} · {row['direction']}", fontsize=8.5,
                      color=colors["muted"], labelpad=3)
    fig.text(.06, .029, "32,768 samples/weight · 12-bit controls + shared ranges\nModel scores, not memory or speed. See benchmark guide for intervals.",
             fontsize=7.6, color=colors["muted"], linespacing=1.6)
    for suffix in ["svg", "png"]:
        fig.savefig(OUT / f"product_benchmarks_mobile.{suffix}", dpi=180,
                    metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    (OUT / "overview_data.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "rows": rows,
        "scope": "Chart of existing five-draw scores; no new measurements. PPL is exp(mean seed NLL); accuracy is mean seed accuracy. Error intervals remain in the source and benchmark guide."
    }, indent=2) + "\n")
    print("Built product_benchmarks.svg/png from audited scores.")


if __name__ == "__main__":
    main()
