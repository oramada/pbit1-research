from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


N_WEIGHTS = {"mnist": 234_752, "fashion": 421_408}


def summarize(paths: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for filename in paths:
        with Path(filename).open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["dataset"], row["representation"], int(float(row["samples"])))
                grouped[key].append(row)
    out = []
    for (dataset, representation, samples), rows in sorted(grouped.items()):
        nweights = N_WEIGHTS.get(dataset)
        if not nweights:
            continue
        accuracies = [float(row["accuracy"]) for row in rows]
        deltas = [float(row["accuracy_delta"]) for row in rows]
        logit_rmses = [float(row["logit_rmse"]) for row in rows]
        accuracy_mean = statistics.mean(accuracies)
        accuracy_sd = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        delta_mean = statistics.mean(deltas)
        delta_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        accuracy_ci95 = 1.96 * accuracy_sd / len(accuracies) ** 0.5
        delta_ci95 = 1.96 * delta_sd / len(deltas) ** 0.5
        rep = representation
        if rep.startswith("SPW"):
            pbits = 1.0
            transitions_per_parameter = float(samples)
            metadata_bytes = int(float(rows[0]["metadata_bytes"]))
            metadata_bits_per_parameter = metadata_bytes * 8 / nweights
        elif rep.startswith("multi_pbit"):
            pbits = float(rows[0]["pbits_per_weight"])
            transitions_per_parameter = pbits * max(samples, 1)
            metadata_bytes = int(float(rows[0]["metadata_bytes"]))
            metadata_bits_per_parameter = metadata_bytes * 8 / nweights
        else:
            pbits = 0.0
            transitions_per_parameter = 0.0
            metadata_bytes = 0
            metadata_bits_per_parameter = 0.0
        if rep.startswith("INT"):
            storage_bits = int(rep[3:])
        elif rep in ("FP16", "BF16"):
            storage_bits = 16
        elif rep == "FP32":
            storage_bits = 32
        else:
            storage_bits = pbits + metadata_bits_per_parameter
        out.append({
            "dataset": dataset,
            "representation": representation,
            "samples": samples,
            "seed_count": len(rows),
            "accuracy_mean": accuracy_mean,
            "accuracy_sd": accuracy_sd,
            "accuracy_ci95_margin": accuracy_ci95,
            "accuracy_ci95_low": accuracy_mean - accuracy_ci95,
            "accuracy_ci95_high": accuracy_mean + accuracy_ci95,
            "accuracy_delta_mean": delta_mean,
            "accuracy_delta_sd": delta_sd,
            "accuracy_delta_ci95_margin": delta_ci95,
            "accuracy_delta_ci95_low": delta_mean - delta_ci95,
            "accuracy_delta_ci95_high": delta_mean + delta_ci95,
            "logit_rmse_mean": statistics.mean(logit_rmses),
            "logit_rmse_sd": statistics.stdev(logit_rmses) if len(logit_rmses) > 1 else 0.0,
            "pbits_per_parameter": pbits,
            "metadata_bytes": metadata_bytes,
            "metadata_bits_per_parameter": metadata_bits_per_parameter,
            "storage_bits_per_parameter_estimate": storage_bits,
            "weight_count": nweights,
            "estimated_transitions_per_parameter": transitions_per_parameter,
            "estimated_transitions_per_inference": transitions_per_parameter * nweights,
            "transition_amortization_note": (
                "T reconstruction transitions/weight; amortize by actual model reuse count"
                if rep == "SPW_reconstruct_then_infer" else
                "T p-bit transitions/weight/inference" if rep.startswith("SPW") else
                "deterministic baseline; no p-bit transitions"
            ),
        })
    return out


def plot(points: list[dict], dataset: str, output: Path, x_key: str,
         x_label: str, x_max: float) -> None:
    w, h = 980, 590
    left, right, top, bottom = 90, 35, 55, 95
    y0, y1 = 0.0, 100.0
    x0 = 0.0
    x1 = max(x_max, 1.0)
    xx = lambda v: left + (float(v) - x0) / (x1 - x0) * (w - left - right)
    yy = lambda v: h - bottom - (float(v) - y0) / (y1 - y0) * (h - top - bottom)
    colors = {
        "FP32": "#202020", "FP16": "#777777", "BF16": "#999999",
        "INT8": "#4c78a8", "INT4": "#f58518", "INT2": "#b279a2",
        "multi_pbit_K4_quantized": "#e45756",
        "SPW_per_output_channel": "#72b7b2",
        "SPW_reconstruct_then_infer": "#54a24b",
    }
    markers = []
    for representation in sorted({p["representation"] for p in points}):
        curve = sorted([p for p in points if p["representation"] == representation],
                       key=lambda p: p[x_key])
        if not curve:
            continue
        color = colors.get(representation, "#888")
        coords = " ".join(f'{xx(p[x_key]):.1f},{yy(100*p["accuracy_mean"]):.1f}' for p in curve)
        if len(curve) > 1:
            markers.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for point in curve:
            x = xx(point[x_key])
            mean_pct = 100 * point["accuracy_mean"]
            sd_pct = 100 * point["accuracy_sd"]
            y = yy(mean_pct)
            upper, lower = yy(min(100, mean_pct + sd_pct)), yy(max(0, mean_pct - sd_pct))
            tip = (f'{representation}; T={point["samples"]}; '
                   f'acc={mean_pct:.2f}% +/- {sd_pct:.2f}% SD; '
                   f'transitions/parameter={point["estimated_transitions_per_parameter"]:g}')
            markers.append(f'<line x1="{x:.1f}" y1="{upper:.1f}" x2="{x:.1f}" y2="{lower:.1f}" stroke="{color}" stroke-width="1.5"/>')
            markers.append(f'<line x1="{x-4:.1f}" y1="{upper:.1f}" x2="{x+4:.1f}" y2="{upper:.1f}" stroke="{color}"/>')
            markers.append(f'<line x1="{x-4:.1f}" y1="{lower:.1f}" x2="{x+4:.1f}" y2="{lower:.1f}" stroke="{color}"/>')
            markers.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>{tip}</title></circle>')
            label = representation.replace("SPW_per_output_channel", "R2")
            label = label.replace("SPW_reconstruct_then_infer", "R1")
            if representation.startswith("SPW"):
                label += f' T={point["samples"]}'
            markers.append(f'<text x="{x+7:.1f}" y="{y-6:.1f}" font-family="sans-serif" font-size="9">{label}</text>')
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{w/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">{dataset.title()}: accuracy vs {x_label} (mean +/- 1 SD, 5 seeds)</text>',
             f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#333"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#333"/>',
             f'<text x="{w/2}" y="{h-20}" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>',
             f'<text transform="translate(22 {h/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">Accuracy (%)</text>']
    for i in range(6):
        value = 100 * i / 5
        y = yy(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.0f}</text>')
    if x_key == "estimated_transitions_per_parameter":
        tick_values = [0, 1, 4, 16, 64, 128, 256]
    elif x_key == "pbits_per_parameter":
        tick_values = [0, 1, 2, 4, 8]
    else:
        tick_values = [0, 1, 2, 4, 8, 16, 32]
    for value in tick_values:
        if value <= x1:
            x = xx(value)
            lines.append(f'<line x1="{x:.1f}" y1="{h-bottom}" x2="{x:.1f}" y2="{h-bottom+5}" stroke="#333"/>')
            lines.append(f'<text x="{x:.1f}" y="{h-bottom+19}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:g}</text>')
    lines.extend(markers)
    # Compact legend, outside data region.
    legend_items = ["FP32", "FP16", "BF16", "INT8", "INT4", "INT2", "multi_pbit_K4_quantized",
                    "SPW_per_output_channel", "SPW_reconstruct_then_infer"]
    lx, ly = left, h - 48
    for i, rep in enumerate(legend_items):
        if rep not in {p["representation"] for p in points}:
            continue
        x = lx + i * 145
        color = colors[rep]
        lines.append(f'<circle cx="{x}" cy="{ly}" r="4" fill="{color}"/>')
        label = {"multi_pbit_K4_quantized": "multi-p-bit K=4",
                 "SPW_per_output_channel": "SPW R2 direct",
                 "SPW_reconstruct_then_infer": "SPW R1 reconstruct"}.get(rep, rep)
        lines.append(f'<text x="{x+8}" y="{ly+4}" font-family="sans-serif" font-size="10">{label}</text>')
    lines.append("</svg>")
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=[
        "results/vision/mnist_r1_r2_seeds.csv",
        "results/vision/fashion_r1_r2_seeds.csv",
    ])
    parser.add_argument("--out-dir", default="results/pareto")
    args = parser.parse_args()
    summary = summarize(args.inputs)
    if not summary:
        raise SystemExit("No R1/R2 multi-seed vision results were found")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "r1_summary.csv"
    with summary_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for dataset in sorted({point["dataset"] for point in summary}):
        points = [point for point in summary if point["dataset"] == dataset]
        plot(points, dataset, out_dir / f"r1_{dataset}_vs_transitions.svg",
             "estimated_transitions_per_parameter", "stochastic transitions per parameter", 256)
        plot(points, dataset, out_dir / f"r1_{dataset}_vs_pbits.svg",
             "pbits_per_parameter", "p-bits per parameter", 4)
        plot(points, dataset, out_dir / f"r1_{dataset}_vs_storage.svg",
             "storage_bits_per_parameter_estimate", "estimated storage bits per parameter", 32)
    report = [
        "# R1 reconstruction Pareto summary", "",
        "These plots use the dedicated five-seed R1/R2 CSVs and do not modify the original direct-R2 pilot SVGs.",
        "The reported standard deviation is the sample SD across seeds. CSV columns include 95% confidence interval bounds computed as mean +/- 1.96*s/sqrt(5), for both accuracy and accuracy delta.",
        "The reconstruction transition estimate is T p-bit transitions per parameter for one full reconstruction.",
        "For repeated inference, divide this cost by the number of inferences that reuse the reconstructed weights; that reuse count was not measured here.",
        "Direct R2 is charged T transitions per parameter per inference. Metadata is measured from stored per-output-channel float32 centers and radii.",
        "The current K=4 reference is a deterministic quantized bipolar-plane encoding; its transition count is a hardware proxy, not a measured p-bit count.",
        "All accuracy summaries share the 2,000-example evaluation subset used in the multi-seed R1/R2 run.", "",
    ]
    (out_dir / "r1_README.md").write_text("\n".join(report))
    print(f"wrote {summary_file} and separate r1_* SVGs")


if __name__ == "__main__":
    main()
