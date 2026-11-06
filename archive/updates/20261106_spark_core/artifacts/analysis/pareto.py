from __future__ import annotations

import argparse
import csv
from pathlib import Path


N_WEIGHTS = {"mnist": 234_752, "fashion": 421_408, "cifar10": 0}
SOURCE_EVAL_N = {"pilot.csv": 2000, "high_t.csv": 1000, "cifar_agent.csv": None}


def load_points(paths: list[str]) -> list[dict]:
    points = []
    for filename in paths:
        path = Path(filename)
        n_eval = SOURCE_EVAL_N.get(path.name)
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                data = row["dataset"]
                n_weights = N_WEIGHTS.get(data, 0)
                if n_weights == 0:
                    continue
                rep = row["representation"]
                samples = int(float(row["samples"]))
                if rep.startswith("SPW"):
                    pbits_per_weight = 1.0
                    transition_count = n_weights * samples
                    metadata_bytes = int(float(row["metadata_bytes"]))
                    storage_bits_per_parameter = 1 + metadata_bytes * 8 / n_weights
                elif rep.startswith("multi_pbit"):
                    pbits_per_weight = float(row["pbits_per_weight"])
                    transition_count = n_weights * pbits_per_weight * max(samples, 1)
                    metadata_bytes = int(float(row["metadata_bytes"]))
                    storage_bits_per_parameter = pbits_per_weight + metadata_bytes * 8 / n_weights
                elif rep.startswith("INT"):
                    pbits_per_weight = 0.0
                    transition_count = 0
                    metadata_bytes = 0
                    storage_bits_per_parameter = int(rep[3:])
                elif rep == "FP32":
                    pbits_per_weight = 0.0
                    transition_count = 0
                    metadata_bytes = 0
                    storage_bits_per_parameter = 32
                else:
                    continue
                points.append({
                    "source": path.name,
                    "eval_examples": n_eval if n_eval is not None else "unknown",
                    "dataset": data,
                    "seed": int(row["seed"]),
                    "representation": rep,
                    "samples_per_weight": samples,
                    "accuracy": float(row["accuracy"]),
                    "accuracy_delta": float(row["accuracy_delta"]),
                    "logit_rmse": float(row["logit_rmse"]),
                    "pbits_per_weight": pbits_per_weight,
                    "metadata_bytes": metadata_bytes,
                    "metadata_bits_per_weight": metadata_bytes * 8 / n_weights,
                    "effective_storage_bits_per_weight": storage_bits_per_parameter,
                    "weights_counted": n_weights,
                    "stochastic_transitions_per_inference": transition_count,
                    "transitions_per_parameter": transition_count / n_weights,
                })
    return points


def is_nondominated(point: dict, peers: list[dict], xkey: str) -> bool:
    x = float(point[xkey])
    y = float(point["accuracy"])
    for other in peers:
        ox = float(other[xkey])
        oy = float(other["accuracy"])
        if ox <= x and oy >= y and (ox < x or oy > y):
            return False
    return True


def svg_plot(points: list[dict], xkey: str, xlabel: str, out: Path) -> None:
    width, height = 920, 560
    left, right, top, bottom = 90, 35, 45, 85
    xvalues = [float(p[xkey]) for p in points]
    yvalues = [float(p["accuracy"]) * 100 for p in points]
    xmin, xmax = min(xvalues), max(xvalues)
    if xmin == xmax:
        xmax = xmin + 1
    ymin = max(0.0, min(yvalues) - 5)
    ymax = min(100.0, max(yvalues) + 3)
    if ymin == ymax:
        ymax = ymin + 1
    px = lambda x: left + (x - xmin) / (xmax - xmin) * (width - left - right)
    py = lambda y: height - bottom - (y - ymin) / (ymax - ymin) * (height - top - bottom)
    colors = {"FP32": "#222222", "INT8": "#4c78a8", "INT4": "#f58518",
              "multi_pbit_K4_quantized": "#e45756", "SPW_per_output_channel": "#54a24b"}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">Accuracy vs {xlabel}</text>',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
             f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">{xlabel}</text>',
             f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">Accuracy (%)</text>']
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        yy = py(y)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.1f}</text>')
    ordered = sorted(points, key=lambda p: float(p[xkey]))
    frontier = [p for p in ordered if is_nondominated(p, points, xkey)]
    if len(frontier) > 1:
        frontier.sort(key=lambda p: float(p[xkey]))
        coord = " ".join(f'{px(float(p[xkey])):.1f},{py(float(p["accuracy"])*100):.1f}' for p in frontier)
        lines.append(f'<polyline points="{coord}" fill="none" stroke="#54a24b" stroke-width="1.5" stroke-dasharray="5,4"/>')
    for p in points:
        x, y = px(float(p[xkey])), py(float(p["accuracy"])*100)
        color = colors.get(p["representation"], "#999")
        title = (f'{p["representation"]}; acc={100*float(p["accuracy"]):.2f}%; '
                 f'T={p["samples_per_weight"]}; transitions={p["stochastic_transitions_per_inference"]}')
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>{title}</title></circle>')
        label = p["representation"].replace("SPW_per_output_channel", "SPW")
        if p["representation"].startswith("SPW"):
            label += f' T={p["samples_per_weight"]}'
        lines.append(f'<text x="{x+7:.1f}" y="{y-6:.1f}" font-family="sans-serif" font-size="9">{label}</text>')
    lines.append('</svg>')
    out.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", default=["results/vision/pilot.csv", "results/vision/high_t.csv"])
    parser.add_argument("--out-dir", default="results/pareto")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = load_points(args.inputs)
    if not points:
        raise SystemExit("No supported vision experiment rows found")
    csv_path = out_dir / "points.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    for source in sorted(set(p["source"] for p in points)):
        for dataset in sorted(set(p["dataset"] for p in points if p["source"] == source)):
            subset = [p for p in points if p["source"] == source and p["dataset"] == dataset]
            stem = f'{Path(source).stem}_{dataset}'
            svg_plot(subset, "pbits_per_weight", "p-bits per parameter", out_dir / f"{stem}_vs_pbits.svg")
            svg_plot(subset, "transitions_per_parameter", "stochastic transitions per parameter", out_dir / f"{stem}_vs_transitions.svg")
            svg_plot(subset, "effective_storage_bits_per_weight", "storage bits per parameter", out_dir / f"{stem}_vs_storage.svg")
    reports = ["# Pareto analysis", "", "Input sweeps are intentionally not merged across evaluation subsets:"]
    reports += [f"- `{src}`: {SOURCE_EVAL_N.get(src, 'unknown')} evaluation examples" for src in sorted(set(p["source"] for p in points))]
    reports += ["", "## Key interpretation", "",
                "Each stochastic transition estimate counts one state update per p-bit per model inference. "
                "SPW metadata uses the measured per-output-channel float32 center/radius bytes. The K=4 row "
                "is the current deterministic 4-bit bipolar-plane quantizer, so its transition count is shown "
                "as four binary device reads per weight; it is not a measured physical p-bit transition rate. "
                "INT and FP baselines have no p-bit transitions in this accounting.", "",
                "Accuracy points are single-seed pilot results, not confidence intervals. Evaluation subsets differ "
                "between `pilot.csv` and `high_t.csv`, so frontiers are drawn separately. Training/inference "
                "latency and energy are not measured; this is a transition/storage Pareto view only.", ""]
    (out_dir / "README.md").write_text("\n".join(reports))
    print(f"wrote {csv_path} and SVG plots for {len(points)} points")


if __name__ == "__main__":
    main()
