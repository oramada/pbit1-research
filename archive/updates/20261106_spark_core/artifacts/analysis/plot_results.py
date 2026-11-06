from __future__ import annotations

"""Render standalone SVG plots from experiment CSV artifacts (stdlib only)."""

import csv
from html import escape
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
WIDTH, HEIGHT = 1080, 680
COLORS = ["#1769aa", "#e76f00", "#248232", "#b52e31", "#7b4ab5", "#008b8b",
          "#9a6b00", "#c04c9b", "#64748b", "#111827"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def svg_start(title: str, subtitle: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<style>text{font-family:Inter,Arial,sans-serif;fill:#202938} '
            '.grid{stroke:#e5e9ef;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.4} '
            '.tick{font-size:12px;fill:#4b5563}.title{font-size:24px;font-weight:700} '
            '.sub{font-size:13px;fill:#667085}.label{font-size:14px;font-weight:600} '
            '.legend{font-size:12px}</style>',
            f'<text x="70" y="43" class="title">{escape(title)}</text>',
            f'<text x="70" y="67" class="sub">{escape(subtitle)}</text>']


def render_line_chart(path: Path, title: str, subtitle: str, x_label: str, y_label: str,
                      series: list[dict], x_mode: str = "linear", y_range=None,
                      x_ticks=None, y_ticks=None, annotations=()) -> None:
    # Each series: {label, points: [(x,y)], color, dash?}
    left, right, top, bottom = 95, 810, 115, 555
    plot_w, plot_h = right - left, bottom - top
    xs = [x for item in series for x, y in item["points"] if x > 0 or x_mode != "log2"]
    ys = [y for item in series for x, y in item["points"] if math.isfinite(y)]
    if not xs or not ys:
        return
    if x_mode == "log2":
        tx = lambda value: math.log2(max(value, 1e-12))
        x_min, x_max = min(tx(v) for v in xs), max(tx(v) for v in xs)
    else:
        tx = lambda value: value
        x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min, x_max = x_min - .5, x_max + .5
    if y_range:
        y_min, y_max = y_range
    else:
        y_min, y_max = min(ys), max(ys)
        span = y_max - y_min
        padding = .08 * span if span else max(abs(y_max) * .08, .1)
        y_min, y_max = y_min - padding, y_max + padding
    if y_min == y_max:
        y_min -= .5
        y_max += .5
    px = lambda value: left + (tx(value) - x_min) / (x_max - x_min) * plot_w
    py = lambda value: bottom - (value - y_min) / (y_max - y_min) * plot_h

    if x_ticks is None:
        if x_mode == "log2":
            x_ticks = [2**i for i in range(math.ceil(x_min), math.floor(x_max) + 1)]
        else:
            x_ticks = [x_min + i * (x_max - x_min) / 5 for i in range(6)]
    if y_ticks is None:
        y_ticks = [y_min + i * (y_max - y_min) / 5 for i in range(6)]
    svg = svg_start(title, subtitle)
    for tick in y_ticks:
        y = py(tick)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.3g}</text>')
    for tick in x_ticks:
        if tick < min(xs) or tick > max(xs):
            continue
        x = px(tick)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>')
        label = str(int(tick)) if float(tick).is_integer() else f"{tick:.2g}"
        svg.append(f'<text x="{x:.1f}" y="{bottom+23}" text-anchor="middle" class="tick">{label}</text>')
    svg.extend([f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
                f'<text x="{(left+right)/2}" y="{bottom+53}" text-anchor="middle" class="label">{escape(x_label)}</text>',
                f'<text x="24" y="{(top+bottom)/2}" text-anchor="middle" class="label" '
                f'transform="rotate(-90 24 {(top+bottom)/2})">{escape(y_label)}</text>'])
    for item in series:
        points = [(x, y) for x, y in item["points"] if math.isfinite(y)]
        if not points:
            continue
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        dash = ' stroke-dasharray="6 5"' if item.get("dash") else ""
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{item["color"]}" '
                   f'stroke-width="2.7"{dash}/>')
        for x, y in points:
            svg.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{item["color"]}"/>')
    legend_y = top + 7
    legend_x = right + 35
    for idx, item in enumerate(series):
        y = legend_y + idx * 25
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+25}" y2="{y}" '
                   f'stroke="{item["color"]}" stroke-width="2.7"/>')
        svg.append(f'<text x="{legend_x+33}" y="{y+4}" class="legend">{escape(item["label"])}</text>')
    for text in annotations:
        svg.append(f'<text x="{left+8}" y="{top+20+18*len([x for x in annotations if x])}" '
                   f'class="sub">{escape(text)}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg))


def render_range_bars(rows: list[dict[str, str]]) -> None:
    rows = [r for r in rows if float(r["range_alpha"]) == 0.0]
    if not rows:
        return
    labels = [r["grouping"] for r in rows]
    radii = [float(r["mean_radius"]) for r in rows]
    metadata = [float(r["metadata_bits_per_weight"]) for r in rows]
    left, top, bottom = 120, 140, 530
    width = 840
    maxval = max(max(radii), max(metadata)) * 1.18
    svg = svg_start("Range localization: metadata versus stochastic range",
                    "Synthetic 256×256 Gaussian matrix; center and radius stored as float32 per group")
    slot = width / len(labels)
    for i, label in enumerate(labels):
        x = left + i * slot + 8
        h1 = radii[i] / maxval * (bottom - top)
        h2 = metadata[i] / maxval * (bottom - top)
        svg.extend([f'<rect x="{x:.1f}" y="{bottom-h1:.1f}" width="24" height="{h1:.1f}" fill="#1769aa"/>',
                    f'<rect x="{x+29:.1f}" y="{bottom-h2:.1f}" width="24" height="{h2:.1f}" fill="#e76f00"/>',
                    f'<text x="{x+26:.1f}" y="{bottom+23}" text-anchor="middle" class="tick">{escape(label)}</text>',
                    f'<text x="{x+12:.1f}" y="{bottom-h1-7:.1f}" text-anchor="middle" class="tick">{radii[i]:.3f}</text>',
                    f'<text x="{x+42:.1f}" y="{bottom-h2-7:.1f}" text-anchor="middle" class="tick">{metadata[i]:.3g}</text>'])
    svg.extend([f'<line x1="{left}" y1="{bottom}" x2="{left+width}" y2="{bottom}" class="axis"/>',
                f'<text x="{left+width/2}" y="{bottom+58}" text-anchor="middle" class="label">Range grouping</text>',
                '<rect x="730" y="100" width="14" height="14" fill="#1769aa"/><text x="752" y="112" class="legend">mean range radius</text>',
                '<rect x="885" y="100" width="14" height="14" fill="#e76f00"/><text x="907" y="112" class="legend">metadata bits/weight</text>',
                f'<text x="38" y="{(top+bottom)/2}" text-anchor="middle" class="label" transform="rotate(-90 38 {(top+bottom)/2})">Metric magnitude (separate quantities; compare labels)</text>',
                '</svg>'])
    (OUT / "range_metadata.svg").write_text("\n".join(svg))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scalar = read_csv(RESULTS / "scalar_metrics.csv")
    if scalar:
        t = [int(r["samples"]) for r in scalar]
        render_line_chart(OUT / "scalar_precision.svg", "Scalar reconstruction error",
                          "1,000 uniformly spaced targets × 1,000 Monte Carlo trials; dashed curve is the analytical RMSE",
                          "Independent Bernoulli samples per weight", "RMSE", [
            {"label": "empirical RMSE", "points": list(zip(t, [float(r["rmse"]) for r in scalar])), "color": COLORS[0]},
            {"label": "theory RMSE", "points": list(zip(t, [math.sqrt(float(r["theoretical_mse"])) for r in scalar])), "color": COLORS[1], "dash": True},
        ], x_mode="log2", y_range=(0, 0.9), x_ticks=t)
        render_line_chart(OUT / "effective_bits.svg", "Temporal sampling has logarithmic precision returns",
                          "Worst-case standard deviation σ=1/√T; rough effective bits = ½ log₂(T)",
                          "Samples T (log₂ scale)", "Effective bits", [
            {"label": "effective precision", "points": list(zip(t, [float(r["effective_bits"]) for r in scalar])), "color": COLORS[2]},
        ], x_mode="log2", y_range=(0, 6.5), x_ticks=t)

    ranges = read_csv(RESULTS / "local_range_metrics.csv")
    if ranges:
        render_range_bars(ranges)

    nonideal = read_csv(RESULTS / "nonideality_metrics.csv")
    if nonideal:
        rho = [float(r["rho"]) for r in nonideal]
        render_line_chart(OUT / "temporal_correlation.svg", "Temporal correlation reduces effective sampling",
                          "512 Markov samples per p-bit; empirical RMSE rises with autocorrelation",
                          "Lag-one correlation ρ", "Probability-estimation RMSE", [
            {"label": "empirical RMSE", "points": list(zip(rho, [float(r["empirical_rmse"]) for r in nonideal])), "color": COLORS[3]},
            {"label": "iid reference", "points": list(zip(rho, [float(r["independent_rmse"]) for r in nonideal])), "color": COLORS[0], "dash": True},
        ], y_range=(0, max(float(r["empirical_rmse"]) for r in nonideal) * 1.2), x_ticks=rho)

    dot = read_csv(RESULTS / "dot_product_metrics.csv")
    if dot:
        dimensions = sorted(set(int(r["dimension"]) for r in dot))
        series = []
        for idx, dimension in enumerate(dimensions):
            subset = sorted((r for r in dot if int(r["dimension"]) == dimension), key=lambda r: int(r["samples"]))
            ts = [int(r["samples"]) for r in subset]
            series.append({"label": f"d={dimension} empirical", "points": list(zip(ts, [math.sqrt(float(r["empirical_mse"])) for r in subset])), "color": COLORS[idx]})
        render_line_chart(OUT / "dot_product_error.svg", "Direct dot-product Monte Carlo error",
                          "Unit-normalized Gaussian input; 32 trials per dimension/sample point",
                          "Samples per stochastic weight realization", "Output RMSE", series,
                          x_mode="log2", y_range=(0, 1.15), x_ticks=[1, 4, 16, 64, 256])

    vision_rows = []
    for filename in ("pilot.csv", "high_t.csv", "metrics.csv", "cifar_agent.csv"):
        vision_rows.extend(read_csv(RESULTS / "vision" / filename))
    if vision_rows:
        for dataset in sorted(set(r["dataset"] for r in vision_rows)):
            subset = [r for r in vision_rows if r["dataset"] == dataset]
            f32 = next((float(r["accuracy"]) for r in subset if r["representation"] == "FP32"), None)
            series = []
            chosen = [r for r in subset if r["representation"].startswith("SPW_")]
            for idx, rep in enumerate(sorted(set(r["representation"] for r in chosen))):
                rep_rows = sorted((r for r in chosen if r["representation"] == rep and int(r["samples"]) > 0),
                                  key=lambda r: int(r["samples"]))
                # Combine pilot and extended sweeps without duplicate sample counts.
                unique = {int(r["samples"]): float(r["accuracy"]) for r in rep_rows}
                series.append({"label": rep.replace("SPW_", "SPW "),
                               "points": sorted(unique.items()), "color": COLORS[idx]})
            if f32 is not None:
                series.append({"label": "FP32 reference", "points": [(1, f32), (max(1, max(int(r["samples"]) for r in subset)), f32)],
                               "color": "#111827", "dash": True})
            for rep in ("INT4", "INT8"):
                value = next((float(r["accuracy"]) for r in subset if r["representation"] == rep), None)
                if value is not None:
                    xmax = max([int(r["samples"]) for r in subset] + [1])
                    series.append({"label": rep + " reference", "points": [(1, value), (xmax, value)],
                                   "color": COLORS[5 if rep == "INT8" else 6], "dash": True})
            render_line_chart(OUT / f"vision_{dataset}.svg", f"{dataset.upper()}: quality versus SPW sample count",
                              "Single-run pilot on a fixed test subset; deterministic references shown for comparison",
                              "Full-network stochastic weight samples", "Classification accuracy", series,
                              x_mode="log2", y_range=(0, 1.0),
                              x_ticks=sorted(set(int(r["samples"]) for r in subset if int(r["samples"]) > 0)))
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
