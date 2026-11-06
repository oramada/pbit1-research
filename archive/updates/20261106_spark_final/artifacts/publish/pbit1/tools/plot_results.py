"""Regenerate publication figures from the audited benchmark snapshot."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / 'benchmarks/lfm3.json').read_text())
OUT = ROOT / 'docs/assets'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
    'axes.labelcolor': '#334155', 'text.color': '#12213a', 'xtick.color': '#475569',
    'ytick.color': '#475569', 'figure.facecolor': '#f8fafc', 'axes.facecolor': '#f8fafc',
    'svg.fonttype': 'none', 'savefig.facecolor': '#f8fafc'})
TEAL, BLUE, ORANGE = '#089b88', '#5260ce', '#cf782d'


def save(fig, name):
    for suffix in ['svg', 'png']:
        fig.savefig(OUT / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def quality():
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    rows = D['cohorts']
    y = np.array([r['perplexity_ratio'] for r in rows]) * 100 - 100
    ci = np.array([r['ratio_MC95'] for r in rows]) * 100 - 100
    ax.axhspan(-3, 2, color=TEAL, alpha=.075)
    ax.axhline(0, color='#8b98ab', lw=1, label='Same-fit BF16')
    ax.axhline(2, color=TEAL, ls='--', lw=1.3, label='Original 2% upper-bound criterion')
    for i, row in enumerate(rows):
        color = TEAL if row['quality_gate_passed'] else ORANGE
        ax.errorbar(i, y[i], yerr=[[y[i]-ci[i, 0]], [ci[i, 1]-y[i]]],
            fmt='o', color=color, capsize=6, markersize=9, elinewidth=2.2)
        ax.annotate(f"PPL {row['native_perplexity']:.2f}", (i, y[i]),
            xytext=(12, 3 if i == 0 else -20), textcoords='offset points', weight='bold', color=color)
    ax.set(xlim=(-.35, 2.65), ylim=(-3, 18), xticks=range(3),
        xticklabels=['1,024 samples', '4,096 samples', '8,192 samples'],
        ylabel='Perplexity change relative to same-fit BF16 (%)', xlabel='One logical p-bit per parameter in every condition')
    ax.set_title('BF16-level task quality, with precision supplied over time', loc='left', fontsize=17, pad=18)
    ax.grid(axis='y', alpha=.17)
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    fig.suptitle('LFM3-1.7B · 1.73B unique parameters · 2,048 fixed targets · five MC seeds',
        fontsize=11, color='#64748b', x=.08, ha='left')
    fig.subplots_adjust(bottom=.24, top=.80, left=.10, right=.98)
    fig.text(.10, .025, 'Bars: conditional 95% t4 Monte Carlo intervals. T8192 upper bound: +1.54%, passing the original +2% criterion.\nFixed validation subset and adapted point; fresh-data generalization and numerical BF16 equivalence are not established.',
        fontsize=9, color='#64748b')
    save(fig, 'quality')


def resources():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    n = D['unique_parameters'] / 1e9
    labels = ['PBit1', '4-p-bit\nreference', '8-p-bit\nreference']
    values = [n, 4*n, 8*n]
    bars = ax.bar(labels, values, color=[TEAL, BLUE, '#aab0db'], width=.55)
    ax.bar_label(bars, labels=[f'{v:.2f}B' for v in values], padding=5, weight='bold')
    ax.set(ylim=(0, 16), ylabel='Logical p-bits (billions)')
    ax.set_title('4× fewer logical elements vs K4', loc='left', pad=15)
    ax.grid(axis='y', alpha=.12)
    model = D['resource_model']
    payload = next(x for x in model['single_payload_sweep'] if x['control_bits_per_parameter'] == 12)
    base = model['BF16_reference_weight_bytes']/1e9
    state = payload['packed_binary_state_bytes']/1e9
    control = payload['packed_commands_bytes']/1e9
    meta = payload['range_metadata_bytes']/1e9
    bx.bar(0, base, color=BLUE, width=.5, label='BF16 weight payload')
    bx.bar(1, state, color=TEAL, width=.5, label='Binary state')
    bx.bar(1, control, bottom=state, color='#62c9b6', width=.5, label='12-bit controller')
    bx.bar(1, meta, bottom=state+control, color='#cbd5e1', width=.5, label='FP32 range metadata')
    bx.text(0, base+.07, f'{base:.2f} GB', ha='center', weight='bold')
    bx.text(1, state+control+meta+.07, f'{state+control+meta:.2f} GB', ha='center', weight='bold')
    bx.set(ylim=(0, 4.2), xticks=[0, 1], xticklabels=['BF16', 'PBit1 + controls'],
        ylabel='Modeled packed payload (decimal GB)')
    bx.set_title('Count the programming precision too', loc='left', pad=15)
    bx.legend(frameon=False, fontsize=9, loc='upper right')
    fig.text(.025, -.07, 'Left: fully resident logical assignments, before controller/peripheral device area. Internal positional references, not an Extropic hardware claim.\nRight: ideal packed payload accounting, not measured RAM or physical hardware memory. Simulator counters/reference tensors add further cost.\nEnergy and latency savings have not been measured. A smaller state count alone is not a systems speedup.', fontsize=9, color='#64748b')
    save(fig, 'resources')


def sampling():
    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    rows = D['cohorts']
    values = [r['retained_scheduled_evaluations_per_seed']/1e12 for r in rows]
    bars = ax.bar([f"T = {r['samples']:,}" for r in rows], values,
        color=[BLUE, '#8590df', TEAL], width=.48)
    ax.bar_label(bars, labels=[f'{v:.2f}T' for v in values], padding=5, weight='bold')
    ax.set(ylim=(0, 67), ylabel='Retained node evaluations (trillions)', xlabel='Per complete 2,048-target validation pass, per Monte Carlo seed')
    ax.set_title('The density advantage has a temporal price', loc='left', fontsize=17, pad=15)
    ax.grid(axis='y', alpha=.13)
    fig.subplots_adjust(bottom=.29, top=.83, left=.10, right=.98)
    fig.text(.10, .02, 'Audited, padding-inclusive scheduled work. Warmup is charged separately. This measures neither physical flips nor joules.\nMatched retained work: K1 × T1 = K4 × T4 = K8 × T8; equal work alone does not imply equal quality or latency.', fontsize=9, color='#64748b')
    save(fig, 'sampling')


if __name__ == '__main__':
    quality(); resources(); sampling()
