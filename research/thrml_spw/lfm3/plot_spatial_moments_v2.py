"""Export all K=2/4/8 comparisons, including stronger spatial references."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b/spatial_moments')
    out = root / 'figure_v2'
    out.mkdir(exist_ok=True)
    r = json.loads((root / 'results.json').read_text())
    a = json.loads((root / 'independent_audit.json').read_text())
    assert a['status'] == 'all_K2_4_8_full_checkpoint_scalar_moments_verified_by_endpoint_and_complete_word_laws'
    assert a['results_sha256'] == sha(root / 'results.json')
    budgets = np.geomspace(8, 1048576, 600)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(left=.13, right=.97, bottom=.27, top=.84)
    colors = {1: '#1768ac', 2: '#95682c', 4: '#c75b12', 8: '#208045'}
    for nodes, method in [(1, 'single12'), (2, 'word2_endpoint'), (4, 'word4_endpoint'), (8, 'word8_endpoint')]:
        m = r['summary'][method]
        curve = m['mean_squared_bias'] + nodes * m['mean_one_realization_variance'] / budgets
        label = 'One p-bit (T = C)' if nodes == 1 else f'{nodes} p-bits (T = C/{nodes})'
        ax.loglog(budgets, curve, color=colors[nodes], linewidth=2, label=label)
        x = np.asarray([int(v) for v in m['scalar_MSE_at_matched_evaluations']])
        ax.scatter(x, list(m['scalar_MSE_at_matched_evaluations'].values()), color=colors[nodes], s=18, zorder=3)
    ax.axvline(1024, color='.5', linestyle='--', linewidth=.8)
    ax.text(1400, 4e-4, 'Current matched budget\nC = 1024', fontsize=9, va='top')
    crossing = r['formal_scalar_MSE_crossing_evaluations_per_weight_by_K']['8']
    single = r['summary']['single12']
    height = single['mean_squared_bias'] + single['mean_one_realization_variance'] / crossing
    ax.annotate(f'Scalar crossing vs 8 p-bits\nC ≈ {crossing:,.0f}', xy=(crossing, height),
                xytext=(15000, 6e-7), fontsize=9,
                arrowprops={'arrowstyle': '->', 'color': '.35'})
    ax.set(xlim=(8, 1048576), ylim=(1e-9, 1e-3),
           xlabel='Retained p-bit evaluations per weight (C)',
           ylabel='Mean scalar reconstruction MSE')
    ax.grid(True, which='major', alpha=.25)
    ax.grid(True, which='minor', alpha=.08)
    ax.legend(loc='lower left', frameon=False, fontsize=9)
    fig.suptitle('LFM3 scalar precision: single versus 2, 4 and 8 p-bits', y=.95, fontsize=14)
    fig.text(.13, .055, 'All 1,728,801,536 unique BF16 weights; block-128 ranges and finite fields ±4.\n'
             'One p-bit: 12-bit bias control. Spatial words: two bias levels per node.\n'
             'Analytical stationary laws. Warmup, neural quality and hardware costs excluded.', fontsize=9)
    paths = []
    for extension in ['png', 'svg']:
        path = out / ('all_spatial_matched_work.' + extension)
        assert not path.exists()
        fig.savefig(path, dpi=180, metadata={'Creator': 'Matplotlib; verified full-checkpoint scalar moments'})
        paths.append(path)
    plt.close(fig)
    path = out / 'plotted_budgets.csv'
    with path.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['retained_evaluations_per_weight', *r['summary']])
        for c in r['summary']['single12']['scalar_MSE_at_matched_evaluations']:
            writer.writerow([c, *[m['scalar_MSE_at_matched_evaluations'][c] for m in r['summary'].values()]])
    paths.append(path)
    manifest = {'source_sha256': sha(Path(__file__)), 'results_sha256': sha(root / 'results.json'),
                'audit_sha256': sha(root / 'independent_audit.json'), 'matplotlib_version': matplotlib.__version__,
                'artifacts_sha256': {str(p): sha(p) for p in paths},
                'scope': 'All scalar baselines included, including K8 where single loses at current budget. Nominal marginal laws only; no neural/hardware/equivalence inference.'}
    with (out / 'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
