"""Export verified analytical scalar curves; no neural-quality inference."""
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
    root = Path('results/thrml_spw/lfm3_1p7b/word_moments')
    out = root / 'figure_v2'
    out.mkdir(exist_ok=True)
    r = json.loads((root / 'results.json').read_text())
    audit = json.loads((root / 'independent_audit.json').read_text())
    assert audit['status'] == 'all_full_checkpoint_scalar_moments_verified_by_endpoint_and_16word_laws'
    assert audit['results_sha256'] == sha(root / 'results.json')
    single, word = [r['summary'][m] for m in ['single12', 'word4_endpoint']]
    budgets = np.geomspace(4, 65536, 500)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(left=.13, right=.97, bottom=.27, top=.84)
    for label, color, moments, nodes in [
            ('One p-bit (T = C)', '#1768ac', single, 1),
            ('Four p-bits (T = C/4)', '#c75b12', word, 4)]:
        curve = moments['mean_squared_bias'] + nodes * moments['mean_one_realization_variance'] / budgets
        ax.loglog(budgets, curve, color=color, linewidth=2, label=label)
        x = np.asarray([int(c) for c in moments['scalar_MSE_at_matched_evaluations']])
        y = np.asarray(list(moments['scalar_MSE_at_matched_evaluations'].values()))
        ax.scatter(x, y, color=color, s=24, zorder=3)
    ax.axhline(word['mean_squared_bias'], color='#c75b12', linestyle=':', linewidth=1.2,
               label='Four-p-bit squared-bias floor')
    crossing = r['formal_scalar_MSE_crossing_evaluations_per_weight']
    height = single['mean_squared_bias'] + single['mean_one_realization_variance'] / crossing
    ax.axvline(crossing, color='.45', linestyle='--', linewidth=.9)
    ax.annotate(f'Scalar MSE crossing\nC ≈ {crossing:.0f}', xy=(crossing, height),
                xytext=(crossing * 1.55, height * 4), fontsize=10,
                arrowprops={'arrowstyle': '->', 'color': '.35'})
    ax.set(xlim=(4, 65536), ylim=(1e-8, 1e-3),
           xlabel='Retained p-bit evaluations per weight (C)',
           ylabel='Mean scalar reconstruction MSE')
    ax.grid(True, which='major', alpha=.25)
    ax.grid(True, which='minor', alpha=.08)
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    fig.suptitle('Scalar precision tradeoff on LFM3-1.7B weights', y=.95, fontsize=14)
    fig.text(.13, .055, 'All 1,728,801,536 unique BF16 weights; block-128 ranges and finite fields ±4.\n'
             'One p-bit: 12-bit bias control. Four p-bits: two bias levels per node.\n'
             'Analytical stationary laws. Warmup, neural quality and hardware costs excluded.', fontsize=9)
    for extension in ['png', 'svg']:
        path = out / ('scalar_matched_work.' + extension)
        assert not path.exists()
        fig.savefig(path, dpi=180, metadata={'Creator': 'Matplotlib; verified analytical LFM3 scalar moments'})
    plt.close(fig)
    path = out / 'plotted_budgets.csv'
    with path.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['retained_evaluations_per_weight', 'single_samples', 'word_samples', 'single_MSE', 'word_MSE'])
        for c, mse in single['scalar_MSE_at_matched_evaluations'].items():
            writer.writerow([c, c, int(c) // 4, mse, word['scalar_MSE_at_matched_evaluations'][c]])
    artifacts = [out / 'scalar_matched_work.png', out / 'scalar_matched_work.svg', path]
    manifest = {'source_sha256': sha(Path(__file__)), 'results_sha256': sha(root / 'results.json'),
                'audit_sha256': sha(root / 'independent_audit.json'),
                'matplotlib_version': matplotlib.__version__,
                'artifacts_sha256': {str(p): sha(p) for p in artifacts},
                'scope': 'Visualization of verified nominal marginal scalar MSE laws only. No empirical error bars, neural quality, BF16 equivalence or hardware improvement claim.'}
    with (out / 'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
