"""Plot completed, independently accepted sampling conditions; no partial points."""
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = Path(__file__)
    assert (Path('results/thrml_spw/source_snapshots') / (sha(source) + '.py')).read_bytes() == source.read_bytes()
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'primary_sampling_figure_v2'
    out.mkdir(exist_ok=False)
    points, inputs = [], {}
    common = None
    for namespace, samples, verifier in [
        ('counted_mean_validation', 1024, 'verify_counted_mean_statistics.py'),
        ('counted_mean_t4096', 4096, 'verify_counted_mean_t4096_statistics.py'),
    ]:
        directory = root / namespace
        stats_path, audit_path = directory / 'statistics.json', directory / 'statistics_audit.json'
        protocol_path = root / (namespace + '_protocol.json')
        stats, audit, protocol = [json.loads(p.read_text()) for p in [stats_path, audit_path, protocol_path]]
        assert audit['status'] == 'all_five_mean_quality_statistics_and_workspace_resource_counts_verified'
        assert audit['statistics_sha256'] == sha(stats_path)
        assert audit['source_sha256'] == sha(source.with_name(verifier)) == protocol['sources_sha256'][verifier]
        assert stats['protocol_sha256'] == sha(protocol_path)
        for p in [stats_path, audit_path, protocol_path]:
            inputs[str(p)] = sha(p)
        for path, digest in stats['inputs_sha256'].items():
            assert sha(Path(path)) == digest
        assert protocol['samples'] == samples and len(stats['native_NLL']['values']) == 5
        if common is None:
            common = stats['controls_NLL']
        assert common == stats['controls_NLL']
        assert abs(math.expm1(protocol['BF16_quality_gate_NLL_margin']) * 100 - 2) < 1e-12
        ratio = stats['native_over_same_fit_BF16_PPL_ratio']
        interval = stats['native_over_same_fit_BF16_PPL_ratio_MC95_interval']
        assert not stats['conditional_same_fit_BF16_gate_passed']
        points.append({'samples': samples, 'MC_seeds': 5,
            'native_mean_NLL': stats['native_NLL']['mean'],
            'native_per_seed_NLL': stats['native_NLL']['values'],
            'native_subset_perplexity': stats['native_subset_perplexity'],
            'PPL_ratio': ratio, 'PPL_ratio_MC95_interval': interval,
            'gap_percent': 100 * (ratio - 1),
            'gap_percent_MC95_interval': [100 * (v - 1) for v in interval],
            'retained_evaluations_per_seed': stats['work_per_seed'][0]['retained_scheduled_evaluations'],
            'fixed_quality_gate_passed': False})
    x = np.array([p['samples'] for p in points])
    y = np.array([p['gap_percent'] for p in points])
    bounds = np.array([p['gap_percent_MC95_interval'] for p in points])
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.subplots_adjust(left=.13, right=.97, top=.80, bottom=.24)
    ax.errorbar(x, y, yerr=np.vstack([y - bounds[:, 0], bounds[:, 1] - y]),
                fmt='o', color='#2455a4', capsize=6, markersize=7, linewidth=1.8,
                label='Mean and approximate 95% MC interval')
    ax.axhline(0, color='#69717b', linewidth=1)
    ax.axhline(2, color='#7a5c00', linestyle='--', linewidth=1.4)
    ax.text(820, 2.45, 'Required upper bound: at most +2%', color='#7a5c00', fontsize=9)
    ax.annotate(f'Mean +{y[0]:.2f}%', (x[0], y[0]), xytext=(12, 6), textcoords='offset points')
    ax.annotate(f'T4096: mean +{y[1]:.2f}%\nUpper bound +{bounds[1, 1]:.2f}%: gate missed',
                xy=(x[1], bounds[1, 1]), xytext=(2200, 6.8), fontsize=9,
                arrowprops={'arrowstyle': '->', 'color': '#69717b', 'linewidth': 1})
    ax.set_xscale('log', base=2)
    ax.set_xticks([1024, 4096], labels=['1024', '4096'])
    ax.set_xlim(760, 5900)
    ax.set_ylim(-1.4, 18)
    ax.set_xlabel('Temporal samples per operator, T')
    ax.set_ylabel('Perplexity gap versus same-fit BF16 (%)')
    ax.grid(axis='y', alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    fig.suptitle('One p-bit per learned weight: completed LFM3 sampling tests', fontsize=12, y=.96)
    fig.text(.13, .86, 'All 1.729B unique parameters; identical adapted expected weights and BF16 reference', fontsize=9)
    fig.text(.13, .11, 'Five independent Monte Carlo seeds per condition; 2,048 fixed validation targets.', fontsize=9)
    fig.text(.13, .075, 'Intervals cover sampling uncertainty, not text or training uncertainty. T8192 is running and excluded.', fontsize=8)
    fig.text(.13, .04, 'Software simulation only: device residency, latency, energy and hardware advantage are not established.', fontsize=8)
    for extension in ['png', 'svg']:
        fig.savefig(out / ('sampling_quality.' + extension), dpi=200, facecolor='white')
    plt.close(fig)
    data = {'status': 'two_completed_audited_conditions_only', 'source_sha256': sha(source),
            'inputs_sha256': inputs, 'same_fit_BF16_NLL': common['same_fit_BF16'],
            'same_fit_BF16_PPL': math.exp(common['same_fit_BF16']), 'points': points,
            'fixed_upper_MC_quality_margin_percent': 2,
            'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
            'scope': 'Conditional fixed-subset model-quality/sampling curve, not a hardware Pareto frontier. No interpolation, pending-budget point, new neural evaluation or equivalence claim.'}
    with (out / 'plotted_data.json').open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
    with (out / 'artifacts.json').open('x') as handle:
        json.dump({p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'artifacts.json'}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
