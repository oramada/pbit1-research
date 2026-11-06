"""Standalone BF16 task-accuracy sampling curves from closed cohorts only."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('results/thrml_spw')
PARENT = ROOT/'bf16_comparison/evidence'
OUT = ROOT/'bf16_comparison/figure_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(p):
    for group in ['source_sha256', 'input_sha256']:
        for path, digest in p[group].items():
            assert sha(path) == digest, path


def write(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, indent=2)+'\n')


def prepare():
    assert not OUT.exists()
    closed = json.loads((PARENT/'completion.json').read_text())
    assert closed['passed']
    for path, digest in closed['artifact_sha256'].items():
        assert sha(path) == digest, path
    OUT.mkdir()
    sources = [Path(__file__), Path(__file__).with_name('verify_figure.py')]
    for path in sources:
        assert (ROOT/'source_snapshots'/(sha(path)+'.py')).exists()
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': {str(path): sha(path) for path in sources},
        'input_sha256': {**closed['artifact_sha256'], str(PARENT/'completion.json'): sha(PARENT/'completion.json')},
        'displayed_control': 'BF16_Torch_CPU', 'displayed_contrasts': 12,
        'CSV_controls': ['BF16_weights_F64', 'BF16_Torch_CPU'], 'CSV_contrasts': 24,
        'samples': [64, 256, 1024],
        'scope': 'Derived paired accuracy differences and descriptive95%intervals from already independently audited retrospective BF16 comparisons. All methods/budgets shown for CPU BF16 execution; bothBF16paths retained in fullCSV. No new native draws, bootstrap replicas, formal equivalence or physical systems claim. T4096 experiment pending and excluded.'}
    guard(p)
    write(OUT/'protocol.json', p)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    s = json.loads((PARENT/'statistics.json').read_text())
    assert len(s['contrasts']) == 24
    rows = []
    for c in s['contrasts']:
        rows.append({'dataset': c['dataset'], 'method': c['method'], 'method_index': c['method_index'],
            'samples': c['T'], 'BF16_control': c['control'], 'pbits_per_weight': 1,
            'pbit_accuracy_percent': 100*c['native_mean'][0], 'BF16_accuracy_percent': 100*c['BF16_mean'][0],
            'accuracy_difference_pp': 100*c['difference'][0],
            'descriptive_95_low_pp': 100*c['descriptive_95_interval'][0][0],
            'descriptive_95_high_pp': 100*c['descriptive_95_interval'][0][1],
            'CE_difference_nats': c['difference'][1]})
    with (OUT/'BF16_sampling_comparisons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'svg.hashsalt': 'spw-bf16-task-differences'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1))
    displayed = []
    for ax, dataset, title in zip(axes, ['fashion', 'cifar'], ['Fashion-MNIST', 'Compact RGB CIFAR-10']):
        local = [r for r in rows if r['dataset'] == dataset and r['BF16_control'] == p['displayed_control']]
        for method, color, marker, label in [(0, '#0072B2', 'o', 'Expected-weight training'),
                                            (1, '#D55E00', 's', 'Sample-aware training')]:
            points = sorted([r for r in local if r['method_index'] == method], key=lambda r: r['samples'])
            assert [r['samples'] for r in points] == p['samples']
            x = [r['samples'] for r in points]
            y = np.array([r['accuracy_difference_pp'] for r in points])
            lo = np.array([r['descriptive_95_low_pp'] for r in points])
            hi = np.array([r['descriptive_95_high_pp'] for r in points])
            error = np.vstack([y-lo, hi-y]);assert error.min() >= 0
            ax.errorbar(x, y, yerr=error, color=color, marker=marker, label=label, capsize=4, linewidth=1.7)
            displayed.extend(points)
        ax.axhline(0, color='#444444', linestyle='--', linewidth=1)
        ax.set_xscale('log', base=2)
        ax.set_xticks(p['samples'], [str(t) for t in p['samples']])
        ax.set_xlim(48, 1400)
        ax.set_ylim(min(r['descriptive_95_low_pp'] for r in local)-1, 1.5)
        ax.grid(axis='y', alpha=.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(f'{title}\nCPU BF16 reference: {local[0]["BF16_accuracy_percent"]:.3f}%')
        ax.set_xlabel('Temporal samples per layer (T)')
        ax.set_ylabel('P-bit minus BF16 accuracy (percentage points)')
    fig.suptitle('One p-bit per weight: task-accuracy gap to BF16', fontsize=13, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .12), ncol=2, frameon=False)
    fig.text(.5, .015, 'Five paired models × ten MC seeds; fixed consumed images. Descriptive 95% intervals do not establish equivalence.\nAdditional training, probability controls, metadata and accumulation remain costs. Numerical BF16 precision and physical advantage are unproven.',
             ha='center', va='bottom', fontsize=8)
    fig.subplots_adjust(left=.09, right=.98, top=.78, bottom=.31, wspace=.28)
    fig.savefig(OUT/'one_pbit_BF16_sampling.svg', metadata={'Date': None})
    fig.savefig(OUT/'one_pbit_BF16_sampling.png', dpi=200)
    plt.close(fig)
    guard(p)
    write(OUT/'figure_data.json', {'rows': rows, 'displayed_rows': displayed, 'protocol_sha256': sha(OUT/'protocol.json'), 'scope': p['scope']})
    print('Two-panel sampling figure and all24BF16contrast CSV exported')


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
