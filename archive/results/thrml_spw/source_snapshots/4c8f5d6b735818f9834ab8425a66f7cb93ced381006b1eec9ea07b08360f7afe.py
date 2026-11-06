"""Standalone scientific figure from closed CNN cohorts only."""
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('results/thrml_spw/fashion_cnn_native')
BASE = ROOT/'cohort_evidence'
WORD = ROOT/'word_comparison_evidence'
OUT = ROOT/'matched_figure_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')


def guard(p):
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for path, h in mapping.items():
            assert sha(path) == h, path


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    inputs = []
    for folder in [BASE, WORD]:
        closed = json.loads((folder/'completion.json').read_text())
        for path, h in closed['artifact_sha256'].items():
            assert sha(path) == h, path
        assert json.loads((folder/'statistics_audit.json').read_text())['passed']
        inputs += [folder/name for name in ['completion.json', 'statistics.json', 'statistical_arrays.npz', 'resource_ledger.json']]
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': {str(path): sha(path) for path in Path(__file__).parent.glob('*.py')},
         'input_sha256': {str(path): sha(path) for path in inputs},
         'budgets_K_times_T': [64, 256, 1024], 'methods': ['One p-bit', 'Four p-bits'],
         'models': 5, 'MC_seeds': 10, 'bootstrap_draws': 5000, 'y_limits_percent': [60, 88],
         'references': ['FP32_F64', 'ACT8_INT4_minmax'],
         'environment': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
         'scope': 'All3budgets/bothfrozenCNNmethods, no fine-tuned results. Two resource slices: schedulednodeupdatesperweightper8imagebank and samplesperlayer; neither is physicalswitches/latency/energy or a full systemsParetofrontier. Absolute accuracy intervals resample modelunits/MCseeds on fixed consumedimages; references drawn as five-model mean lines.'}
    guard(p)
    write(OUT/'protocol.json', p)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'data.json').exists()
    base = np.load(BASE/'statistical_arrays.npz')
    word = np.load(WORD/'statistical_arrays.npz')
    sb = json.loads((BASE/'statistics.json').read_text())
    sw = json.loads((WORD/'statistics.json').read_text())
    ledger = json.loads((WORD/'resource_ledger.json').read_text())
    bm = base['bootstrap_models']
    point = np.empty((2, 3))
    intervals = np.empty((2, 3, 2))
    draws = np.empty((2, 3, 5000))
    rows = []
    for method, arrays, seed_name, k in [(0, base, 'bootstrap_seeds', 1), (1, word, 'bootstrap_word_seeds', 4)]:
        values = arrays['values']
        seeds = arrays[seed_name]
        assert values.shape == (3, 5, 10, 2) and seeds.shape == (5000, 5, 10)
        for bi, u in enumerate(p['budgets_K_times_T']):
            v = values[bi, :, :, 0]
            point[method, bi] = 100*v.mean()
            draws[method, bi] = 100*v[bm[:, :, None], seeds].mean((1, 2))
            intervals[method, bi] = np.quantile(draws[method, bi], [.025, .975])
            rows.append({'method': p['methods'][method], 'pbits_per_weight': k, 'samples_per_layer': u//k,
                         'scheduled_updates_per_weight_bank': u, 'accuracy_percent': float(point[method, bi]),
                         'descriptive_low_percent': float(intervals[method, bi, 0]),
                         'descriptive_high_percent': float(intervals[method, bi, 1]),
                         'weight_nodes': 2664*k, 'nominal_parameter_bytes': ledger['single_nominal_parameter_bytes' if method == 0 else 'word_nominal_parameter_bytes'],
                         'scheduled_updates_full10000image_run': 2664*1250*u})
    assert len(rows) == 6
    references = {name: 100*sb['control_summary'][name][0] for name in p['references']}
    with (OUT/'accuracy_resource_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'figure_arrays.npz', points=point, intervals=intervals, draws=draws)
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'svg.hashsalt': 'spw-frozen-cnn'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    colors, markers = ['#0072B2', '#D55E00'], ['o', 's']
    budgets = np.array(p['budgets_K_times_T'])
    for panel, ax in enumerate(axes):
        for method, k in [(0, 1), (1, 4)]:
            x = budgets if panel == 0 else budgets//k
            errors = np.vstack([point[method]-intervals[method, :, 0], intervals[method, :, 1]-point[method]])
            assert errors.min() >= 0
            ax.errorbar(x, point[method], yerr=errors, color=colors[method], marker=markers[method],
                        linewidth=1.6, markersize=5, capsize=3, label=p['methods'][method])
        ax.axhline(references['FP32_F64'], color='#333333', linestyle='--', linewidth=1, label='FP32 mean')
        ax.axhline(references['ACT8_INT4_minmax'], color='#777777', linestyle=':', linewidth=1.2, label='INT4 min–max mean')
        ax.set_xscale('log', base=2)
        ticks = [64, 256, 1024] if panel == 0 else [16, 64, 256, 1024]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlim((48, 1536) if panel == 0 else (12, 1536))
        ax.set_ylim(*p['y_limits_percent'])
        ax.grid(axis='y', alpha=.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Scheduled node updates / weight / bank (K × T)' if panel == 0 else 'Samples per layer (T)')
        ax.set_title('(a) Matched stochastic work' if panel == 0 else '(b) Temporal sampling')
    axes[0].set_ylabel('Accuracy (%)')
    fig.suptitle('Frozen Fashion-MNIST CNNs', fontsize=12, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .035))
    fig.text(.5, .005, 'Five models × ten MC seeds; descriptive model/seed intervals on fixed consumed images. Not hardware latency or energy.', ha='center', fontsize=8)
    fig.subplots_adjust(left=.075, right=.985, top=.83, bottom=.23, wspace=.12)
    fig.savefig(OUT/'frozen_cnn_matched_work.svg', metadata={'Date': None})
    fig.savefig(OUT/'frozen_cnn_matched_work.png', dpi=200)
    plt.close(fig)
    guard(p)
    write(OUT/'data.json', {'rows': rows, 'reference_mean_accuracy_percent': references,
                           'protocol_sha256': sha(OUT/'protocol.json'), 'scope': p['scope'],
                           'interpretation': 'Four-node baseline has higher low-budget accuracy; largest-budget difference interval includes zero in the separate paired contrast, not equivalence. One-node uses fewer dynamic nodes but larger nominal control payload; no systems superiority follows.'})
    print('Closed-cohort figure, CSV and all30000bootstrapaccuracyvalues generated', flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
