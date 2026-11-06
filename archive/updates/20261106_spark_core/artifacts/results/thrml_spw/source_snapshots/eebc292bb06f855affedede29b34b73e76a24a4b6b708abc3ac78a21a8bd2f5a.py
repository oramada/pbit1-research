"""Complete sampling curves from the closed frozen and fine-tuned CNN cohorts."""
import csv
import datetime
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from results.thrml_spw.fashion_cnn_native.finetune_figures.common import BASE, WORD, FINE, OUT, LABELS, sha, write, guard, seal


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    paths = []
    for folder in [BASE, WORD, FINE]:
        seal(folder/'completion.json')
        assert json.loads((folder/'statistics_audit.json').read_text())['passed']
        paths += [folder/name for name in ['completion.json', 'statistics.json', 'statistical_arrays.npz', 'resource_ledger.json']]
    seal(FINE/'review_completion.json')
    paths += [FINE/'review_completion.json', FINE/'warmup_accounting.json']
    p = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_sha256': {str(path): sha(path) for path in Path(__file__).parent.glob('*.py')},
         'input_sha256': {str(path): sha(path) for path in paths}, 'methods': LABELS,
         'retained_updates_per_weight_bank': [64, 256, 1024], 'physical_pbits_per_weight': [1, 1, 1, 4],
         'models': 5, 'MC_seeds': 10, 'bootstrap_draws': 5000,
         'accuracy_y_limits_percent': [60, 88], 'CE_y_limits_nats': [.35, 1.5],
         'environment': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
         'scope': 'All12method/budgetpoints from closed native CNN cohorts. Absolute accuracy/CE model/seed bootstrap intervals on fixed consumedimages. Same retained-sample work K*T, with warmup separately counted; training and independent replay costs not on x-axis. Extra training confounds representation-only comparisons. Not a physical energy/latency/area/memory Pareto proof or equivalence test.'}
    guard(p)
    write(OUT/'protocol.json', p)


def run():
    p = json.loads((OUT/'protocol.json').read_text())
    guard(p)
    assert not (OUT/'data.json').exists()
    base, fine, word = [np.load(folder/'statistical_arrays.npz') for folder in [BASE, FINE, WORD]]
    ledgers = [json.loads((folder/'resource_ledger.json').read_text()) for folder in [BASE, FINE, WORD]]
    payload = [ledgers[0]['parameter_payload_bytes'], ledgers[1]['nominal_parameter_payload_bytes'],
               ledgers[1]['nominal_parameter_payload_bytes'], ledgers[2]['word_nominal_parameter_bytes']]
    bm = base['bootstrap_models']
    point = np.empty((4, 3, 2))
    intervals = np.empty((4, 3, 2, 2))
    draws = np.empty((4, 3, 5000, 2))
    rows = []
    for method, values, seeds in [(0, base['values'], base['bootstrap_seeds']),
                                  (1, fine['values'][0], fine['bootstrap_method_seeds'][0]),
                                  (2, fine['values'][1], fine['bootstrap_method_seeds'][1]),
                                  (3, word['values'], word['bootstrap_word_seeds'])]:
        assert values.shape == (3, 5, 10, 2) and seeds.shape == (5000, 5, 10)
        k = p['physical_pbits_per_weight'][method]
        for bi, work in enumerate(p['retained_updates_per_weight_bank']):
            point[method, bi] = values[bi].mean((0, 1))*[100, 1]
            draws[method, bi] = values[bi][bm[:, :, None], seeds].mean((1, 2))*[100, 1]
            intervals[method, bi] = np.quantile(draws[method, bi], [.025, .975], axis=0).T
            acc, ce = point[method, bi]
            rows.append({'method': LABELS[method], 'pbits_per_weight': k, 'samples_per_layer': work//k,
                         'retained_sample_updates_per_weight_bank': work, 'accuracy_percent': float(acc),
                         'accuracy_low_percent': float(intervals[method, bi, 0, 0]), 'accuracy_high_percent': float(intervals[method, bi, 0, 1]),
                         'mean_CE_nats': float(ce), 'CE_low_nats': float(intervals[method, bi, 1, 0]),
                         'CE_high_nats': float(intervals[method, bi, 1, 1]),
                         'weight_nodes': 2664*k, 'nominal_parameter_payload_bytes': payload[method],
                         'retained_sample_updates_per_full_run': 2664*1250*work,
                         'warmup_updates_per_full_run': 2664*1250*k,
                         'retained_plus_warmup_updates_per_full_run': 2664*1250*(work+k)})
    sb = json.loads((BASE/'statistics.json').read_text())
    references = {name: (np.array(sb['control_summary'][name])*[100, 1]).tolist() for name in ['FP32_F64', 'ACT8_INT4_minmax']}
    np.savez_compressed(OUT/'figure_arrays.npz', points=point, intervals=intervals, draws=draws)
    with (OUT/'sampling_curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': 'spw-cnn-finetuning'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1))
    colors, markers = ['#666666', '#0072B2', '#009E73', '#D55E00'], ['o', '^', 'D', 's']
    work = p['retained_updates_per_weight_bank']
    for metric, ax in enumerate(axes):
        for method in range(4):
            errors = np.vstack([point[method, :, metric]-intervals[method, :, metric, 0], intervals[method, :, metric, 1]-point[method, :, metric]])
            assert errors.min() >= 0
            ax.errorbar(work, point[method, :, metric], yerr=errors, marker=markers[method], color=colors[method],
                        linewidth=1.6, markersize=5, capsize=3, label=LABELS[method])
        ax.axhline(references['FP32_F64'][metric], color='#222222', linestyle='--', linewidth=1, label='FP32 mean')
        ax.axhline(references['ACT8_INT4_minmax'][metric], color='#777777', linestyle=':', linewidth=1.2, label='INT4 min–max mean')
        ax.set_xscale('log', base=2)
        ax.set_xticks(work, [str(x) for x in work])
        ax.set_xlim(48, 1536)
        ax.set_ylim(*(p['accuracy_y_limits_percent'] if metric == 0 else p['CE_y_limits_nats']))
        ax.set_ylabel('Accuracy (%)' if metric == 0 else 'Mean cross-entropy (nats)')
        ax.set_xlabel('Retained-sample node updates / weight / bank (K × T)')
        ax.grid(axis='y', alpha=.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title('(a) Task accuracy' if metric == 0 else '(b) Task loss')
    fig.suptitle('Native Fashion-MNIST CNN: probability-aware fine-tuning', fontsize=12, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .085), ncol=3, frameon=False, fontsize=8)
    fig.text(.5, .05, 'Five models × ten MC seeds; descriptive intervals on consumed images. Extra training is included in trained-versus-frozen comparisons.', ha='center', fontsize=8)
    fig.text(.5, .018, 'Warmup, training and replay work are separately charged. These curves do not measure physical energy or latency.', ha='center', fontsize=8)
    fig.subplots_adjust(left=.07, right=.985, top=.85, bottom=.31, wspace=.28)
    fig.savefig(OUT/'native_cnn_finetuning.svg', metadata={'Date': None})
    fig.savefig(OUT/'native_cnn_finetuning.png', dpi=200)
    plt.close(fig)
    guard(p)
    write(OUT/'data.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'rows': rows, 'reference_mean_metrics': references,
                           'figure_arrays_sha256': sha(OUT/'figure_arrays.npz'), 'scope': p['scope']})
    print('All12points and120000bootstrap values exported withSVG/PNG/CSV', flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
