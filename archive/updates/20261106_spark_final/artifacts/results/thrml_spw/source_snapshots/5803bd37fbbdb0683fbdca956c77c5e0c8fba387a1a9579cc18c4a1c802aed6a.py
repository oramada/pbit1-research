"""Frozen RGB sampling-quality figure; all conventional controls retained in CSV."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'frozen_figure_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from results.thrml_spw.fashion_cnn_native.cohort.common import CONTROLS
    assert not OUT.exists()
    for folder in ['cohort_evidence', 'precision_evidence', 'closure_review']:
        completion = json.loads((ROOT/folder/'completion.json').read_text())
        for path, expected in completion['artifact_sha256'].items():
            assert sha(path) == expected, path
    OUT.mkdir()
    sources = {str(path): sha(path) for path in Path(__file__).parent.glob('*.py')}
    inputs = [ROOT/'cohort_evidence/statistical_arrays.npz', ROOT/'precision_evidence/statistical_arrays.npz',
              ROOT/'cohort_evidence/completion.json', ROOT/'precision_evidence/completion.json', ROOT/'closure_review/completion.json']
    protocol = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_sha256': sources, 'input_sha256': {str(path): sha(path) for path in inputs},
                'scope': 'Derivative plot only, no new cases/native calls. All3native budgets and14conventional controls retained in data/CSV. Figure shows preselectedFP32/ACT8minmaxINT4/INT8reference levels. Fixed consumed images; absolute descriptive model/MCseed intervals are not paired tests or fresh generalization. Sampling counts are not energy/latency.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    base = np.load(inputs[0])
    precision = np.load(inputs[1])
    bm, bs = base['bootstrap_models'], base['bootstrap_seeds']
    rows, bootstraps = [], {}
    for bi, t in enumerate([64, 256, 1024]):
        v = base['values'][bi]
        dist = v[bm[:, :, None], bs].mean((1, 2))
        rows.append({'condition': 'Single_pbit_T'+str(t), 'kind': 'native', 'T': t,
                     'mean': v.mean((0, 1)).tolist(), 'interval': np.quantile(dist, [.025, .975], axis=0).T.tolist()})
        bootstraps[f'row{len(rows)-1}'] = dist
    for ci, name in enumerate(CONTROLS):
        v = base['baseline_metrics'][ci]
        dist = v[bm].mean(1)
        rows.append({'condition': name, 'kind': 'deterministic_control', 'T': None,
                     'mean': v.mean(0).tolist(), 'interval': np.quantile(dist, [.025, .975], axis=0).T.tolist()})
        bootstraps[f'row{len(rows)-1}'] = dist
    for ci, name in enumerate(['FP16_weights_F64', 'BF16_weights_F64', 'FP16_Torch_CPU', 'BF16_Torch_CPU']):
        v = precision['control_metrics'][ci]
        dist = v[bm].mean(1)
        rows.append({'condition': name, 'kind': 'precision_control', 'T': None,
                     'mean': v.mean(0).tolist(), 'interval': np.quantile(dist, [.025, .975], axis=0).T.tolist()})
        bootstraps[f'row{len(rows)-1}'] = dist
    assert len(rows) == 17
    np.savez_compressed(OUT/'bootstrap_values.npz', **bootstraps)
    (OUT/'figure_data.json').write_text(json.dumps({'rows': rows, 'metric_order': ['accuracy_fraction', 'CE_nats'],
             'bootstrap_values_sha256': sha(OUT/'bootstrap_values.npz'), 'scope': protocol['scope']}, indent=2)+'\n')
    with (OUT/'sampling_curve.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['condition', 'kind', 'samples_per_layer', 'accuracy_fraction', 'accuracy_lo', 'accuracy_hi', 'CE_nats', 'CE_lo', 'CE_hi', 'retained_weight_node_updates_per8imagebank'])
        for r in rows:
            writer.writerow([r['condition'], r['kind'], r['T'] if r['T'] is not None else 'NA',
                             r['mean'][0], *r['interval'][0], r['mean'][1], *r['interval'][1],
                             3928*r['T'] if r['T'] is not None else 0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ts = np.array([64, 256, 1024])
    for metric, ax in enumerate(axes):
        factor = 100 if metric == 0 else 1
        point = np.array([r['mean'][metric] for r in rows[:3]])*factor
        bounds = np.array([r['interval'][metric] for r in rows[:3]])*factor
        ax.errorbar(ts, point, yerr=np.stack([point-bounds[:, 0], bounds[:, 1]-point]),
                    fmt='o-', color='#a34b16', linewidth=2, capsize=4, label='Single p-bit, frozen weights')
        for name, label, color, linestyle in [('FP32_F64', 'FP32 reference', '#333333', '--'),
                ('ACT8_INT4_minmax', 'INT4 min-max + ACT8', '#7651a8', ':'),
                ('ACT8_INT8_minmax', 'INT8 min-max + ACT8', '#287ba2', '-.')]:
            value = next(r for r in rows if r['condition'] == name)['mean'][metric]*factor
            ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.5, label=label)
        ax.set_xscale('log', base=2)
        ax.set_xticks(ts, ['64', '256', '1024'])
        ax.set_xlim(48, 1365)
        ax.set_xlabel('Samples per layer, T (log scale)')
        ax.set_ylabel('Accuracy (%)' if metric == 0 else 'Mean cross-entropy (nats)')
        ax.set_ylim((0, 65) if metric == 0 else (.9, 2.8))
        ax.grid(alpha=.2)
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Compact RGB CIFAR-10: verified frozen-model sampling curve', fontsize=12)
    fig.text(.5, .03, 'Five models; 10 inference seeds per budget; fixed, previously consumed 10,000 test images.\n'
             'Bars: descriptive model/seed intervals, not paired tests. Control levels omit their arithmetic costs.\n'
             'All 14 controls are retained in CSV. No physical energy, latency, or density claim.',
             ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .18, 1, .94))
    fig.savefig(OUT/'frozen_cifar_sampling.svg')
    fig.savefig(OUT/'frozen_cifar_sampling.png', dpi=180)
    plt.close(fig)
    print('All3native points/14controls/170000bootstrap values exported; independent verification pending', flush=True)


if __name__ == '__main__':
    main()
