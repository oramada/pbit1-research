"""Plot only independently audited matched-work LFM3 cohort statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    p = Path('/proc') / str(pid)
    try:
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'start_ticks': fields[19], 'state': fields[0],
                'argv': [v.decode() for v in (p / 'cmdline').read_bytes().split(b'\0') if v]}
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-pid', type=int, required=True)
    args = parser.parse_args()
    root = Path('results/thrml_spw/lfm3_1p7b')
    parent = root / 'word_language'
    out = parent / 'figure'
    out.mkdir(exist_ok=True)
    upstream = identity(args.analysis_pid)
    with (out / 'protocol.json').open('x') as f:
        json.dump({'source_sha256': sha(Path(__file__)), 'upstream_pid': args.analysis_pid,
            'upstream_identity_at_start': upstream, 'scope': 'Plot all five independently audited MC seeds, fixed512targets and resource costs; no new model scores or physical-performance inference.'}, f, indent=2)
    while True:
        try:
            state = json.loads((parent / 'analysis_pipeline.json').read_text())
        except json.JSONDecodeError:
            time.sleep(.2)
            continue
        if state['status'] == 'five_seed_word_statistics_independently_verified':
            break
        assert state['status'] in ['waiting_for_existing_word_cohort', 'running_analysis']
        actual = identity(args.analysis_pid)
        assert upstream is not None and actual is not None and actual['state'] != 'Z'
        assert actual['start_ticks'] == upstream['start_ticks'] and actual['argv'] == upstream['argv']
        (out / 'progress.json').write_text(json.dumps({'stage': 'waiting_on_verified_analysis_process',
            'last_verified_live_unix': time.time(), 'analysis_status': state['status']}) + '\n')
        time.sleep(30)
    audit_path = parent / 'statistics_audit.json'
    statistics_path = parent / 'statistics.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'all_five_paired_seeds_nine_contrasts_and_resource_totals_verified'
    assert audit['statistics_sha256'] == sha(statistics_path)
    s = json.loads(statistics_path.read_text())
    for path, digest in s['inputs_sha256'].items():
        assert sha(Path(path)) == digest
    controls_path = root / 'language_controls_batch/results.json'
    controls = {v['condition']: v['mean_NLL_FP64'] for v in json.loads(controls_path.read_text())['metrics']}
    single = np.asarray([v['single_T1024']['NLL'] for v in s['seeds']])
    word = np.asarray([v['word4_T256']['NLL'] for v in s['seeds']])
    assert single.shape == word.shape == (5,)
    primary = s['primary_single_minus_word']
    differences = np.asarray(primary['per_seed_NLL_difference'])
    np.testing.assert_allclose(single - word, differences, rtol=1e-12, atol=1e-12)
    intervals = []
    for label in ['single_T1024', 'word4_T256']:
        rec = next(v for v in s['secondary_control_contrasts'] if v['name'] == label + '_minus_BF16')
        intervals.append(np.asarray(rec['MC95_t4_NLL_difference_interval']) + controls['BF16'])
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.7), gridspec_kw={'width_ratios': [1.15, 1]})
    fig.subplots_adjust(top=.79, bottom=.29, left=.08, right=.98, wspace=.32)
    fig.suptitle('LFM3-1.7B: one versus four p-bits per weight', y=.97, fontsize=17, fontweight='bold')
    fig.text(.5, .91, 'Matched retained sampling work · 512 fixed prediction targets · 5 Monte Carlo seeds',
             ha='center', color='#424b55', fontsize=12)
    colors = ['#2166ac', '#b35806', '#1b7837', '#762a83', '#d6604d']
    jitter = np.linspace(-.07, .07, 5)
    for i in range(5):
        axes[0].plot([jitter[i], 1 + jitter[i]], [single[i], word[i]], color=colors[i], alpha=.65, linewidth=1.15)
        axes[0].scatter([jitter[i], 1 + jitter[i]], [single[i], word[i]], color=colors[i], s=37, zorder=3)
    means = [s['single_mean_NLL'], s['word_mean_NLL']]
    for index, (mean, interval) in enumerate(zip(means, intervals, strict=True)):
        axes[0].errorbar(index, mean, yerr=np.asarray([[mean - interval[0]], [interval[1] - mean]]),
                         fmt='D', color='#18212b', capsize=5, linewidth=2, markersize=5, zorder=4)
    axes[0].axhline(controls['BF16'], color='#34495e', linestyle='--', linewidth=1.4, label='BF16')
    axes[0].axhline(controls['INT4_affine_g128'], color='#8b939c', linestyle=':', linewidth=1.7, label='INT4')
    axes[0].set_xticks([0, 1], ['1 p-bit\n1,024 samples', '4 p-bits\n256 samples'])
    axes[0].set_xlim(-.32, 1.32)
    axes[0].set_ylim(min(controls['BF16'], single.min(), word.min()) - .12, max(single.max(), word.max(), *[v[1] for v in intervals]) + .12)
    axes[0].set_ylabel('Mean negative log-likelihood (lower is better)')
    axes[0].set_title('Task quality at equal retained work', fontsize=12)
    axes[0].legend(loc='upper right', frameon=False, fontsize=10)
    axes[0].grid(axis='y', alpha=.16)
    interval = primary['MC95_t4_NLL_difference_interval']
    center = primary['mean_NLL_difference']
    axes[1].axhspan(interval[0], interval[1], color='#2166ac', alpha=.13, label='95% MC t4 interval')
    axes[1].axhline(0, color='#34495e', linewidth=1.2)
    axes[1].axhline(center, color='#2166ac', linewidth=1.7, label='Paired mean')
    axes[1].scatter(np.arange(1, 6), differences, color=colors, s=45, zorder=3)
    axes[1].set_xticks(np.arange(1, 6))
    axes[1].set_xlabel('Monte Carlo seed cluster')
    axes[1].set_ylabel('NLL(single) − NLL(four-node)')
    axes[1].set_title('Paired quality difference', fontsize=12)
    axes[1].legend(loc='upper right', frameon=False, fontsize=9)
    axes[1].grid(axis='y', alpha=.16)
    axes[1].text(.5, -.23, f'Mean {center:+.3f}; MC95% [{interval[0]:+.3f}, {interval[1]:+.3f}]\nNegative values favor one p-bit.',
                 ha='center', transform=axes[1].transAxes, fontsize=10)
    retained = s['matched_retained_evaluations_per_method_cohort']
    foot = ('Colored lines pair seeds; black diamonds show means and 95% MC t4 intervals. Intervals condition on this text.\n'
            f'Retained node evaluations: {retained:.3e} per method/cohort, including padding; four-node warmup costs 4× more.\n'
            'Logical p-bit assignments: 1.729B versus 6.915B; simulator banks are time multiplexed.\n'
            'Single controls use 12-bit fields; the internal four-node positional baseline uses two bias levels per node.\n'
            'BF16/INT4 use conventional operators; stochastic execution uses ACT12. Hardware energy/latency are unmeasured.')
    fig.text(.08, .035, foot, fontsize=9, color='#424b55', linespacing=1.5, va='bottom')
    png, svg = out / 'matched_work_language.png', out / 'matched_work_language.svg'
    fig.savefig(png, dpi=170, facecolor='white')
    fig.savefig(svg, facecolor='white')
    plt.close(fig)
    payload = {'status': 'audited_five_seed_matched_work_figure_generated', 'source_sha256': sha(Path(__file__)),
        'statistics_sha256': sha(statistics_path), 'statistics_audit_sha256': sha(audit_path),
        'single_NLLs': single.tolist(), 'word_NLLs': word.tolist(), 'absolute_MC95_intervals': [v.tolist() for v in intervals],
        'primary_contrast': primary, 'outputs_sha256': {str(p): sha(p) for p in [png, svg]},
        'scope': 'Visualizes accepted conditional MC statistics, all five seeds and retained-work normalization. Not a Pareto frontier, unseen-text equivalence, vendor hardware weight format or measured system advantage.'}
    with (out / 'results.json').open('x') as f:
        json.dump(payload, f, indent=2)
    (out / 'progress.json').write_text(json.dumps({'stage': 'complete'}) + '\n')
    print(json.dumps({'status': payload['status'], 'PNG': str(png), 'SVG': str(svg)}))


if __name__ == '__main__':
    main()
