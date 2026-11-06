"""Describe the completed synthetic nonideal operator study and its limits."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('results/thrml_spw/calibration_operator')


def main():
    audit = json.loads((OUT/'evidence_audit.json').read_text())
    assert audit['passed'] and audit['native_conditions'] == 160
    with (OUT/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    summaries = []
    keys = sorted({(int(r['sigma_index']), int(r['K']), r['dependence'], int(r['observations'])) for r in rows})
    for si, k, dep, obs in keys:
        group = sorted([r for r in rows if (int(r['sigma_index']), int(r['K']), r['dependence'], int(r['observations'])) == (si, k, dep, obs)], key=lambda r: int(r['device_sampling_seed']))
        assert len(group) == 10
        values = np.array([float(r['observed_total_mse']) for r in group])
        summary = {'sigma_b': [0., .05, .2][si], 'K': k, 'dependence': dep, 'observations': obs, 'T': obs//k,
            'realizations': 10, 'MSE_mean': float(values.mean()),
            'approximate_95_margin': float(1.96*values.std(ddof=1)/np.sqrt(10))}
        for key in ['conditional_bias_floor', 'offset_error_floor_vs_nominal', 'predicted_total_mse', 'predicted_sampling_mse', 'observed_sampling_mse']:
            summary[key] = float(np.mean([float(r[key]) for r in group]))
        summary['observed_sampling_over_prediction'] = summary['observed_sampling_mse']/summary['predicted_sampling_mse']
        summaries.append(summary)
    with (OUT/'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    comparisons = []
    for si, sigma in enumerate([.0, .05, .2]):
        for dep in (['independent', 'shared'] if sigma else ['independent']):
            # Pair by device seed: the first spatial replica and temporal device
            # use the same fixed standardized offset, not the same sampling path.
            a = sorted([r for r in rows if int(r['K']) == 1 and int(r['sigma_index']) == si and int(r['observations']) == 512], key=lambda r: int(r['device_sampling_seed']))
            b = sorted([r for r in rows if int(r['K']) == 4 and int(r['sigma_index']) == si and r['dependence'] == dep and int(r['observations']) == 512], key=lambda r: int(r['device_sampling_seed']))
            difference = np.array([float(v['observed_total_mse'])-float(u['observed_total_mse']) for u, v in zip(a, b)])
            comparisons.append({'sigma_b': sigma, 'replica_dependence': dep,
                'spatial_minus_temporal_MSE_at_512': float(difference.mean()),
                'approximate_paired_95_margin': float(1.96*difference.std(ddof=1)/np.sqrt(10))})
    (OUT/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
    for ax, sigma in zip(axes, [0., .05, .2]):
        for k, dep, color, marker in [(1, 'independent', 'tab:blue', 'o'), (4, 'independent', 'tab:orange', 's'), (4, 'shared', 'tab:green', '^')]:
            group = [r for r in summaries if r['sigma_b'] == sigma and r['K'] == k and r['dependence'] == dep]
            if not group:
                continue
            group.sort(key=lambda r: r['observations'])
            label = 'K1 temporal' if k == 1 else 'K4 '+dep
            xx = [r['observations'] for r in group]
            ax.errorbar(xx, [r['MSE_mean'] for r in group], yerr=[r['approximate_95_margin'] for r in group],
                fmt=marker, color=color, label=label, capsize=3)
            ax.plot(xx, [r['predicted_total_mse'] for r in group], '--', color=color, alpha=.8)
            ax.axhline(group[0]['conditional_bias_floor'], ls=':', color=color, alpha=.7)
        ax.set(xscale='log', yscale='log', xticks=[128, 512], xticklabels=['128', '512'],
            xlabel='Total observations / unique weight', title=f'Synthetic field offset SD = {sigma:g}')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel('Conditional output MSE')
    fig.suptitle('Official THRML: markers observed, dashed prediction, dotted bias floor', fontsize=11)
    fig.savefig('results/thrml_spw/calibration_operator_evidence.png', dpi=180); plt.close(fig)
    lines = ['# Fixed calibration error in a native large stochastic operator\n',
        'The same 2,359,296-weight DistilGPT-2 block-2 MLP-down projection executes through official THRML with explicit K1 or K4 replica nodes. Inputs are the first eight previously consumed old development contexts, not fresh neural-quality confirmation. All 160 planned conditions and raw counters passed an independent audit. CPU/CUDA trajectory checks compare dynamic unary fields against separately rebuilt public THRML factors.\n',
        '## Frozen scenarios\n',
        'Starting from the same packed eight-bit control codes and local affine ranges, each free device realizes b_real=float32(b_code+sigma_b*epsilon), with sigma_b in {0,0.05,0.2}. Offsets are fixed across all time samples and both budgets. Independent replicas have distinct epsilon values; shared replicas reuse the same epsilon for each weight. Distinct weights have independent offsets in both cases. Exact endpoint clamps are not perturbed. These dimensionless offsets are synthetic stress scenarios, not measured Extropic calibration specifications, dynamic drift, cross-device sampling correlations or voltage units.\n',
        'Each condition uses 128 or 512 total observations per unique weight: K1/T128 versus K4/T32, and K1/T512 versus K4/T128. At a given observation budget, total free updates, logical readouts and nominal binary-product terms match. Ten paired realizations vary both device offsets and sampling streams; error bars are approximate normal margins across these joint realizations, not ten pure Monte Carlo repeats on a fixed physical chip. Device offsets share the first replica for pairing, but sampled trajectories have disjoint keys.\n',
        '## Results\n',
        '| Offset SD | K | Replica offsets | Observations | Steps | Observed MSE | Approx. 95% margin | Conditional bias floor | Sampling MSE / prediction |\n|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['sigma_b']:g} | {r['K']} | {r['dependence']} | {r['observations']} | {r['T']} | {r['MSE_mean']:.6g} | {r['approximate_95_margin']:.3g} | {r['conditional_bias_floor']:.6g} | {r['observed_sampling_over_prediction']:.4f} |")
    lines.extend(['\nThe conditional expectation and bias floor are computed analytically from the actual rounded perturbed fields and saved inputs; finite-T errors are measured from native sample outputs. More temporal samples reduce conditional sampling variance as 1/T but do not change this fixed mean-error floor. At zero offset, both representations retain the same finite-code bias floor. For shared replica offsets, the K4 and K1 conditional means and floors agree for every device seed. Independent replicas average different device errors, so their conditional floors can differ. Nonlinearity of sigmoid can leave a population mean shift even when offsets are zero-mean; a universal exact 1/K total-bias law is not assumed. The error relative to the nominal programmed mean is stored separately, and cannot simply be added to the nominal control-bias MSE because of a cross term.\n',
        '| Offset SD | K4 offset dependence | K4 minus K1 MSE at 512 observations | Approx. paired 95% margin |\n|---:|---|---:|---:|'])
    for r in comparisons:
        lines.append(f"| {r['sigma_b']:g} | {r['replica_dependence']} | {r['spatial_minus_temporal_MSE_at_512']:.6g} | {r['approximate_paired_95_margin']:.3g} |")
    zero = comparisons[0]
    lines.append(f"\nThe zero-offset ten-seed sample has K4-minus-K1 MSE {zero['spatial_minus_temporal_MSE_at_512']:.6g} with unadjusted approximate margin {zero['approximate_paired_95_margin']:.3g}; this interval excludes zero. That finite-seed discrepancy is preserved, not treated as formal equivalence or a systematic spatial advantage. The five displayed contrasts are not multiplicity-adjusted, and normal ten-seed margins have limited coverage accuracy. Both conditional analytic targets remain identical at zero offset. This discrepancy motivated a fixed additional-seed diagnostic.\n")
    followup = Path('results/thrml_spw/zero_offset_recheck')
    if (followup/'evidence_audit.json').exists():
        assert json.loads((followup/'evidence_audit.json').read_text())['passed']
        fresh = next(r for r in json.loads((followup/'summary.json').read_text()) if r['batch'] == 'fresh40')
        lines.append(f"\nFollow-up now completed: exactly forty new seeds were fixed and audited separately. Their K4-minus-K1 difference is {fresh['K4_minus_K1_mean']:.7g} with approximate margin {fresh['approximate_paired_95_margin']:.7g}; the interval includes zero: {fresh['interval_includes_zero']}. Fresh K1/K4 sampling MSEs are {fresh['K1_sampling_over_prediction']:.4f}/{fresh['K4_sampling_over_prediction']:.4f} times the identical analytic prediction. The original ten-seed result above remains unchanged; see ZERO_OFFSET_RECHECK_REPORT.md. This is a fixed diagnostic follow-up, not equivalence proof.\n")
    lines.extend(['\nThis is an operator-level stress test, not a model-accuracy comparison, positional multi-bit encoding or an official Extropic learned-weight baseline. Four replicas require four times the logical nodes and have one quarter the parallel sampling depth. Equal update counts do not establish equal energy, and independent physical calibration errors cannot be assumed without device measurements. A practical one-p-bit design must either control these offsets, calibrate them, train for them, or tolerate the resulting mean error. Calibration overhead and the precision of the controller remain part of the systems problem.\n',
        'Raw cases, fixed-field hashes, source/data lineage, metrics and independent audit are under calibration_operator/. The goal remains active for device calibration/drift, broader conversion, training and physical measurements.\n'])
    Path('results/thrml_spw/CALIBRATION_OPERATOR_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'summary_rows': len(summaries), 'comparisons': comparisons}, indent=2))


if __name__ == '__main__':
    main()
