"""Measured native calibration results, with explicit amortization accounting."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('results/thrml_spw/calibrated_operator')


def main():
    audit = json.loads((OUT/'evidence_audit.json').read_text()); assert audit['passed']
    report = json.loads((OUT/'REPORT.json').read_text())
    with (OUT/'metrics.csv').open() as f: rows = list(csv.DictReader(f))
    methods = ['uncalibrated', 'oracle_offset', 'measured_256', 'measured_1024', 'measured_4096']
    summaries = []
    for method in methods:
        group = sorted([r for r in rows if r['method'] == method], key=lambda r: int(r['seed']))
        assert len(group) == 10
        mse = np.array([float(r['observed_total_mse']) for r in group])
        r = {'method': method, 'Ncal': int(group[0]['calibration_samples']), 'native_runs': 10,
            'MSE_mean': float(mse.mean()), 'approximate_95_margin': float(1.96*mse.std(ddof=1)/np.sqrt(10))}
        for key in ['conditional_bias_floor', 'predicted_sampling_mse', 'observed_sampling_mse', 'offset_error_floor_vs_nominal', 'bound_clipped_commands']:
            r[key] = float(np.mean([float(g[key]) for g in group]))
        if r['Ncal']:
            r['estimated_offset_rmse'] = float(np.mean([float(g['estimated_offset_rmse']) for g in group]))
            r['calibration_frequency_mse_over_prediction'] = float(np.mean([float(g['calibration_frequency_mse'])/float(g['calibration_predicted_frequency_mse']) for g in group]))
        summaries.append(r)
    with (OUT/'summary.csv').open('w', newline='') as f:
        fields = list(dict.fromkeys(k for r in summaries for k in r)); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summaries)
    uncal = np.array([float(next(r['observed_total_mse'] for r in rows if r['method'] == 'uncalibrated' and int(r['seed']) == s)) for s in range(10)])
    comparisons = []
    for method in methods[1:]:
        values = np.array([float(next(r['observed_total_mse'] for r in rows if r['method'] == method and int(r['seed']) == s)) for s in range(10)])
        delta = values-uncal
        comparisons.append({'method': method, 'MSE_difference_vs_uncalibrated': float(delta.mean()),
            'approximate_paired_95_margin': float(1.96*delta.std(ddof=1)/np.sqrt(10)),
            'ratio_of_mean_MSE_to_uncalibrated': float(values.mean()/uncal.mean())})
    (OUT/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    measured = summaries[2:]; xx = [r['Ncal'] for r in measured]
    axes[0].errorbar(xx, [r['MSE_mean'] for r in measured], yerr=[r['approximate_95_margin'] for r in measured], fmt='o-', label='Measured calibration')
    for row, color, label in [(summaries[0], 'tab:red', 'Uncalibrated offset'), (summaries[1], 'tab:green', 'Known-offset oracle')]:
        axes[0].axhline(row['MSE_mean'], ls='--', color=color, label=label)
    axes[0].set(xscale='log', yscale='log', xlabel='Native probe samples / free device', ylabel='Output MSE at T512', title='One p-bit / encoded weight'); axes[0].legend(fontsize=7)
    axes[1].plot(xx, [r['estimated_offset_rmse'] for r in measured], 'o-', label='Measured offset estimate')
    axes[1].plot(xx, 1/np.sqrt(xx), '--', label='1 / sqrt(Ncal) reference')
    axes[1].set(xscale='log', yscale='log', xlabel='Native probe samples / free device', ylabel='Offset estimation RMSE', title='Calibration comes from samples'); axes[1].legend(fontsize=7)
    reuse = np.array([1, 10, 100, 1000])
    amortization = []
    for ncal in [256, 1024, 4096]:
        ratios = 1+ncal/(512*reuse)
        axes[2].plot(reuse, ratios, 'o-', label=f'Ncal={ncal}')
        amortization.extend({'Ncal': ncal, 'bank_reuse_count': int(n), 'draw_count_ratio': float(r)} for n, r in zip(reuse, ratios))
    axes[2].set(xscale='log', xlabel='Eight-context bank invocations / calibration', ylabel='(Inference + amortized probe) draws / inference draws', title='Draw accounting; not energy'); axes[2].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig('results/thrml_spw/calibrated_operator_evidence.png', dpi=180); plt.close(fig)
    (OUT/'amortization.json').write_text(json.dumps({'equation': '1 + Ncal/(512*Nreuse)',
        'assumption': 'Same fixed additive offset survives all bank invocations. Probe updates and inference updates counted as draws, not assumed equal energy. Programming/readout/recalibration/accumulation costs excluded.', 'values': amortization}, indent=2)+'\n')
    lines = ['# Sample-based calibration reduces single-p-bit operator error\n',
        'A 2,359,296-weight projection with one logical p-bit per normalized weight was evaluated under the fixed synthetic sigma_b=0.2 field-offset scenario. All ten offset realizations use the same old development activation inputs as the stress study. This is one operator, not new model-quality confirmation or physical hardware.\n',
        '## Procedure\n',
        'For each free device, program actual eight-bit code128 in [-4,4], whose decoded bias is +0.0156862754 (zero is not a representable code). Collect 256,1024 or4096 samples through the official THRML Gibbs sampler. From counts C, compute p_hat=(C+0.5)/(Ncal+1) and delta_hat=0.5*logit(p_hat)-probe_bias. The measured estimator receives no true offset or task labels. Program the nearest eight-bit code to clip(nominal_bias-delta_hat,-4,4), then execute T512 inference using separate keys. The fixed simulated device offset affects both the probe and inference. Endpoint clamps remain exact and are not calibrated.\n',
        'Uncalibrated inference and a known-offset oracle are separate controls. The oracle receives true offsets and is only a diagnostic; even it retains finite control range and quantization. No calibration budget is selected after seeing outputs; all three are reported. Ten realizations vary both device offsets and sampling streams. Normal ten-realization margins are approximate, not formal corpus/device coverage.\n',
        '| Method | Probe samples / free device | Output MSE | Approx. 95% margin | Conditional bias floor | Offset RMSE | Probe frequency MSE / theory |\n|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        offset = f"{r['estimated_offset_rmse']:.5f}" if r['Ncal'] else '--'
        frequency = f"{r['calibration_frequency_mse_over_prediction']:.4f}" if r['Ncal'] else '--'
        lines.append(f"| {r['method']} | {r['Ncal']} | {r['MSE_mean']:.6g} | {r['approximate_95_margin']:.3g} | {r['conditional_bias_floor']:.6g} | {offset} | {frequency} |")
    best = comparisons[-1]
    lines.extend([f"\nAt the4096-sample probe budget, mean output MSE is {summaries[-1]['MSE_mean']:.6g}, versus {summaries[0]['MSE_mean']:.6g} uncalibrated: a {100*(1-best['ratio_of_mean_MSE_to_uncalibrated']):.2f}% reduction in this synthetic operator-error metric. The paired difference is {best['MSE_difference_vs_uncalibrated']:.6g} with approximate margin {best['approximate_paired_95_margin']:.3g}. This is not an accuracy improvement percentage or a hardware speed/energy gain. The zero-offset nominal finite-control conditional bias floor is {report['nominal_control_bias_floor']:.6g}; finite calibration, quantized correction and range limits leave residual error.\n",
        '## Costs and limits\n',
        f"The executed study contains30 native calibration-count runs and50 native inference runs, independently audited from stored counts, commanded codes, realized-field hashes, raw outputs and fixed device offsets. Across all ten offset realizations and all budgets/controls, calibration executes {report['total_calibration_free_node_evaluations']:,} free-node evaluations and inference executes {report['total_inference_free_node_evaluations']:,}. Calibration counts are stored as int32; corrected command codes use one uint8 per free device ({report['free_nodes']:,} bytes) plus existing modes/ranges/biases. Temporary software fields, estimates and sampler objects add memory. This is not one total storage bit per weight.\n",
        'For one calibration budget reused over Nreuse eight-context bank invocations, draw-count overhead relative to T512 alone is 1+Ncal/(512*Nreuse). At Ncal4096 this is9x for one invocation,1.08x over100 invocations and1.008x over1000. That assumes the additive offsets remain fixed. Probe and inference draws need different readout/accumulation, and programming, buffering, readout and drift-driven recalibration costs remain separate. These ratios are not energy or latency predictions.\n',
        'A single probe only identifies additive offset under the assumed unit gain. Real gain error, hysteresis, temporal/cross-device correlation, readout errors and drift can invalidate this correction and require additional characterization. No claim is made about achievable calibration stability or precision on Extropic hardware. The positive result is narrower: under the stated nonideal simulator model, an implementable count-based finite-control procedure substantially reduces the fixed bias floor without adding replica p-bits.\n',
        'Native count accumulation was checked against separately constructed public THRML factor trajectories on CPU and CUDA; an independent auditor reconstructs every measured command directly from saved counts without importing the estimator. The long-horizon research goal stays active.\n'])
    text = '\n'.join(lines)+'\n'
    for a, b in {'256,1024 or4096':'256, 1024 or 4096','the4096':'the 4096','contains30':'contains 30','and50':'and 50','is9x':'is 9x','invocation,1.08':'invocation, 1.08','over100 ':'over 100 ','and1.008':'and 1.008','over1000':'over 1000','Ncal4096':'Ncal=4096'}.items(): text = text.replace(a, b)
    Path('results/thrml_spw/CALIBRATED_OPERATOR_REPORT.md').write_text(text)
    print(json.dumps({'summaries': summaries, 'comparisons': comparisons}, indent=2))


if __name__ == '__main__': main()
