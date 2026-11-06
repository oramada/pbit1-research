"""Report all equal-budget one/two-probe native calibration outcomes."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

OUT = Path('results/thrml_spw/gain_calibrated_operator')


def main():
    audit = json.loads((OUT/'evidence_audit.json').read_text()); assert audit['passed']
    report = json.loads((OUT/'REPORT.json').read_text())
    with (OUT/'metrics.csv').open() as f: rows = list(csv.DictReader(f))
    methods = ['uncalibrated', 'oracle', 'single256', 'two256', 'single1024', 'two1024', 'single4096', 'two4096']
    summaries = []
    for scenario in [0, 1]:
        for method in methods:
            group = [r for r in rows if int(r['scenario']) == scenario and r['method'] == method]
            assert len(group) == 10
            mse = np.array([float(r['observed_total_mse']) for r in group]); ncal = int(group[0]['total_probe_samples'])
            r = {'scenario': scenario, 'log_gain_sd': .2*scenario, 'method': method, 'total_probe_samples': ncal,
                'native_runs': 10, 'MSE_mean': float(mse.mean()), 'approximate_95_margin': float(1.96*mse.std(ddof=1)/np.sqrt(10))}
            for key in ['conditional_bias_floor', 'predicted_sampling_mse', 'observed_sampling_mse', 'bound_clipped_commands', 'gain_guard_low', 'gain_guard_high']:
                r[key] = float(np.mean([float(g[key]) for g in group]))
            if ncal:
                for key in ['estimated_gain_rmse', 'estimated_offset_rmse']:
                    r[key] = float(np.mean([float(g[key]) for g in group]))
                r['probe_frequency_mse_over_prediction'] = float(np.mean([float(g['calibration_frequency_mse'])/float(g['calibration_predicted_frequency_mse']) for g in group]))
            summaries.append(r)
    with (OUT/'summary.csv').open('w', newline='') as f:
        keys = list(dict.fromkeys(k for r in summaries for k in r)); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summaries)
    comparisons = []
    for scenario in [0, 1]:
        for ncal in [256, 1024, 4096]:
            paired = []
            for seed in range(10):
                vals = [float(next(r['observed_total_mse'] for r in rows if int(r['scenario']) == scenario and int(r['seed']) == seed and r['method'] == name+str(ncal))) for name in ['single', 'two']]
                paired.append(vals[1]-vals[0])
            delta = np.array(paired)
            a, b = [next(r for r in summaries if r['scenario'] == scenario and r['method'] == name+str(ncal)) for name in ['single', 'two']]
            comparisons.append({'log_gain_sd': .2*scenario, 'total_probe_samples': ncal,
                'two_minus_single_MSE': float(delta.mean()), 'approximate_paired_95_margin': float(1.96*delta.std(ddof=1)/np.sqrt(10)),
                'ratio_of_mean_two_to_single_MSE': b['MSE_mean']/a['MSE_mean']})
    (OUT/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), layout='constrained')
    xx = [256, 1024, 4096]
    for ax, scenario in zip(axes[0], [0, 1]):
        for name, color, label in [('single', 'tab:blue', 'One probe, gain assumed 1'), ('two', 'tab:orange', 'Two probes, gain + offset')]:
            group = [next(r for r in summaries if r['scenario'] == scenario and r['method'] == name+str(n)) for n in xx]
            ax.errorbar(xx, [r['MSE_mean'] for r in group], yerr=[r['approximate_95_margin'] for r in group], fmt='o-', color=color, label=label)
        for name, color, label in [('uncalibrated', 'tab:red', 'No calibration'), ('oracle', 'tab:green', 'Known-parameter diagnostic')]:
            r = next(r for r in summaries if r['scenario'] == scenario and r['method'] == name)
            ax.axhline(r['MSE_mean'], ls='--', color=color, label=label)
        ax.set(yscale='log', ylabel='Output MSE at T512', title='Unit gain, offset SD .2' if not scenario else 'Log-gain SD .2, offset SD .2')
        ax.legend(fontsize=7)
    for ax, key, title in zip(axes[1], ['estimated_gain_rmse', 'estimated_offset_rmse'], ['Gain estimate, varying-gain devices', 'Offset estimate, varying-gain devices']):
        for name, label in [('single', 'One probe'), ('two', 'Two probes')]:
            group = [next(r for r in summaries if r['scenario'] == 1 and r['method'] == name+str(n)) for n in xx]
            ax.plot(xx, [r[key] for r in group], 'o-', label=label)
        ax.set(yscale='log', ylabel='Parameter RMSE', title=title); ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xscale('log'); ax.set_xticks(xx, ['256', '1024', '4096']); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Total native probe samples / free device'); ax.grid(alpha=.2)
    fig.savefig('results/thrml_spw/gain_calibrated_operator_evidence.png', dpi=180); plt.close(fig)
    lines = ['# Finite-control two-probe calibration under device gain error\n',
        'The same 2,359,296-weight native operator was evaluated with one p-bit per normalized weight, fixed synthetic offsets, and either unit gain or positive varying gain. Inputs are eight previously consumed old development contexts. All scenarios and budgets were fixed before sampling. No neural test data, true parameters or task labels are provided to either measured estimator.\n',
        '## Model and fair comparison\n',
        'Each free device realizes b_real=float32(a*b_command+d), with d~N(0,0.2^2). Varying gains are a=exp(0.2*z-0.2^2/2), z~N(0,1), giving mean1 positive gains; the control scenario has a=1. These are dimensionless synthetic field errors, not measured Extropic device distributions. Gains/offsets stay fixed across probe and inference calls. Exact endpoint clamps remain unperturbed.\n',
        'One-probe calibration uses all Ncal samples at eight-bit code128 and assumes unit gain. Two-probe calibration splits the same total Ncal equally across codes96/159, with fields about-0.9882353/+0.9882353. Both transform Jeffreys-smoothed frequencies through 0.5*logit. Two probes fit a line in command bias to identify gain and offset; inferred gain is clipped to[0.25,4] as an explicit estimator guard. Both procedures round/clamp corrected commands to the same eight-bit range[-4,4]. The guard is not a measured physical specification. All guard and range-clipping counts are stored.\n',
        'Each method then executes T512 native inference with a separate key. Total probe samples are256,1024 or4096 for either method. The known-gain/offset inverse is a diagnostic with finite controls, not an implementable calibration or a guarantee of global optimum. Ten paired realizations vary both device parameters and sampling.\n',
        '## All outcomes\n',
        '| Log-gain SD | Method | Total probe samples | Output MSE | Approx. 95% margin | Conditional bias floor | Gain RMSE | Offset RMSE |\n|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        g = f"{r['estimated_gain_rmse']:.5g}" if r['total_probe_samples'] else '--'
        d = f"{r['estimated_offset_rmse']:.5g}" if r['total_probe_samples'] else '--'
        lines.append(f"| {r['log_gain_sd']:g} | {r['method']} | {r['total_probe_samples']} | {r['MSE_mean']:.6g} | {r['approximate_95_margin']:.3g} | {r['conditional_bias_floor']:.6g} | {g} | {d} |")
    varying = next(r for r in comparisons if r['log_gain_sd'] == .2 and r['total_probe_samples'] == 4096)
    unit = next(r for r in comparisons if r['log_gain_sd'] == 0 and r['total_probe_samples'] == 4096)
    one, two = [next(r for r in summaries if r['scenario'] == 1 and r['method'] == method) for method in ['single4096', 'two4096']]
    lines.insert(2, f"At4096 total calibration reads per free device, two probes reduce varying-gain mean operator MSE from {one['MSE_mean']:.6g} to {two['MSE_mean']:.6g} ({100*(1-varying['ratio_of_mean_two_to_single_MSE']):.2f}% lower). With actual unit gain, the same extra fit instead increases MSE {100*(unit['ratio_of_mean_two_to_single_MSE']-1):.2f}%. At the low256-read varying-gain budget it is also worse. Thus extra identifiability helps only when the device mismatch and read budget justify its estimation noise in these tested scenarios.\n")
    lines.extend(['\n| Log-gain SD | Equal total probe samples | Two minus one-probe MSE | Approx. paired 95% margin | Two / one mean MSE |\n|---:|---:|---:|---:|---:|'])
    for r in comparisons:
        lines.append(f"| {r['log_gain_sd']:g} | {r['total_probe_samples']} | {r['two_minus_single_MSE']:+.6g} | {r['approximate_paired_95_margin']:.3g} | {r['ratio_of_mean_two_to_single_MSE']:.4f} |")
    lines.extend(['\nThe extra probe identifies a second device parameter but divides the read budget and increases estimator noise. All unit-gain and varying-gain outcomes above are retained, including cases where the extra parameter hurts. These approximate normal ten-realization intervals and the six contrasts are not multiplicity-adjusted formal coverage or equivalence tests. Results concern operator MSE, not model accuracy.\n',
        '## Calibration and storage costs\n',
        f"The lower gain-estimate guard activates {sum(int(r['gain_guard_low']) for r in rows)} times across all saved fits; the upper guard activates {sum(int(r['gain_guard_high']) for r in rows)} times. These counts, and command-range clipping by condition, remain explicit in the raw metrics.\n",
        f"Independent audits verify {audit['native_inference_runs']} native inference runs and {audit['native_calibration_probe_runs']} native calibration-probe runs: saved counts, code/field hashes, every conditional mean/error and all resource counters. The executed study uses {report['total_calibration_free_node_evaluations']:,} calibration free-node evaluations and {report['total_inference_free_node_evaluations']:,} inference evaluations. Both measured methods have exactly Ncal calibration draws per free device, but the two-probe method has two distinct command settings rather than one. Programming/settling overhead is therefore not necessarily equal.\n",
        f"Corrected commands retain one uint8 per free node ({report['free_nodes']:,} bytes) plus existing endpoint modes, local ranges and bias metadata. Stored probe counts are int32, two arrays for the two-probe fit. Gain/offset estimates and actual float32 simulation fields are additional temporary/runtime arrays; the method does not store an entire weight in one total binary memory bit.\n",
        'For static devices reused over R eight-context bank invocations, the probe-plus-inference draw ratio relative to inference alone is1+Ncal/(512*R) for either measured method. It is9x at Ncal4096,R1 and1.08x at R100. Readout, programming, accumulation and drift-driven recalibration costs remain separate, so draw ratios are not physical energy or latency predictions.\n',
        'This extends sample-based correction beyond the previous unit-gain assumption. The linear logistic transfer law, stable device parameters and independent unary samples remain assumptions. Drift, nonlinear response, temporal/cross-device correlations and real device measurements still require separate work. Existing successful quarter-model conversion and failed budgets are unchanged. The broad research goal remains active.\n'])
    text = '\n'.join(lines)+'\n'
    for a,b in {'At4096':'At 4096','low256':'low 256','mean1':'mean 1','code128':'code 128','codes96':'codes 96','about-':'about -','to[0.25':'to [0.25','range[-4':'range [-4','are256,1024 or4096':'are 256, 1024 or 4096','is1+':'is 1+','is9x':'is 9x','Ncal4096,R1':'Ncal=4096, R=1','and1.08':'and 1.08','R100':'R=100'}.items(): text=text.replace(a,b)
    Path('results/thrml_spw/GAIN_CALIBRATED_OPERATOR_REPORT.md').write_text(text)
    print(json.dumps({'summaries': summaries, 'paired_comparisons': comparisons}, indent=2))


if __name__ == '__main__': main()
