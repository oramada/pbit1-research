"""Render audited matched-observation probe-placement results without new sampling."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('results/thrml_spw/probe_placement')

def rows(name):
    with (OUT / name).open() as handle:
        return list(csv.DictReader(handle))

def main():
    for name in ['evidence_audit.json', 'cost_evidence_audit.json']:
        assert json.loads((OUT/name).read_text())['passed']
    task = rows('summary.csv')
    cost = rows('calibration_summary.csv')
    base = rows('baseline_metrics.csv')
    pair = json.loads((OUT/'paired_summary.json').read_text())
    fixed, candidate = cost
    reduction = 100*(1-float(candidate['weight_MSE'])/float(fixed['weight_MSE']))
    pred = 100*(1-float(candidate['nominal_Fisher_predicted_MSE'])/float(fixed['nominal_Fisher_predicted_MSE']))
    ck = 'analytic_expected_recorded_calibration_changes_per_profile'
    ik = 'analytic_expected_recorded_inference_changes_per_profile'
    cal_extra = 100*(float(candidate[ck])/float(fixed[ck])-1)
    total_extra = 100*((float(candidate[ck])+float(candidate[ik]))/(float(fixed[ck])+float(fixed[ik]))-1)
    fig, axes = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    x = np.arange(2)
    labels = ['Fixed probes','Weight-only Fisher']
    point = np.array([100*(float(r['relative_perplexity'])-1) for r in task])
    upper = np.array([100*(float(r['bootstrap_relative_upper'])-1) for r in task])
    axes[0,0].plot(x,point,'o',label='Point estimate')
    axes[0,0].plot(x,upper,'_',ms=14,label='Crossed upper sensitivity bound')
    axes[0,0].axhline(5,color='black',ls='--',label='Frozen 5% limit')
    axes[0,0].set(xticks=x,xticklabels=labels,ylabel='PPL increase over fresh FP32 (%)',title='Both strict quality gates fail')
    axes[0,0].legend(fontsize=7)
    axes[0,1].bar(x-.17,[float(r['weight_MSE']) for r in cost],width=.34,label='Actual calibrated weight MSE')
    axes[0,1].bar(x+.17,[float(r['nominal_Fisher_predicted_MSE']) for r in cost],width=.34,label='Nominal first-order prediction')
    axes[0,1].set(xticks=x,xticklabels=labels,ylabel='Mean squared signed-weight error',title=f'Actual weight MSE reduced {reduction:.2f}%')
    axes[0,1].legend(fontsize=7)
    y=pair['CE_difference']
    axes[1,0].errorbar([0],[y],yerr=[[y-pair['CE_sensitivity_low']],[pair['CE_sensitivity_high']-y]],fmt='o',capsize=5)
    axes[1,0].axhline(0,color='black',ls='--')
    axes[1,0].set(xticks=[0],xticklabels=['Fisher minus fixed'],ylabel='CE difference (nats / target)',title='Paired uncertainty includes zero')
    axes[1,1].bar(x,[float(r[ck])/1e9 for r in cost],label='Calibration')
    axes[1,1].bar(x,[float(r[ik])/1e9 for r in cost],bottom=[float(r[ck])/1e9 for r in cost],label='Inference')
    axes[1,1].set(xticks=x,xticklabels=labels,ylabel='Expected recorded state changes (billions)',title='Post-hoc IID activity proxy; not energy')
    axes[1,1].legend(fontsize=7)
    for ax in axes.ravel(): ax.grid(axis='y',alpha=.2)
    fig.savefig('results/thrml_spw/probe_placement_evidence.png',dpi=180)
    plt.close(fig)
    parts=['# Weight-only probe placement improves calibration; neural benefit remains uncertain\n']
    parts.append(f'At equal native sampling budgets, weight-only probe placement reduces calibrated weight MSE by **{reduction:.2f}%**. Fresh-text perplexity changes from **{float(task[0]["perplexity"]):.3f} to {float(task[1]["perplexity"]):.3f}**, but the paired cross-entropy sensitivity interval includes zero. **Both methods fail the predeclared strict quality gate.** This is evidence for improved weight calibration in this regime, with no confirmed task-quality or hardware advantage. The broad goal remains active.\n')
    parts.append('## Prospective design\n')
    parts.append('Before new sampling or text scoring, a nominal Fisher-information rule chooses one symmetric probe-code pair for each of twelve converted projections. It uses only nominal target codes and local weight spans, assumes unit gain and zero offset, and evaluates all 128 symmetric pairs on the eight-bit [-4,4] controller grid. If q=σ(2b), the squared derivative of a signed weight with respect to field is s=(2 span q(1-q))². The score propagates the two binomial probe-logit variances through the affine gain/offset inverse and averages the resulting first-order weight error. The matrix inverse and all 1,536 candidate scores were independently checked. Finite-estimator bias, actual device variability, controller clipping and rounding are outside this approximation.\n')
    parts.append(f'The selected code pairs are 110/145 or 111/144 (fields near ±0.55 or ±0.52), compared with fixed 96/159 (near ±0.99). The pooled nominal prediction is {pred:.2f}% lower weight MSE. Each method uses two actual THRML probe windows of 2,048 observations per weight (P4096) and T1024 for inference, with the same five fresh paired device/sampling profiles. Device profiles have synthetic static lognormal gain and normal offset variability. The estimator sees only requested probe fields and actual native counts; true device parameters are not available to control selection or fitting. No oracle correction, task-based selection, extra probes or rescue budget is used.\n')
    parts.append('One pinned DistilGPT-2 checkpoint has its first three blocks\' twelve projections converted: 21,233,664 weights, 25.92% of its unique parameters. Every converted normalized signed weight has one free THRML p-bit; all remaining parameters stay FP32. Accumulation precedes the original nonlinearities. This is not full-model conversion or direct p=raw signed weight. The experiment uses 64 new nonoverlapping 65-token WikiText-2 windows (4,096 targets), excluding previously consumed positions. Eight contexts share each sampled matrix bank. Official THRML 0.1.4 is pinned at 2d54cd8404df85bb65703d7e8b4e84e48c1b72a9.\n')
    parts.append('## Task results\n\n| Condition | Perplexity | Token accuracy (%) |\n|---|---:|---:|')
    for r in base: parts.append(f'| {r["condition"]} | {float(r["perplexity"]):.3f} | {100*float(r["token_accuracy"]):.3f} |')
    for r in task: parts.append(f'| {r["method"]} | {float(r["perplexity"]):.3f} | {100*float(r["token_accuracy"]):.3f} |')
    parts.append('\n| Native condition | PPL increase (%) | Crossed upper bound (%) | Point gate | Strict gate |\n|---|---:|---:|---|---|')
    for r in task: parts.append(f'| {r["method"]} | {100*(float(r["relative_perplexity"])-1):.3f} | {100*(float(r["bootstrap_relative_upper"])-1):.3f} | {r["point_passed"]} | {r["passes"]} |')
    parts.append(f'\nThe paired Fisher-minus-fixed CE difference is {pair["CE_difference"]:+.6f}, with crossed sensitivity interval [{pair["CE_sensitivity_low"]:+.6f}, {pair["CE_sensitivity_high"]:+.6f}] nats/target. The PPL ratio is {pair["perplexity_ratio"]:.6f}; the accuracy difference is {100*pair["accuracy_difference"]:+.4f} percentage points. These intervals cross five joint device/sampling seeds with eight whole context banks using 10,000 fixed-seed resamples. They describe sensitivity under exchangeable clusters, not guaranteed coverage or equivalence. Both frozen strict rules require the PPL upper bound within 5% and accuracy lower bound within 1 percentage point of FP32. The Fisher point estimate meets 5%, but its upper bound does not. The prior failed measured-calibration regime remains unchanged.\n')
    parts.append('## Weight error and resource accounting\n\n| Method | Weight MSE | Gain RMSE | Offset RMSE |\n|---|---:|---:|---:|')
    for r in cost: parts.append(f'| {r["method"]} | {float(r["weight_MSE"]):.9g} | {float(r["gain_RMSE"]):.6f} | {float(r["offset_RMSE"]):.6f} |')
    parts.append(f'\nMore central probes improve offset estimates while worsening gain estimates. Actual pooled weight MSE improves {reduction:.2f}%, compared with the nominal {pred:.2f}% prediction. This does not guarantee task-quality improvement.\n')
    parts.append('The ten production model runs contain 120 native operators and 240 probe windows: 1,739,461,754,880 inference and 869,730,877,440 calibration observed free-node updates. Independent native replay repeats those budgets for verification. Calibration adds 50% of inference observations for the tested stream in each method. Control/range/bias payload remains 8.53125 bits per converted weight, excluding software workspace. Stage-specific probe commands require 24 bytes instead of two shared fixed codes (22 additional bytes), plus offline selection work. One free stochastic node does not mean one total stored bit.\n')
    parts.append(f'A separately audited **post-hoc analytic activity proxy** uses p+2(T-1)p(1-p) expected differences between IID recorded states for each False-initialized sample call. More central probes increase the calibration proxy by {cal_extra:.2f}% and total calibration-plus-inference proxy by {total_extra:.2f}%. These are not observed native flip traces, physical thermal transitions, energy, sampling rates or latency measurements. No choices or quality rules were changed using this supplement. Equal observation counts do not establish equal energy.\n')
    parts.append('## Evidence and reproduction\n')
    parts.append('All 240 probes and 120 operators passed independently constructed public-THRML replay. Independent original-model hooks regenerated all 120 runtime input arrays with exact NPY SHA matches and reproduced final per-token losses/predictions for all ten model cases. Separate audits verified all deterministic baselines, 130 prefix rows, raw task metrics, 60,000 bootstrap values and pooled diagnostic/resource totals. Native output arrays, probe counts, corrected codes and per-token evidence remain stored. To conserve disk, runtime inputs are regenerated from pinned text/model and preceding saved outputs; their exact original hashes are retained and temporary copies are removed after successful checks. This does not discard unique stochastic observations.\n')
    parts.append('```bash\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.probe_placement_transformer\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.audit_probe_placement_paths\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_probe_placement\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_probe_placement_costs\n.venv-thrml/bin/python -m research.thrml_spw.probe_placement_report\n```\n')
    parts.append('The completed-driver cache check passed without new native samples or manifest changes. Audits intentionally replay samples from preserved keys. Raw files and frozen design are in `probe_placement/`; the figure is `probe_placement_evidence.png`. Future practical work should use prospective validation-only budget/estimator choices followed by separate confirmation. Physical transfer functions, correlation, drift, full-model conversion and measured system costs remain open.\n')
    Path('results/thrml_spw/PROBE_PLACEMENT_REPORT.md').write_text('\n'.join(parts)+'\n')
    print(json.dumps({'weight_MSE_reduction_percent':reduction,'nominal_reduction_percent':pred,'calibration_activity_proxy_extra_percent':cal_extra,'total_activity_proxy_extra_percent':total_extra,'both_strict_gates_failed':all(r['passes']=='False' for r in task)},indent=2))

if __name__=='__main__': main()
