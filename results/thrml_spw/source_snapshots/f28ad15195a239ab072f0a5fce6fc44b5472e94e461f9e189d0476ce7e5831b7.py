"""Report native drift retention with changing fields and explicit probe clock cost."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/thrml_spw/drift_retention_operator')


def load(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed']
    report=json.loads((OUT/'REPORT.json').read_text());rows=load(OUT/'query_metrics.csv');cal=load(OUT/'calibration_metrics.csv')
    summaries=[]
    for sigma in [0.,.01,.04]:
        for age in [0,8,32]:
            for policy in (['retain'] if age==0 else ['retain','refresh']):
                group=[r for r in rows if float(r['sigma'])==sigma and int(r['age'])==age and r['policy']==policy];assert len(group)==10
                r={'sigma':sigma,'age':age,'policy':policy,'native_runs':10}
                for key in ['observed_total_mse','conditional_mean_error_mse','predicted_sampling_mse','observed_sampling_mse']:
                    values=np.array([float(g[key]) for g in group]);r[key]=float(values.mean())
                    if key in ['observed_total_mse','conditional_mean_error_mse']:r[key+'_approx95_margin']=float(1.96*values.std(ddof=1)/np.sqrt(10))
                for key in ['counterfactual_calibration_events','counterfactual_active_sweeps','counterfactual_idle_sweeps','query_end_clock_sweeps']:
                    assert len({g[key] for g in group})==1;r[key]=int(group[0][key])
                summaries.append(r)
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    comparisons=[]
    for sigma in [0.,.01,.04]:
        for age in [8,32]:
            dif=[]
            for seed in range(10):
                values=[float(next(r['observed_total_mse'] for r in rows if int(r['seed'])==seed and float(r['sigma'])==sigma and int(r['age'])==age and r['policy']==policy)) for policy in ['retain','refresh']]
                dif.append(values[1]-values[0])
            a,b=[next(r for r in summaries if r['sigma']==sigma and r['age']==age and r['policy']==policy) for policy in ['retain','refresh']]
            delta=np.array(dif);margin=float(1.96*delta.std(ddof=1)/np.sqrt(10))
            comparisons.append({'sigma':sigma,'age':age,'refresh_minus_retained_MSE':float(delta.mean()),'approximate_paired_95_margin':margin,
                'ratio_of_mean_refreshed_to_retained_MSE':b['observed_total_mse']/a['observed_total_mse'],'unadjusted_interval_includes_zero':bool(abs(delta.mean())<=margin)})
    (OUT/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    cal_summary=[]
    for sigma in [0.,.01,.04]:
        for start in [0,64,256]:
            group=[r for r in cal if float(r['sigma'])==sigma and int(r['start_segment'])==start];assert len(group)==10
            r={'sigma':sigma,'start_segment':start,'end_clock_intervals':(start+64)/8}
            for key in ['estimated_gain_rmse','estimated_offset_rmse_at_end','gain_guard_low','gain_guard_high','bound_clipped_commands']:
                r[key]=float(np.mean([float(g[key]) for g in group]))
            cal_summary.append(r)
    with (OUT/'calibration_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cal_summary[0]));w.writeheader();w.writerows(cal_summary)
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
    for ax,sigma in zip(axes,[0.,.01,.04]):
        for policy,color in [('retain','tab:blue'),('refresh','tab:orange')]:
            group=[next(r for r in summaries if r['sigma']==sigma and r['age']==age and r['policy']==('retain' if age==0 else policy)) for age in [0,8,32]]
            ax.errorbar([0,8,32],[r['observed_total_mse'] for r in group],yerr=[r['observed_total_mse_approx95_margin'] for r in group],fmt='o-',color=color,
                label='Retain initial codes' if policy=='retain' else 'Refresh before query')
        ax.set(yscale='log',xticks=[0,8,32],xlabel='Age after initial calibration\n(512-sweep intervals)',title=f'Synthetic drift SD / interval = {sigma:g}')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Native output MSE at T512')
    fig.savefig('results/thrml_spw/drift_retention_operator_evidence.png',dpi=180);plt.close(fig)
    costs=[]
    for age in [0,8,32]:
        for policy in (['retain'] if age==0 else ['retain','refresh']):
            g=next(r for r in summaries if r['age']==age and r['policy']==policy)
            costs.append({k:g[k] for k in ['age','policy','counterfactual_calibration_events','counterfactual_active_sweeps','counterfactual_idle_sweeps','query_end_clock_sweeps']})
    (OUT/'clock_cost_ledger.json').write_text(json.dumps({'rows':costs,'interval_sweeps':512,
        'calibration_sweeps_per_event':4096,'query_sweeps':512,'programming_and_fitting_time_in_clock':False,
        'scope':'Counterfactual schedules sharing an exogenous drift path; refresh occupies an idle gap. Idle physical energy, real switching rate and throughput not measured.'},indent=2)+'\n')
    highest=next(r for r in comparisons if r['sigma']==.04 and r['age']==32)
    a,b=[next(r for r in summaries if r['sigma']==.04 and r['age']==32 and r['policy']==policy) for policy in ['retain','refresh']]
    text=['# Native calibration retention with drift during probes and inference\n',
        f"At synthetic drift SD0.04 per512-sweep interval and age32 intervals after initial calibration, mean retained-code MSE is {a['observed_total_mse']:.6g}, versus {b['observed_total_mse']:.6g} after a fresh calibration ({100*(1-highest['ratio_of_mean_refreshed_to_retained_MSE']):.2f}% lower). The paired difference is {highest['refresh_minus_retained_MSE']:+.6g} with approximate margin {highest['approximate_paired_95_margin']:.3g}. This measures one old-input stochastic operator, not model accuracy or a hardware recalibration interval in seconds.\n",
        '## Clock and device model\n',
        'The single-p-bit operator has2,359,296 normalized weight nodes. Its fixed gains have positive lognormal mean1/log-SD0.2 and initial additive field offsets have SD0.2. A modeled interval is512 Gibbs-sweep ticks. Offsets follow a piecewise Brownian path, changing after each64-sweep segment by an independent normal increment of SD sigma/sqrt(8), with sigma in {0,0.01,0.04}. Parameters drift during both finite control probes and during inference. Exact endpoint clamps remain unperturbed. This is a stated discrete synthetic model, not measured device physics or a continuous-time convergence result.\n',
        'Every measured calibration uses4096 samples per free device:2048 at actual eight-bit code96 followed by2048 at code159. It therefore occupies eight clock intervals, not zero time. Counts are summed from32 native segments per probe. The same smoothed-logit gain/offset fit (declared gain guard[0.25,4]) returns nearest bounded eight-bit commands. The estimator does not receive true parameters, the drift path or labels. Counts now pool changing probabilities, so a stationary binomial model for those pooled calibration counts would be wrong.\n',
        'The initial calibration spans segments0..63. Queries start at segments64,128 or320, corresponding to ages0,8 or32 intervals after it finishes. Retained controls always use that initial fit. Refreshed controls use a new eight-interval probe ending at the same query start (segments64..127 or256..319). Age0 has one shared baseline, not duplicate inference evidence.\n',
        'These are separate counterfactual schedules coupled to the same absolute-time device path. The refreshed policy uses a gap where the retained policy idles. Idle time still advances the modeled offset. No throughput improvement or free refresh is inferred: probe work is counted, programming/settling/fitting time is not included in the clock, and idle device transition energy is not measured. Drift is independent of commands, heating and sampled states in this model.\n',
        '## All native query outcomes\n',
        '| Drift SD / interval | Age | Policy | Native MSE | Approx. 95% margin | Conditional mean-error MSE | Predicted sampling MSE |\n|---:|---:|---|---:|---:|---:|---:|']
    for r in summaries:text.append(f"| {r['sigma']:g} | {r['age']} | {r['policy']} | {r['observed_total_mse']:.6g} | {r['observed_total_mse_approx95_margin']:.3g} | {r['conditional_mean_error_mse']:.6g} | {r['predicted_sampling_mse']:.6g} |")
    text.extend(['\n| Drift SD | Age | Refresh minus retained MSE | Approx. paired 95% margin | Refreshed / retained mean MSE |\n|---:|---:|---:|---:|---:|'])
    for r in comparisons:text.append(f"| {r['sigma']:g} | {r['age']} | {r['refresh_minus_retained_MSE']:+.6g} | {r['approximate_paired_95_margin']:.3g} | {r['ratio_of_mean_refreshed_to_retained_MSE']:.4f} |")
    text.extend(['\nTen paired realizations vary initial devices, Brownian paths, calibration and inference sampling. Normal ten-seed margins and six displayed contrasts are approximate and not multiplicity-adjusted formal coverage. Curves connect fixed query ages, not one continuously run periodic recalibration policy. Zero-drift retained conditional means agree exactly across ages in the independent audit; any observed zero-drift query difference is finite sampling. No success threshold or recalibration interval is selected from these outcomes.\n',
        '## Drift during calibration limits refresh\n',
        '| Drift SD | Probe start segment | Probe ends at interval | Gain-fit RMSE | Offset RMSE against probe-end device state |\n|---:|---:|---:|---:|---:|'])
    for r in cal_summary:text.append(f"| {r['sigma']:g} | {r['start_segment']} | {r['end_clock_intervals']:g} | {r['estimated_gain_rmse']:.6g} | {r['estimated_offset_rmse_at_end']:.6g} |")
    text.extend(['\nOffset changes between the low and high probes can be mistaken for gain mismatch; averaging counts over a long probe also lags its final device state. Fresh calibration is therefore not assumed to recover the zero-drift floor. Faster probes, interleaving, tracking or other estimators could alter this tradeoff and need separate evaluation.\n',
        'For query segments j=1..8, conditional on the full drift path, mean output uses qbar=mean_j(q_j). Its sampling variance uses mean_j(q_j*(1-q_j))/512, scaled by the actual input squares and affine spans. It does not use qbar*(1-qbar)/512, the endpoint probability, or an unconditional1/T convergence law. Random path-to-path mean error remains in the conditional-mean error term. Native segment means are accumulated before any nonlinearity.\n',
        '## Clock and cost accounting\n',
        '| Age | Policy | Calibration events including initial | Active sweeps | Idle sweeps | Query-end clock sweeps |\n|---:|---|---:|---:|---:|---:|'])
    for r in costs:text.append(f"| {r['age']} | {r['policy']} | {r['counterfactual_calibration_events']} | {r['counterfactual_active_sweeps']:,} | {r['counterfactual_idle_sweeps']:,} | {r['query_end_clock_sweeps']:,} |")
    text.extend([f"\nThe actual executed study has {report['native_calibration_events']} calibration events ({report['native_probe_windows']} probe windows,{report['native_count_segments']} native64-sweep count segments) and {report['native_query_windows']} inference windows ({report['native_linear_segments']} native linear segments). It executes {report['executed_calibration_free_node_evaluations']:,} calibration and {report['executed_query_free_node_evaluations']:,} query free-node evaluations. Initial calibrations are shared as source artifacts rather than redundantly resampled for every counterfactual query; per-policy accounting includes their cost.\n",
        f"The independent audit reconstructs every one of {audit['independently_reconstructed_segment_field_hashes']:,} active field hashes from a separately streamed Brownian path, every count-derived command, query mean/variance/error and active-plus-idle clock identity. CPU/CUDA tests separately match changed-field segment counts and linear means to explicit public-factor native trajectories.\n",
        'Corrected codes, local ranges, endpoint modes, count buffers and runtime arrays retain their earlier costs; storing one stochastic state is not one total storage bit per weight. The full Brownian path is a transient simulation truth buffer, not an implemented on-device controller. No device sampling frequency, drift-per-second, physical area or energy is inferred here.\n',
        'The broad research goal remains active for calibration-aware full-network quality, stronger independent-checkpoint native training, broader conversion and physical device characterization. A suitable drift-aware controller or refresh policy would require validation of its quality/cost target, not just this retention curve.\n',
        '## Reproduction\n',
        'Run from the repository root with the recorded CUDA/model environments and old development artifacts. Matching durable cases are verified and resumed; a changed protocol is refused in the same output directory.\n',
        '```bash\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.drift_retention_operator\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_drift_retention_operator\n.venv-thrml/bin/python -m research.thrml_spw.drift_retention_operator_report\n```\n'])
    rendered='\n'.join(text)+'\n'
    for a,b in {'SD0.04':'SD 0.04','per512':'per 512','age32':'age 32','has2,359':'has 2,359','mean1':'mean 1','log-SD0.2':'log-SD 0.2','SD0.2':'SD 0.2','is512':'is 512','each64':'each 64','uses4096':'uses 4096','device:2048':'device: 2048','code96':'code 96','by2048':'by 2048','code159':'code 159','from32':'from 32','guard[':'guard [','segments0':'segments 0','segments64':'segments 64','or256':'or 256','segments64,128 or320':'segments 64, 128 or 320','ages0,8 or32':'ages 0, 8 or 32','Age0':'Age 0','unconditional1/T':'unconditional 1/T',',5760':', 5760','native64':'native 64'}.items():rendered=rendered.replace(a,b)
    Path('results/thrml_spw/DRIFT_RETENTION_OPERATOR_REPORT.md').write_text(rendered)
    print(json.dumps({'query_summaries':summaries,'paired_comparisons':comparisons,'calibration_summaries':cal_summary},indent=2))


if __name__=='__main__':main()
