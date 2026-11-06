"""Report calibrated spatial/temporal budgets without hiding physical parallelism."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

OUT=Path('results/thrml_spw/calibrated_matched_operator')


def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed']
    report=json.loads((OUT/'REPORT.json').read_text())
    with (OUT/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    methods=['uncalibrated','oracle','cal256','cal1024','cal4096'];summaries=[]
    for k in [1,4]:
        for method in methods:
            group=[r for r in rows if int(r['K'])==k and r['method']==method];assert len(group)==10
            r={'K':k,'method':method,'P':int(group[0]['P']),'probe_samples_per_device_per_setting':int(group[0]['probe_samples_per_device_per_setting']),
                'inference_samples_per_device':512//k,'native_runs':10}
            for field in ['observed_total_mse','conditional_bias_floor','predicted_sampling_mse','observed_sampling_mse','bound_clipped_commands','gain_guard_low','gain_guard_high']:
                values=np.array([float(g[field]) for g in group]);r[field]=float(values.mean())
                if field in ['observed_total_mse','conditional_bias_floor']:r[field+'_approx95_margin']=float(1.96*values.std(ddof=1)/np.sqrt(10))
            if r['P']:
                r['gain_estimation_rmse']=float(np.mean([float(g['estimated_gain_rmse']) for g in group]))
                r['offset_estimation_rmse']=float(np.mean([float(g['estimated_offset_rmse']) for g in group]))
                r['probe_frequency_mse_over_prediction']=float(np.mean([float(g['calibration_frequency_mse'])/float(g['calibration_predicted_frequency_mse']) for g in group]))
            summaries.append(r)
    with (OUT/'summary.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(key for r in summaries for key in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summaries)
    comparisons=[]
    for method in methods:
        differences=[]
        for seed in range(10):
            vals=[float(next(r['observed_total_mse'] for r in rows if int(r['K'])==k and int(r['seed'])==seed and r['method']==method)) for k in [1,4]]
            differences.append(vals[1]-vals[0])
        delta=np.array(differences);margin=float(1.96*delta.std(ddof=1)/np.sqrt(10))
        a,b=[next(r for r in summaries if r['K']==k and r['method']==method) for k in [1,4]]
        comparisons.append({'method':method,'P':a['P'],'K4_minus_K1_MSE':float(delta.mean()),'approximate_paired_95_margin':margin,
            'ratio_K4_to_K1_MSE':b['observed_total_mse']/a['observed_total_mse'],
            'unadjusted_interval_includes_zero':bool(abs(delta.mean())<=margin)})
    (OUT/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');xx=[256,1024,4096]
    for ax,field,title in zip(axes,['observed_total_mse','conditional_bias_floor'],['Native output error at equal draws','Conditional fixed mean-error floor']):
        for k,color in [(1,'tab:blue'),(4,'tab:orange')]:
            group=[next(r for r in summaries if r['K']==k and r['P']==p) for p in xx]
            ax.errorbar(xx,[r[field] for r in group],yerr=[r[field+'_approx95_margin'] for r in group],fmt='o-',color=color,label=f'K{k}, T={512//k}')
            oracle=next(r for r in summaries if r['K']==k and r['method']=='oracle')
            ax.axhline(oracle[field],ls=':',color=color,alpha=.7,label=f'K{k} known-parameter diagnostic')
        ax.set(xscale='log',yscale='log',xlabel='Total probe draws / unique weight',ylabel='Output MSE',title=title)
        ax.set_xticks(xx,['256','1024','4096']);ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig('results/thrml_spw/calibrated_matched_operator_evidence.png',dpi=180);plt.close(fig)
    ledger=[]
    for k in [1,4]:
        g=next(r for r in rows if int(r['K'])==k and r['method']=='cal4096')
        ledger.append({'K':k,'P':4096,'logical_nodes':int(g['logical_nodes']),'free_nodes':int(g['free_nodes']),
            'calibration_step_depth_two_probes':4096//k,'inference_step_depth':512//k,
            **{field:int(g[field]) for field in ['calibration_free_node_evaluations','inference_free_node_evaluations','inference_logical_reads','inference_binary_products','stored_command_bytes','base_float32_field_bytes','calibration_count_bytes','replica_output_accumulator_bytes']},
            'unique_range_metadata_bits':report['controls'][str(k)]['unique_range_metadata_bits']})
    (OUT/'resource_ledger.json').write_text(json.dumps({'rows':ledger,'scope':'Selected actual software arrays and logical counters, not total runtime memory, chip area, physical transitions, energy or measured latency. Group ranges shared in software; physical sharing circuitry not modeled.'},indent=2)+'\n')
    one,four=[next(r for r in summaries if r['K']==k and r['method']=='cal4096') for k in [1,4]]
    last=comparisons[-1]
    text=['# Calibrated single-p-bit versus spatial replicas at matched draws\n',
        f"At4096 total calibration draws and512 inference draws per unique weight, native K1/T512 output MSE is {one['observed_total_mse']:.6g}, versus K4/T128 {four['observed_total_mse']:.6g}. The paired K4-minus-K1 difference is {last['K4_minus_K1_MSE']:+.6g} with approximate margin {last['approximate_paired_95_margin']:.3g}; its unadjusted interval includes zero: {last['unadjusted_interval_includes_zero']}. K4 uses four times the logical nodes and one quarter the probe/inference sampling depth. These are distinct resources, not a fourfold systems improvement.\n",
        '## Fixed design and equal opportunities\n',
        'One 2,359,296-weight DistilGPT-2 projection executes through official THRML with K1 or K4 independent explicit p-bits per normalized weight. Both use the same eight old development contexts, nominal packed control target and local affine ranges. Gains are positive lognormal mean1, log-SD0.2; additive field offsets have SD0.2. All device parameters are fixed through calibration/inference. The first device in K4 is paired with K1 by seed, while the other three have independent parameters. These are synthetic dimensionless device errors, not measured Extropic specifications.\n',
        'Both measured designs use exactly the same two-probe estimator: finite eight-bit codes96/159, Jeffreys-smoothed logit line fit, declared gain guard[0.25,4], and nearest bounded eight-bit inverse command in[-4,4]. No true parameters or task labels enter the measured estimator. Exact endpoint clamps remain unperturbed. The K4 probe observes every distinct free replica rather than reusing a single-device trace.\n',
        'For total probe budget P per unique weight, each simulated device receives P/K probe samples, split equally across two settings. Thus K1 and K4 execute the same total free-node calibration updates. Inference uses T=512/K, matching total free updates, logical reads and nominal binary-product terms. Inputs are fixed before linear sampling, and outputs are averaged before any later nonlinearity.\n',
        'The inherited control-object mechanism label describes the intended nominal replica target. Actual realized marginals differ by the recorded per-device fields after gain/offset and finite calibration; independent audits recompute those probabilities rather than assuming identical realized marginals.\n',
        'All three budgets256/1024/4096 and all ten joint device/sampling realizations are retained without selection or significance-based stopping. Uncalibrated and known-parameter inverse controls are reported separately. The oracle obtains true simulated parameters for a diagnostic; its zero measured probe cost is not an implementable comparator.\n',
        '## All native results\n',
        '| K | Method | Total probe draws P | Reads/device/setting | Inference T | Output MSE | Approx. 95% margin | Conditional bias floor |\n|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:text.append(f"| {r['K']} | {r['method']} | {r['P']} | {r['probe_samples_per_device_per_setting']} | {r['inference_samples_per_device']} | {r['observed_total_mse']:.6g} | {r['observed_total_mse_approx95_margin']:.3g} | {r['conditional_bias_floor']:.6g} |")
    text.extend(['\n| Method | P | K4 minus K1 MSE | Approx. paired 95% margin | K4 / K1 mean MSE | Unadjusted interval includes zero |\n|---|---:|---:|---:|---:|---|'])
    for r in comparisons:text.append(f"| {r['method']} | {r['P']} | {r['K4_minus_K1_MSE']:+.6g} | {r['approximate_paired_95_margin']:.3g} | {r['ratio_K4_to_K1_MSE']:.4f} | {r['unadjusted_interval_includes_zero']} |")
    text.extend(['\nNormal ten-realization margins summarize joint fixed-device and sampling variation on one old input tensor. The five contrasts are not multiplicity-adjusted formal coverage or equivalence tests. Similar total-draw budgets do not ensure identical learned quality, calibrated conditional means, or estimator bias. Four noisy parameter fits average differently from one better-measured fit. Other calibrators, probe locations, allocation policies or shared device errors could change the result.\n',
        '## Resource ledger at P4096\n',
        '| Resource | K1 | K4 |\n|---|---:|---:|'])
    labels={'logical_nodes':'Logical p-bit nodes','free_nodes':'Free nodes','calibration_step_depth_two_probes':'Calibration step depth, both probes','inference_step_depth':'Inference step depth','calibration_free_node_evaluations':'Calibration free-node updates','inference_free_node_evaluations':'Inference free-node updates','inference_logical_reads':'Inference logical readouts','inference_binary_products':'Nominal binary-product terms','stored_command_bytes':'Corrected uint8 command bytes','base_float32_field_bytes':'Base float32 field bytes','calibration_count_bytes':'Two int32 count arrays, bytes','replica_output_accumulator_bytes':'Replica output accumulators, bytes','unique_range_metadata_bits':'Shared unique group-range bits'}
    for key,label in labels.items():text.append(f"| {label} | {ledger[0][key]:,} | {ledger[1][key]:,} |")
    text.extend([f"\nThe executed study has100 native inference and120 calibration-probe runs, all independently audited from packed bits, device/field hashes, saved counts, independently fitted commands and raw outputs. Its calibration draws total {report['total_calibration_free_node_evaluations']:,} free-node updates, and inference draws total {report['total_inference_free_node_evaluations']:,}. The runtime has additional program, probability, index, expanded-range and analysis buffers beyond the selected array ledger. This is not one total storage bit per weight.\n",
        'At a reuse count R, either design has free-draw overhead1+P/(512*R) relative to its own inference draw count. The controller population, programming, per-step read width, accumulation and depth differ. Neither equal draws nor ideal step depth is a physical energy/latency measurement. Stable parameter reuse and independent device errors remain assumptions.\n',
        'This is a spatial-replica control, not positional multi-bit encoding or an established official Extropic learned-weight reference. It establishes an operator-level calibrated space/time comparison under a stated simulator model. It does not establish full-model accuracy, physical density, hardware energy superiority or any contradiction of Extropic hardware. The broad goal remains active for drift/recalibration, stronger training, broader model conversion and physical characterization.\n',
        '## Reproduction\n',
        'From the repository root with the recorded environments and old development artifacts. Existing matching cases are hash-verified and resumed; a changed design is refused in the same evidence directory.\n',
        '```bash\n.venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.calibrated_matched_operator\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_calibrated_matched_operator\n.venv-thrml/bin/python -m research.thrml_spw.calibrated_matched_operator_report\n```\n'])
    rendered='\n'.join(text)+'\n'
    for a,b in {'At4096':'At 4096','and512':'and 512','mean1':'mean 1','log-SD0.2':'log-SD 0.2','SD0.2':'SD 0.2','codes96':'codes 96','guard[':'guard [','in[-4':'in [-4','budgets256':'budgets 256','P4096':'P=4096','has100':'has 100','and120':'and 120','overhead1+':'overhead 1+'}.items():rendered=rendered.replace(a,b)
    Path('results/thrml_spw/CALIBRATED_MATCHED_OPERATOR_REPORT.md').write_text(rendered)
    print(json.dumps({'summaries':summaries,'paired_comparisons':comparisons,'resources':ledger},indent=2))


if __name__=='__main__':main()
