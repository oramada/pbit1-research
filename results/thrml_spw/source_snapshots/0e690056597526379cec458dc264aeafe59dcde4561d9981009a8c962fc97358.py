"""Full-network calibration accuracy with model-level uncertainty and controls."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/thrml_spw/calibrated_mnist')
CONDITIONS=['ideal8','uncalibrated','cal1024','cal4096','oracle']


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def margin(values):return float(1.96*np.std(values,ddof=1)/np.sqrt(len(values)))


def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed']
    replay=json.loads((OUT/'public_path_audit.json').read_text());assert replay['passed']
    report=json.loads((OUT/'REPORT.json').read_text());rows=read(OUT/'native_metrics.csv');base=read(OUT/'baseline_metrics.csv');means=read(OUT/'mean_network_metrics.csv')
    fp=np.array([float(next(r['accuracy'] for r in base if int(r['model'])==m and r['condition']=='FP32')) for m in range(5)])
    baselines=[]
    for condition in ['FP32','UINT4_per_layer_scale','UINT8_per_layer_scale','programmed_mean_ideal8']:
        vals=np.array([float(next(r['accuracy'] for r in base if int(r['model'])==m and r['condition']==condition)) for m in range(5)])
        baselines.append({'condition':condition,'accuracy_mean':float(vals.mean()),'model_normal95_margin':margin(vals),'paired_difference_vs_FP32':float(np.mean(vals-fp))})
    summaries=[];model_rows=[];vectors={}
    for condition in CONDITIONS:
        for t in [64,256,1024]:
            a=[];ce=[]
            for model in range(5):
                group=[r for r in rows if int(r['model'])==model and r['condition']==condition and int(r['T'])==t];assert len(group)==10
                acc=float(np.mean([float(r['accuracy']) for r in group]));loss=float(np.mean([float(r['cross_entropy']) for r in group]));a.append(acc);ce.append(loss)
                model_rows.append({'model':model,'condition':condition,'T':t,'device_sampling_realizations':10,'accuracy_mean':acc,'cross_entropy_mean':loss,'difference_vs_FP32':acc-fp[model]})
            a=np.array(a);vectors[condition,t]=a;delta=a-fp
            summaries.append({'condition':condition,'T':t,'trained_models':5,'device_sampling_realizations_per_model':10,'accuracy_mean':float(a.mean()),
                'model_normal95_margin':margin(a),'paired_difference_vs_FP32':float(delta.mean()),'paired_model_normal95_margin':margin(delta),'cross_entropy_mean':float(np.mean(ce))})
    for path,rr in [('baseline_summary.csv',baselines),('native_summary.csv',summaries),('by_model_summary.csv',model_rows)]:
        with (OUT/path).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    comparisons=[]
    for t in [64,256,1024]:
        for c0,c1 in [('uncalibrated','cal1024'),('uncalibrated','cal4096'),('ideal8','cal4096'),('oracle','cal4096')]:
            delta=vectors[c1,t]-vectors[c0,t]
            comparisons.append({'T':t,'reference':c0,'condition':c1,'paired_accuracy_difference':float(delta.mean()),'paired_model_normal95_margin':margin(delta)})
    (OUT/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    mean_summary=[]
    for condition in CONDITIONS:
        vals=np.array([np.mean([float(r['accuracy']) for r in means if int(r['model'])==m and r['condition']==condition]) for m in range(5)])
        mean_summary.append({'condition':condition,'accuracy_mean':float(vals.mean()),'model_normal95_margin':margin(vals),'paired_difference_vs_FP32':float(np.mean(vals-fp))})
    with (OUT/'mean_network_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(mean_summary[0]));w.writeheader();w.writerows(mean_summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    colors=['tab:gray','tab:red','tab:blue','tab:orange','tab:purple']
    labels=['Ideal finite8','Uncalibrated','Measured P1024','Measured P4096','Known-parameter diagnostic']
    for condition,color,label in zip(CONDITIONS,colors,labels):
        g=[next(r for r in summaries if r['condition']==condition and r['T']==t) for t in [64,256,1024]]
        axes[0].errorbar([64,256,1024],[100*r['accuracy_mean'] for r in g],yerr=[100*r['model_normal95_margin'] for r in g],fmt='o-',color=color,label=label)
    axes[0].axhline(100*fp.mean(),ls='--',color='black',label='Same-model FP32')
    uq4=next(r for r in baselines if r['condition']=='UINT4_per_layer_scale');axes[0].axhline(100*uq4['accuracy_mean'],ls=':',color='tab:green',label='UINT4 numerical control')
    axes[0].set(xscale='log',xticks=[64,256,1024],xticklabels=['64','256','1024'],xlabel='Samples per layer',ylabel='MNIST accuracy (%)',title='Complete direct p=w networks');axes[0].legend(fontsize=7)
    pos=np.arange(5);width=.36
    axes[1].bar(pos-width/2,[100*r['accuracy_mean'] for r in mean_summary],width,yerr=[100*r['model_normal95_margin'] for r in mean_summary],label='Mean-weight diagnostic',color='tab:gray',alpha=.7)
    high=[next(r for r in summaries if r['condition']==c and r['T']==1024) for c in CONDITIONS]
    axes[1].bar(pos+width/2,[100*r['accuracy_mean'] for r in high],width,yerr=[100*r['model_normal95_margin'] for r in high],label='Native T1024',color=colors)
    axes[1].axhline(100*fp.mean(),ls='--',color='black');axes[1].set(xticks=pos,xticklabels=['Ideal8','Uncal.','P1024','P4096','Oracle'],ylabel='Accuracy (%)',title='Programmed means and sampled execution');axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig('results/thrml_spw/calibrated_mnist_evidence.png',dpi=180);plt.close(fig)
    primary=next(r for r in comparisons if r['T']==256 and r['reference']=='uncalibrated' and r['condition']=='cal4096')
    ideal,uncal,cal,highcal=[next(r for r in summaries if r['condition']==c and r['T']==t) for c,t in [('ideal8',256),('uncalibrated',256),('cal4096',256),('cal4096',1024)]]
    resources={'per_model':report['model_controls'],'native_evaluations':750,'calibration_probe_runs':200,
        'test_images_per_evaluation':10000,'batch_size':32,'batches_including_last_partial':313,
        'calibration_free_evaluations_executed':report['executed_calibration_free_evaluations'],'inference_free_evaluations_executed':report['executed_inference_free_evaluations'],
        'hardware_population_interpretation':'2368 unique matrix-weight coordinates per model, reused across calibration and inference modes/batches. Software allocates a joint calibration graph, separate layer graphs and vmap chain copies; do not call its RAM allocation2368 bits.',
        'scope':'Active sampler evaluations and selected data/control counts, not physical transitions, all runtime memory, background switching energy, hardware parallelism or measured latency.'}
    (OUT/'resource_ledger.json').write_text(json.dumps(resources,indent=2)+'\n')
    text=['# Full-network measured calibration for direct p=w weights\n',
        f"Across five trained MNIST models and ten joint device/sampling realizations each, native T256 accuracy is {100*cal['accuracy_mean']:.3f}% with measured P4096 calibration, versus {100*uncal['accuracy_mean']:.3f}% uncalibrated and {100*ideal['accuracy_mean']:.3f}% ideal finite controls. Same-model FP32 averages {100*fp.mean():.3f}%. The predeclared primary paired calibration improvement is {100*primary['paired_accuracy_difference']:+.3f} percentage points with approximate model-level normal margin {100*primary['paired_model_normal95_margin']:.3f} points. At predeclared T1024, calibrated accuracy is {100*highcal['accuracy_mean']:.3f}% ({100*highcal['paired_difference_vs_FP32']:+.3f} points versus FP32).\n",
        '## Protocol and scope\n',
        'Five previously trained64->32->10 positive-weight MNIST checkpoints are frozen. Every matrix parameter is directly its intended p-bit probability p=w; there are2368 logical weight coordinates and42 deterministic neural biases per model. No signed affine range conversion is used. Exact endpoints stay clamped. Nominal free-node fields are rounded/clipped to eight-bit codes in[-4,4].\n',
        'Static synthetic device fields are b_real=float32(a*b_command+d), with positive lognormal gain mean1/log-SD0.2 and offset SD0.2. These are not measured Extropic specifications. The two measured budgets use1024 or4096 native probe samples per free device, split across actual codes96/159. A smoothed-logit line fit with declared gain guard[0.25,4] emits finite eight-bit corrected commands. It receives no true device parameters or task labels. Known-parameter inversion is a diagnostic, not an implementable oracle or optimum task solution.\n',
        'Each neural layer accumulates actual THRML sampled linear outputs before its activation; ReLU is applied once after the first-layer mean. All ten thousand test examples are scored in313 batches of32, with16 discarded zero-padding rows in the last batch. Independent batch chains are vectorized for simulation; weight samples are shared within a32-image batch. Gains/offsets remain fixed across all batches in this study. Clocked drift is not introduced here.\n',
        'Common inference random numbers pair conditions under the same model, device realization and T; differing fields produce differing bit states. Each of the five trained models has ten joint device/sampling seeds, not ten new trained checkpoints or separately identified hardware/Monte Carlo variance components. The three T budgets64/256/1024 and every condition are fixed before scoring and all750 runs are retained. There is no outcome-selected sample budget or success gate.\n',
        'Images were held out from training, but the same standard test set was used in prior experiments. This is a fixed new nonideal-condition study, not a fresh test reservation or broad dataset confirmation. Positive-weight model accuracy is compared with its own deterministic baseline; it is not a claim to competitive MNIST architectures.\n',
        '## Conventional controls\n',
        '| Control | Accuracy (%) | Approx. model-level 95% margin (pp) | Difference vs FP32 (pp) |\n|---|---:|---:|---:|']
    for r in baselines:text.append(f"| {r['condition']} | {100*r['accuracy_mean']:.3f} | {100*r['model_normal95_margin']:.3f} | {100*r['paired_difference_vs_FP32']:+.3f} |")
    text.extend(['\nUINT4/UINT8 are per-layer-scale numerical dequantization controls, not optimized integer hardware kernels or probabilistic encodings. Programmed-mean diagnostics run a deterministic network with realized probabilities as weights. Because ReLU is nonlinear, that network is not the exact expectation of sampled finite-T network logits.\n',
        '## All native full-test outcomes\n',
        '| Condition | T/layer | Accuracy (%) | Approx. model-level 95% margin (pp) | Paired difference vs FP32 (pp) | Paired margin (pp) | CE |\n|---|---:|---:|---:|---:|---:|---:|'])
    for r in summaries:text.append(f"| {r['condition']} | {r['T']} | {100*r['accuracy_mean']:.3f} | {100*r['model_normal95_margin']:.3f} | {100*r['paired_difference_vs_FP32']:+.3f} | {100*r['paired_model_normal95_margin']:.3f} | {r['cross_entropy_mean']:.5f} |")
    text.extend(['\nUncertainty uses1.96*SD/sqrt(5) across five model means, after averaging ten joint device/sampling realizations per model; paired margins subtract each model baseline or matched condition first. These normal five-model intervals and multiple displayed comparisons are descriptive, not formal coverage or equivalence guarantees. Primary comparison is P4096 versus uncalibrated at T256; all others are predeclared secondary outcomes.\n',
        '## Verification and resource costs\n',
        f"All750 full-test runs and200 native calibration-probe runs passed independent raw audits: source/data/model hashes, actual counts, independently reconstructed corrected codes/fields, expected-weight controls, all reported scores and logical resource counters. A second CUDA audit reconstructs every example in both layers from separately constructed public THRML factors and integer state frequencies, independently of the production output-sum observer. Maximum scaled first-layer/logit errors are {replay['max_scaled_first_preactivation_error']:.3g}/{replay['max_scaled_logit_error']:.3g}, below2e-5 F32 summation tolerance. CPU/CUDA tests also detect the wrong activation ordering.\n",
        f"Executed calibration free-node evaluations total {report['executed_calibration_free_evaluations']:,}; native inference evaluations total {report['executed_inference_free_evaluations']:,}. Each model/device pair calibrates once per tested P and reuses those codes across all test batches and T settings. Those shared artifacts are not repeatedly charged as new calibration executions. An inference evaluation uses313*T*n_free updates; the logical weight read and binary-product counts retain batch sharing and padded rows explicitly. These are active sampler evaluations, not measured physical transitions or background idle switching energy.\n",
        'One deployed condition needs one uint8 command per free weight plus endpoint modes, deterministic neural biases, target/programming information and calibration buffers; these are not one total storage bit per learned parameter. Gain/offset truth arrays are simulation state. The joint calibration graph and the two inference graphs represent the same device coordinates in different modes; their metadata and vmap chains are duplicated in software, not a physical population claim. Resource ledgers retain those distinctions, and no physical energy, density or latency superiority follows from this software result.\n',
        'The broader goal remains active for clocked/nonstationary full-network quality, independent-checkpoint native training, larger converted fractions and real device characterization. Existing Transformer findings, failures and calibration-retention limits are unchanged.\n',
        '## Reproduction\n',
        'Run from the repository root with the recorded environments and original positive-MNIST exports. Matching durable cases are hash-verified and resumed; a changed protocol is refused in the same output directory.\n',
        '```bash\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.calibrated_mnist\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.calibrated_mnist_path_audit\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_calibrated_mnist\n.venv-thrml/bin/python -m research.thrml_spw.calibrated_mnist_report\n```\n'])
    rendered='\n'.join(text)+'\n'
    for a,b in {'trained64':'trained 64','are2368':'are 2368','and42':'and 42','in[-4':'in [-4','mean1':'mean 1','log-SD0.2':'log-SD 0.2','SD0.2':'SD 0.2','use1024':'use 1024','or4096':'or 4096','codes96':'codes 96','guard[':'guard [','in313':'in 313','of32':'of 32','with16':'with 16','a32':'a 32','budgets64':'budgets 64','all750':'all 750','uses1.96':'uses 1.96','All750':'All 750','and200':'and 200','below2e':'below 2e','uses313':'uses 313'}.items():rendered=rendered.replace(a,b)
    Path('results/thrml_spw/CALIBRATED_MNIST_REPORT.md').write_text(rendered)
    print(json.dumps({'baseline_summary':baselines,'native_summary':summaries,'primary_comparison':primary},indent=2))


if __name__=='__main__':main()
