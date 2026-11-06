"""Report endpoint leakage, range/precision and full-network task controls."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/thrml_spw/finite_endpoint_mnist')
ENC=['clamped_B4_b8','free_B4_b8','free_B8_b8','free_B8_b10']
COND=['ideal','uncalibrated','cal4096','oracle']


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def margin(x):return float(1.96*np.std(x,ddof=1)/np.sqrt(len(x)))


def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed']
    replay=json.loads((OUT/'public_path_audit.json').read_text());assert replay['passed']
    result=json.loads((OUT/'REPORT.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    rows=read(OUT/'native_metrics.csv');baseline=read(OUT/'baseline_metrics.csv');means=read(OUT/'mean_network_metrics.csv')
    fp=np.array([float(next(r['accuracy'] for r in baseline if int(r['model'])==m and r['condition']=='FP32')) for m in range(5)])
    uq4=np.array([float(next(r['accuracy'] for r in baseline if int(r['model'])==m and r['condition']=='UINT4')) for m in range(5)])
    summaries=[];vectors={}
    for en in ENC:
        for condition in COND:
            for t in [256,1024]:
                aa=np.array([np.mean([float(r['accuracy']) for r in rows if int(r['model'])==m and r['encoding']==en and r['condition']==condition and int(r['T'])==t]) for m in range(5)])
                loss=np.array([np.mean([float(r['cross_entropy']) for r in rows if int(r['model'])==m and r['encoding']==en and r['condition']==condition and int(r['T'])==t]) for m in range(5)])
                vectors[en,condition,t]=aa;delta=aa-fp
                summaries.append({'encoding':en,'condition':condition,'T':t,'trained_models':5,'device_sampling_realizations_per_model':10,
                    'accuracy_mean':float(aa.mean()),'model_normal95_margin':margin(aa),'paired_difference_vs_FP32':float(delta.mean()),'paired_model_normal95_margin':margin(delta),'cross_entropy':float(loss.mean())})
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    comparisons=[]
    for en in ENC[1:]:
        for condition in COND:
            for t in [256,1024]:
                delta=vectors[en,condition,t]-vectors[ENC[0],condition,t]
                comparisons.append({'encoding':en,'reference':ENC[0],'condition':condition,'T':t,'paired_accuracy_difference':float(delta.mean()),'paired_model_normal95_margin':margin(delta)})
    (OUT/'endpoint_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    quant=[]
    for en in ENC:
        for t in [256,1024]:
            delta=vectors[en,'cal4096',t]-uq4
            quant.append({'encoding':en,'condition':'cal4096','T':t,'reference':'UINT4','paired_accuracy_difference':float(delta.mean()),'paired_model_normal95_margin':margin(delta)})
    (OUT/'quantization_comparisons.json').write_text(json.dumps(quant,indent=2)+'\n')
    mean_summary=[]
    for en in ENC:
        for c in COND:
            aa=np.array([np.mean([float(r['accuracy']) for r in means if int(r['model'])==m and r['encoding']==en and r['condition']==c]) for m in range(5)])
            mean_summary.append({'encoding':en,'condition':c,'accuracy_mean':float(aa.mean()),'model_normal95_margin':margin(aa)})
    with (OUT/'mean_network_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(mean_summary[0]));w.writeheader();w.writerows(mean_summary)
    resources=[]
    for e in p['encodings']:
        en=e['name'];ctrl=[result['model_controls'][str(m)][en] for m in range(5)]
        row={'encoding':en,'bound':e['bound'],'bits':e['bits'],'clamp_endpoints':e['clamp_endpoints'],
            'ideal_endpoint_probability_floor':ctrl[0]['ideal_endpoint_probability_floor'],
            'nominal_bias_step':2*e['bound']/((1<<e['bits'])-1)}
        for key in ['logical_nodes','free_nodes','clamped_nodes','logical_command_payload_bits','actual_command_array_bytes','endpoint_mode_bits_if_packed','actual_base_field_bytes']:
            row[key+'_mean']=float(np.mean([c[key] for c in ctrl]))
        row['active_update_ratio_to_clamped']=row['free_nodes_mean']/float(np.mean([result['model_controls'][str(m)][ENC[0]]['free_nodes'] for m in range(5)]))
        resources.append(row)
    (OUT/'resource_ledger.json').write_text(json.dumps({'encodings':resources,
        'production_calibration_free_evaluations':result['executed_calibration_free_evaluations'],
        'production_inference_free_evaluations':result['executed_inference_free_evaluations'],
        'independent_replay_free_evaluations':result['executed_inference_free_evaluations'],
        'scope':'Selected control arrays and logical payload/counters; counts averaged over five models. Ten-bit codes use actual uint16 containers, not achieved ten-bit packing. No total runtime compression or physical area/energy/latency claim.'},indent=2)+'\n')
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');labels=['Clamp\nB4/8b','Free\nB4/8b','Free\nB8/8b','Free\nB8/10b'];xx=np.arange(4)
    for t,color in [(256,'tab:blue'),(1024,'tab:orange')]:
        g=[next(r for r in summaries if r['encoding']==en and r['condition']=='cal4096' and r['T']==t) for en in ENC]
        ax[0,0].errorbar(xx,[100*r['accuracy_mean'] for r in g],yerr=[100*r['model_normal95_margin'] for r in g],fmt='o-',color=color,label=f'Native T{t}')
    ax[0,0].axhline(100*fp.mean(),ls='--',color='black',label='Same-model FP32');ax[0,0].axhline(100*uq4.mean(),ls=':',color='tab:green',label='UINT4 numerical control')
    ax[0,0].set(title='Measured calibration P4096',ylabel='Accuracy (%)',xticks=xx,xticklabels=labels);ax[0,0].legend(fontsize=7)
    for c,color in [('ideal','tab:gray'),('cal4096','tab:orange')]:
        g=[next(r for r in mean_summary if r['encoding']==en and r['condition']==c) for en in ENC]
        ax[0,1].errorbar(xx,[100*r['accuracy_mean'] for r in g],yerr=[100*r['model_normal95_margin'] for r in g],fmt='o-',color=color,label=c+' mean-weight diagnostic')
    ax[0,1].axhline(100*fp.mean(),ls='--',color='black');ax[0,1].set(title='Programmed-mean diagnostics',ylabel='Accuracy (%)',xticks=xx,xticklabels=labels);ax[0,1].legend(fontsize=7)
    g=[next(r for r in comparisons if r['encoding']==en and r['condition']=='cal4096' and r['T']==1024) for en in ENC[1:]]
    ax[1,0].errorbar(np.arange(3),[100*r['paired_accuracy_difference'] for r in g],yerr=[100*r['paired_model_normal95_margin'] for r in g],fmt='o',capsize=3)
    ax[1,0].axhline(0,ls='--',color='black');ax[1,0].set(title='Calibrated T1024: paired endpoint contrasts',ylabel='Difference vs clamped control (pp)',xticks=np.arange(3),xticklabels=labels[1:])
    free=[r['free_nodes_mean'] for r in resources];clamp=[r['clamped_nodes_mean'] for r in resources]
    ax[1,1].bar(xx,free,label='Free weight nodes');ax[1,1].bar(xx,clamp,bottom=free,label='Exact held endpoint nodes',color='tab:gray')
    ax[1,1].set(title='All variants keep 2,368 weight coordinates',ylabel='Mean modeled node count',xticks=xx,xticklabels=labels);ax[1,1].legend(fontsize=7)
    for a in ax.ravel():a.grid(axis='y',alpha=.2)
    fig.savefig('results/thrml_spw/finite_endpoint_mnist_evidence.png',dpi=180);plt.close(fig)
    primary=next(r for r in comparisons if r['encoding']=='free_B4_b8' and r['condition']=='cal4096' and r['T']==1024)
    control,free4,free8,free10=[next(r for r in summaries if r['encoding']==en and r['condition']=='cal4096' and r['T']==1024) for en in ENC]
    text=['# Finite endpoint probabilities without exact clamp modes\n',
        f"Measured-calibrated T1024 accuracy is {100*free4['accuracy_mean']:.3f}% with every weight a free p-bit at eight-bit bias range[-4,4], versus {100*control['accuracy_mean']:.3f}% with the paired clamped control. The primary paired difference is {100*primary['paired_accuracy_difference']:+.3f} percentage points with approximate five-model normal margin {100*primary['paired_model_normal95_margin']:.3f}. Wider-range free-node variants score {100*free8['accuracy_mean']:.3f}% (eight bits) and {100*free10['accuracy_mean']:.3f}% (ten bits). FP32 is {100*fp.mean():.3f}%. These are all predeclared variants; none is selected as an optimized hardware design.\n",
        '## What changes\n',
        'A finite-bias free Bernoulli element cannot have exactly zero or one stationary probability. The previous inference experiments explicitly held endpoint nodes. Here the three free-node encodings instead clip probability targets to epsilon_B and1-epsilon_B, where epsilon_B=sigmoid(-2B). They approximate endpoint weights rather than exactly represent zero/one. Every2368 matrix-weight coordinate then has a free native THRML SpinNode, with zero clamp blocks. Original learned weights and their FP32 task baseline remain unchanged.\n',
        'The four controller/endpoint configurations are fixed: clamped B4/b8, free B4/b8, free B8/b8 and free B8/b10. Increasing B enlarges reachable bias range but coarsens the grid at fixed bit count; ten bits compensate that coarsening at extra code precision/storage. Ideal epsilon_B is not a guarantee on realized probabilities under gain and offset. All free-node fields and codes are finite.\n',
        'Five frozen positive-weight64->32->10 models, ten paired full-coordinate device/sampling realizations each and all10000 existing MNIST test images are evaluated at T256/T1024. Static synthetic device gain is positive lognormal mean1/log-SD0.2 and field offset SD0.2. True parameters align by original2368-coordinate index across encodings, including cells hidden by clamps in the control. Different free-state dimensions mean identical sampled trajectories across geometries are not assumed.\n',
        'Each measured fit uses4096 native probe draws per free device, split over the nearest finite codes to-1,+1. A smoothed-logit gain/offset fit with predeclared[0.25,4] gain guard emits inverse commands on the tested grid; no true parameters or labels enter it. Neural biases remain deterministic, and sampled linear outputs accumulate before ReLU. Vectorized dataset chains and separate calibration/layer graph objects are software copies of device modes, not claims to hardware parallel capacity.\n',
        'All1600 native full-test cases and400 native probe windows are retained. Test images were already used in prior studies; this is not fresh held-out data or training. No budget, encoding, gate or success threshold is selected from test outcomes. Primary comparison is free B4/b8 versus its new paired clamped control at calibrated T1024; all other contrasts are predeclared secondary comparisons.\n',
        '## All full-test results\n',
        '| Encoding | Condition | T/layer | Accuracy (%) | Model normal margin (pp) | Paired FP32 difference (pp) | Paired margin (pp) |\n|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:text.append(f"| {r['encoding']} | {r['condition']} | {r['T']} | {100*r['accuracy_mean']:.3f} | {100*r['model_normal95_margin']:.3f} | {100*r['paired_difference_vs_FP32']:+.3f} | {100*r['paired_model_normal95_margin']:.3f} |")
    text.extend(['\nUncertainty is1.96*SD/sqrt(5) over trained-model means after averaging ten joint device/sampling realizations. Endpoint comparisons pair those model means first. These normal five-model intervals and multiple secondary contrasts are descriptive, not formal equivalence/coverage guarantees. Complete paired contrasts, all programmed-mean diagnostics and conventional UINT4 comparisons are retained in the CSV/JSON artifacts. Mean-weight inference is not expectation of the finite-T noisy ReLU network.\n',
        '## Resource and control ledger\n',
        '| Encoding | Field bound | Logical bits | Ideal endpoint epsilon | Bias step | Mean free nodes | Mean exact clamps | Logical command bits | Actual command bytes | Mode bits if packed |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in resources:text.append(f"| {r['encoding']} | {r['bound']:g} | {r['bits']} | {r['ideal_endpoint_probability_floor']:.6g} | {r['nominal_bias_step']:.6g} | {r['free_nodes_mean']:.1f} | {r['clamped_nodes_mean']:.1f} | {r['logical_command_payload_bits_mean']:.1f} | {r['actual_command_array_bytes_mean']:.1f} | {r['endpoint_mode_bits_if_packed_mean']:.0f} |")
    text.extend([f"\nRemoving clamps increases active sampled nodes and free-evaluation counts by {(resources[1]['active_update_ratio_to_clamped']-1)*100:.2f}% on average at equal T/P, while logical weight readouts and nominal binary-product counts still include all2368 coordinates in both cases. These are not matched active-update budgets. Ten-bit codes occupy actual uint16 arrays; logical ten-bit packing is not implemented or claimed. Fully free variants need no per-weight endpoint mode flags. Neural biases, target/programming information, calibration buffers and larger runtime graph/field/state arrays remain additional costs; this is not one total storage bit per weight.\n",
        f"Production sampling executes {result['executed_calibration_free_evaluations']:,} calibration and {result['executed_inference_free_evaluations']:,} inference free-node evaluations. Independent all-example replay repeats the latter count as verification work, not extra independent accuracy replicates or candidate inference cost.\n",
        f"Every raw calibration/field/code/mean/score/count passed an independent audit. A second CUDA program separately rebuilds public THRML factors and reconstructs both layers for every example in every case using exact integer state frequencies. Maximum scaled preactivation/logit errors are {replay['max_scaled_preactivation_error']:.3g}/{replay['max_scaled_logit_error']:.3g}, below2e-5 F32 summation tolerance. CPU/CUDA unit tests explicitly check zero clamp blocks, node counts, control endpoints/storage types and same-path neural outputs.\n",
        'These field ranges and code precisions are simulator scenarios, not demonstrated Extropic hardware capabilities. The result does not prove a physical control voltage span, area, energy/latency improvement, general network performance or superiority over conventional low-bit arithmetic. No drift occurs here. The broad goal remains active for clocked network quality, training, larger conversion and physical characterization.\n',
        '## Reproduction\n',
        'Run from the repository root with the recorded environments and original exports. Durable matching cases are hash-verified and resumed; a changed protocol is refused in the same directory.\n',
        '```bash\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.finite_endpoint_mnist\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.finite_endpoint_mnist_path_audit\nOPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_finite_endpoint_mnist\n.venv-thrml/bin/python -m research.thrml_spw.finite_endpoint_mnist_report\n```\n'])
    rendered='\n'.join(text)+'\n'
    for a,b in {'range[-4':'range [-4','and1-epsilon':'and 1-epsilon','Every2368':'Every 2368','positive-weight64':'positive-weight 64','all10000':'all 10000','mean1':'mean 1','log-SD0.2':'log-SD 0.2','SD0.2':'SD 0.2','original2368':'original 2368','uses4096':'uses 4096','to-1':'to -1','predeclared[':'predeclared [','All1600':'All 1600','and400':'and 400','is1.96':'is 1.96','all2368':'all 2368','below2e':'below 2e'}.items():rendered=rendered.replace(a,b)
    Path('results/thrml_spw/FINITE_ENDPOINT_MNIST_REPORT.md').write_text(rendered)
    print(json.dumps({'primary_comparison':primary,'calibrated_highT':[control,free4,free8,free10],'resources':resources},indent=2))


if __name__=='__main__':main()
