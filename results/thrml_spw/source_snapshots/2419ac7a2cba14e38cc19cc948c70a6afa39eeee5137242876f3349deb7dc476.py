"""Audited nonideal quarter-model quality and explicit programming/calibration costs."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .confirmation_statistics import bootstrap_paired_difference
OUT=Path('results/thrml_spw/calibrated_transformer');CONDS=['ideal','uncalibrated','cal4096']


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed'];m=json.loads((OUT/'manifest.json').read_text());p=m['config'];base=read(OUT/'baseline_metrics.csv');summary=read(OUT/'summary.csv');prefix=read(OUT/'prefix_metrics.csv')
    fp=next(r for r in base if r['condition']=='FP32');pp=np.load(OUT/'tokens.npz')['tokens'];native={}
    for condition in CONDS:
        native[condition]=np.stack([np.load(OUT/'runs'/f'{condition}_seed{s}'/'prefix_tokens.npz')['prefix12_nll'].astype(float).mean(-1) for s in range(5)])
    d=native['cal4096']-native['uncalibrated'];draws=bootstrap_paired_difference(native['cal4096'],native['uncalibrated'],seed=2026114701)
    paired={'contrast':'cal4096-minus-uncalibrated','CE_difference':float(d.mean()),'perplexity_ratio':float(np.exp(d.mean())),
        'crossed_seed_bank_CE_interval':[float(q) for q in np.quantile(draws,[.025,.975])],'crossed_seed_bank_perplexity_ratio_interval':[float(q) for q in np.exp(np.quantile(draws,[.025,.975]))],
        'scope':'Five joint device/sampling seeds crossed with eight banks of eight contexts. Heuristic exchangeable clusters, not formal coverage or five independently pretrained models.'}
    np.savez_compressed(OUT/'paired_calibration_draws.npz',CE_difference=draws);(OUT/'paired_calibration_summary.json').write_text(json.dumps(paired,indent=2)+'\n')
    storage=[]
    for c in p['control_artifacts']:
        a=np.load(c['path']);payload=sum(a[k].nbytes for k in a.files)
        storage.append({'stage':c['stage'],'weights':c['weights'],'code_bytes':a['codes'].nbytes,'range_bytes':a['lower'].nbytes+a['span'].nbytes,'bias_bytes':a['bias'].nbytes,'numeric_payload_bytes':payload,'archive_bytes':Path(c['path']).stat().st_size})
    infer=8*1024*21233664;cal=4096*21233664;totalpayload=sum(r['numeric_payload_bytes'] for r in storage)
    resources={'converted_weights':21233664,'model_unique_parameters':81912576,'converted_fraction':21233664/81912576,'free_weight_nodes':21233664,'exact_clamped_weight_nodes':0,
        'numeric_control_range_bias_payload_bytes':totalpayload,'numeric_payload_bits_per_converted_weight':8*totalpayload/21233664,'actual_control_archive_bytes':sum(r['archive_bytes'] for r in storage),
        'inference_free_evaluations_per_64context_profile':infer,'measured_calibration_free_evaluations_per_profile':cal,'observed_evaluation_multiplier_calibrated_vs_uncalibrated':1+cal/infer,
        'production_inference_free_evaluations':m['total_native_inference_free_evaluations'],'production_calibration_free_evaluations':m['total_native_calibration_free_evaluations'],
        'verification_inference_free_evaluations':m['total_native_inference_free_evaluations'],'verification_calibration_free_evaluations':m['total_native_calibration_free_evaluations'],
        'storage_by_stage':storage,'scope':'Observed native updates and finite artifact payload only. Expanded fields/ranges/state objects, readout and float accumulation, programming, calibration buffers and autonomous background switching are additional. No physical energy/latency or runtime-memory compression claim.'}
    (OUT/'resource_ledger.json').write_text(json.dumps(resources,indent=2)+'\n')
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');xx=np.arange(3)
    ratio=[float(r['relative_perplexity']) for r in summary];upper=[float(r['bootstrap_relative_upper']) for r in summary]
    ax[0,0].bar(xx,ratio,color=['tab:blue','tab:red','tab:green']);ax[0,0].scatter(xx,upper,marker='_',s=120,color='black',label='97.5% crossed upper');ax[0,0].axhline(1.05,ls='--',color='black',label='Frozen 5% quality threshold')
    ax[0,0].set(title='Quarter-model task quality',ylabel='Perplexity / FP32',xticks=xx,xticklabels=CONDS);ax[0,0].legend(fontsize=7)
    for condition in CONDS:
        means=[];margins=[]
        for stage in range(13):
            values=np.array([float(r['cross_entropy']) for r in prefix if r['condition']==condition and int(r['converted_prefix'])==stage]);means.append(values.mean()-float(fp['cross_entropy']));margins.append(1.96*values.std(ddof=1)/np.sqrt(5))
        ax[0,1].errorbar(np.arange(13),means,yerr=margins,fmt='.-',label=condition)
    ax[0,1].set(title='Regenerated sequential stochastic prefixes',xlabel='Converted projections in forward order',ylabel='CE change vs FP32');ax[0,1].legend(fontsize=8)
    ax[1,0].bar(xx,[100*float(r['accuracy_difference']) for r in summary]);ax[1,0].scatter(xx,[100*float(r['bootstrap_accuracy_difference_lower']) for r in summary],marker='_',s=120,color='black',label='2.5% crossed lower');ax[1,0].axhline(-1,ls='--',color='black')
    ax[1,0].set(title='Token prediction quality',xticks=xx,xticklabels=CONDS,ylabel='Accuracy difference vs FP32 (pp)');ax[1,0].legend(fontsize=7)
    ax[1,1].bar(xx,[infer/1e9]*3,label='Inference observed updates');ax[1,1].bar(xx,[0,0,cal/1e9],bottom=[infer/1e9]*3,label='Measured probes for this stream')
    ax[1,1].set(title='Candidate cost per 64-context profile',xticks=xx,xticklabels=CONDS,ylabel='Billion observed free-node evaluations');ax[1,1].legend(fontsize=7)
    for a in ax.ravel():a.grid(axis='y',alpha=.2)
    fig.savefig('results/thrml_spw/calibrated_transformer_evidence.png',dpi=180);plt.close(fig)
    calibrated=next(r for r in summary if r['condition']=='cal4096');uncal=next(r for r in summary if r['condition']=='uncalibrated')
    text=['# Measured finite-control calibration of a pretrained Transformer subset\n',
        f"The calibrated quarter-model scores perplexity {float(calibrated['perplexity']):.3f} versus {float(uncal['perplexity']):.3f} with uncalibrated synthetic devices and {float(fp['perplexity']):.3f} FP32. Relative calibrated perplexity is {float(calibrated['relative_perplexity']):.5f}; token-accuracy difference is {100*float(calibrated['accuracy_difference']):+.3f} percentage points. The frozen point-quality rule passes={calibrated['point_tolerance_passed']}; the additional crossed seed/bank margin rule passes={calibrated['passes']}. No failed gate is relaxed.\n",
        '## Representation and native execution\n',
        'The first three DistilGPT-2 blocks have all twelve affine projections converted,21,233,664 of81,912,576 unique model parameters(25.922%). Each signed matrix uses per-output/input-block128 lower/span metadata, with W=lower+span*p. Each normalized probability gets exactly one free THRML SpinNode. This is not direct p=raw signed weight; that direct mapping applies only to weights already in[0,1]. The other74.08% of the model, LayerNorms, biases and nonlinearities remain deterministic FP32.\n',
        'Normalized target endpoints are clipped to epsilon=sigmoid(-8),1-epsilon. There are zero exact clamp nodes, but target endpoint weights are approximated. Nominal fields use finite eight-bit controls in[-4,4]. Runtime operators load only codes, range metadata and neural biases; target floating matrices are used for independent references, not loaded into the native worker. Expanded simulator arrays and state objects still cost much more than the finite artifact.\n',
        'Every later projection receives student inputs regenerated after all earlier native replacements. The native observer accumulates sampled linear outputs before the original Transformer nonlinearities. One sampled matrix state is shared across eight contexts per bank. Software JAX/GPU arithmetic, sampling parallelism and float readout/accumulation are not native physical TSU gates or throughput measurements.\n',
        '## Frozen device and text protocol\n',
        'Static gain is independent lognormal mean1/log-SD0.2 and unary-field offset SD0.2 per weight. Synthetic device profiles remain fixed across probes, all input banks and all three conditions. No physical device measurement, temporal/cross-node persistence, drift or readout noise is added. Five independent joint device/sampling seeds are used for one fixed pretrained checkpoint.\n',
        'Two native probes collect2,048 samples each at finite command codes96/159. Jeffreys-smoothed logit counts fit a gain/offset line with declared gain guard[0.25,4]; corrected target commands are clipped/rounded to the same eight-bit grid. True device parameters, text inputs, labels and quality scores never enter this fit. Probes are shared as experimental evidence across target conditions on the same synthetic devices; only calibrated candidate deployment requires them.\n',
        'All15 quarter-model runs are predeclared: ideal, uncalibrated and measuredcal4096, five joint seeds, fixed T1024. There is no sample search or outcome-dependent extension. T/P budgets come from earlier independent ideal-model/calibration work. All64 new short-context windows(4,096 next-token targets) are disjoint from previously scored positions, including legacy overlapping windows. They come from the same pinned WikiText-2 corpus, not a new corpus or full-corpus/long-context evaluation.\n',
        'The fixed point rule requires relative perplexity<=1.05 and token-accuracy loss<=1 percentage point. The additional prospective sensitivity rule requires crossed-seed/bank97.5%-CE change<=log1.05 and2.5%-accuracy change>=-0.01,10,000 resamples with fixed seed2026114701. Banks contain all eight contexts sharing a matrix draw. These are exchangeable-cluster sensitivity intervals, not formal coverage guarantees or five independently pretrained models.\n',
        '## All quality conditions\n',
        '| Condition | Perplexity | PPL / FP32 | Token accuracy (%) | Accuracy delta (pp) | Point gate | Crossed-margin gate |\n|---|---:|---:|---:|---:|---|---|']
    for r in summary:text.append(f"| {r['condition']} | {float(r['perplexity']):.3f} | {float(r['relative_perplexity']):.5f} | {100*float(r['token_accuracy_mean']):.3f} | {100*float(r['accuracy_difference']):+.3f} | {r['point_tolerance_passed']} | {r['passes']} |")
    text.extend(['\n| Deterministic numerical control | Perplexity | Token accuracy (%) |\n|---|---:|---:|'])
    for r in base:text.append(f"| {r['condition']} | {float(r['perplexity']):.3f} | {100*float(r['token_accuracy']):.3f} |")
    text.extend([f"\nPrimary measured-calibration versus uncalibrated CE difference is {paired['CE_difference']:+.6f}; perplexity ratio {paired['perplexity_ratio']:.5f}, with crossed sensitivity interval {paired['crossed_seed_bank_perplexity_ratio_interval']}. Complete prefix scores, gate draws and raw per-token losses/predictions are preserved. Numerical low-bit controls are retained; this does not establish superiority over conventional arithmetic or an official Extropic four-p-bits-per-weight reference.\n",
        '## Resource ledger\n',
        f"The finite control/range/bias payload is {totalpayload:,} bytes, {resources['numeric_payload_bits_per_converted_weight']:.5f} bits per converted weight. This includes one eight-bit code per weight, two float32 range values per128-weight group and deterministic neural biases. Actual artifact archives total {resources['actual_control_archive_bytes']:,} bytes. Zero endpoint modes are needed for the all-free representation. These are artifact totals, not total runtime RAM, silicon area or one storage bit per weight.\n",
        f"One64-context profile executes {infer:,} observed inference free-node updates. Measured calibration adds {cal:,},50% more for this finite stream; reuse over R such streams would reduce count overhead to1+0.5/R under static retained controls. That amortization is not an energy/latency claim and fails if devices drift or controls are lost. Background switching, programming, calibration fitting, readout and external accumulation require separate accounting.\n",
        f"Production executes {m['total_native_calibration_free_evaluations']:,} probe and {m['total_native_inference_free_evaluations']:,} inference updates. Independent public replay repeats both only for verification, not extra device/model replicates.\n",
        '## Independent evidence and reporting correction\n',
        'Every native node in all120 probe windows and180 operator executions is independently replayed through separately constructed public THRML frequency programs. All fields, measured fits/codes, conditional bias/variance diagnostics, update counters and all native operator outputs are independently reconstructed. A separate original-model pre/post-hook replay checks every saved student input and every final token loss/prediction for all15 complete quarter-model cases. Deterministic baselines, all prefix links, per-token scores, all60,000 condition-quality bootstrap values and both fixed gates are independently recomputed.\n',
        'The first driver reused a summary helper hardcoded for128 windows. The64-window compatibility issue was identified during the unchanged native phase. After the original driver terminates, a separate finalizer reconstructs summaries from the fifteen completed raw cases using the prospectively specified formula at64 windows. No token, condition, sampling budget, quality threshold or raw native result is changed or resampled. The reporting-only correction and exact source versions are retained.\n',
        'The goal remains active. A successful bounded simulated regime would still leave full-model conversion, long contexts/corpus coverage, correlated/drifting physical devices, controller area/precision and total hardware costs unresolved. Any failed quality rule remains a reported negative result.\n'])
    rendered='\n'.join(text)+'\n'
    for a,b in {'converted,21':'converted, 21','of81':'of 81','parameters(':'parameters (','notdirect':'not direct','in[0':'in [0','other74':'other 74','in[-4':'in [-4','mean1':'mean 1','log-SD0.2':'log-SD 0.2','SD0.2':'SD 0.2','collect2,048':'collect 2,048','codes96':'codes 96','guard[':'guard [','All15':'All 15','All64':'All 64','windows(':'windows (','bank97.5':'bank 97.5','and2.5':'and 2.5',',10,000':', 10,000','seed2026114701':'seed 2026114701','per128':'per 128','One64':'One 64',',50%':', 50%','to1+':'to 1+','all120':'all 120','and180':'and 180','all15':'all 15','all60,000':'all 60,000','for128':'for 128','The64':'The 64','at64':'at 64'}.items():rendered=rendered.replace(a,b)
    Path('results/thrml_spw/CALIBRATED_TRANSFORMER_REPORT.md').write_text(rendered)
    print(json.dumps({'paired_calibration':paired,'quality':summary,'resources':{k:v for k,v in resources.items() if k!='storage_by_stage'}},indent=2))


if __name__=='__main__':main()
