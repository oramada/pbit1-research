"""Report the prospective larger-calibration study without hiding its extra cost."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/thrml_spw');OUT=ROOT/'quarter_conservative'

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    final=json.loads((OUT/'manifest.json').read_text());audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed']
    p=final['config'];selection=final['selection'];t=selection['selected_samples'];test=final['confirmation'];base=read(OUT/'baseline_metrics.csv')
    fp={split:next(r for r in base if r['split']==split and r['condition']=='FP32') for split in ['validation','test']}
    # Display the other side of each predeclared percentile bound from saved draws.
    full=[]
    for profile in final['profiles']+[test]:
        split,time=profile['split'],profile['samples'];draw=np.load(OUT/f'{split}_T{time}_selection_draws.npz')
        row={**profile,'bootstrap_relative_lower':float(np.exp(np.quantile(draw['CE_difference'],.025))),
            'bootstrap_accuracy_difference_upper':float(np.quantile(draw['accuracy_difference'],.975))}
        full.append(row)
    with (OUT/'uncertainty_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(full[0]));w.writeheader();w.writerows(full)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for split,color in [('validation','tab:blue'),('test','tab:orange')]:
        rows=sorted([r for r in full if r['split']==split],key=lambda r:r['samples'])
        axes[0].errorbar([r['samples'] for r in rows],[r['relative_perplexity'] for r in rows],
            yerr=[[r['relative_perplexity']-r['bootstrap_relative_lower'] for r in rows],[r['bootstrap_relative_upper']-r['relative_perplexity'] for r in rows]],fmt='o-',color=color,label=split)
    axes[0].axhline(1.05,color='black',ls=':',label='5% tolerance');axes[0].set(xlabel='Samples per operator',ylabel='Perplexity / same-text FP32',title='Paired seed/bank resampling sensitivity');axes[0].set_xticks(sorted(set(r['samples'] for r in full)));axes[0].legend(fontsize=7)
    names=['FP32','affine4_block128','affine8_block128','programmed_mean8','native_finite8']
    ratios=[1]+[float(np.exp(float(next(r for r in base if r['split']=='test' and r['condition']==name)['cross_entropy'])-float(fp['test']['cross_entropy']))) for name in names[1:4]]+[test['relative_perplexity']]
    axes[1].bar(range(5),ratios,color=['black','tab:green','tab:purple','tab:gray','tab:orange']);axes[1].axhline(1.05,color='black',ls=':')
    axes[1].set(xticks=range(5),xticklabels=['FP32','Affine4','Affine8','Mean8',f'Native T{t}'],ylabel='Confirmation perplexity / FP32',title='New reserved confirmation, no selection');axes[1].tick_params(axis='x',labelrotation=20)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'quarter_conservative_evidence.png',dpi=180);plt.close(fig)
    run=json.loads((OUT/'runs'/f'test_T{t}_seed0'/'run.json').read_text());controls=[json.loads(Path(s['metadata']).read_text())['controls'] for s in run['stages']]
    n=sum(c['logical_pbits'] for c in controls);free=sum(c['free_pbits'] for c in controls)
    resources={'logical_nodes':n,'free_nodes':free,'selected_samples':t,'free_node_evaluations_per_confirmation_seed':test['free_node_updates'],
        'sampling_rounds_per_bank_batch':12*t,'bank_batches':16,'serial_rounds_all_confirmation_banks':16*12*t,'factor_versus_previous_T512_same_window_count':t/512,
        'range_metadata_bytes':sum(c['metadata_bits'] for c in controls)//8,'actual_field_bytes':sum(c['bias_storage_bytes_simulator'] for c in controls),
        'actual_uint32_code_bytes':4*free,'expanded_range_buffer_bytes':sum(c['expanded_range_buffer_bytes_simulator'] for c in controls),
        'total_executed_study_free_node_evaluations':sum(int(r['free_node_updates']) for r in read(OUT/'native_metrics.csv')),
        'scope':'Counters and selected simulator array subtotals; calibration and confirmation work separated from per-inference counts; not physical transition/area/energy/latency measurements'}
    (OUT/'resource_ledger.json').write_text(json.dumps(resources,indent=2)+'\n')
    test_full=next(r for r in full if r['split']=='test')
    lines=['# Conservative sampling study after the failed T512 confirmation\n',
        'This is a new prospective study. The earlier frozen quarter-model confirmation missed the5% point-mean tolerance at T512; its result remains unchanged. This study uses128 new validation windows and128 new test windows,8,192 targets per split, excluding all previously consumed positions and the documented old pilot. It uses the same WikiText-2 corpus, not a new dataset or pretraining-clean guarantee.\n',
        '## Fixed representation and selection rule\n',
        'All12 projections in DistilGPT-2 blocks0/1/2 execute through official THRML:21,233,664 normalized weight nodes,25.9223% of model parameters. Controls stay at8-bit uniform dimensionless Ising-b levels in [-4,4], with local input-block128 affine lower/span metadata. Biases, LayerNorm, embeddings/head and the last three blocks stay deterministic. Student activations are regenerated after earlier native substitutions. Linear accumulation precedes downstream nonlinearities.\n',
        'The predeclared ascending search is1024/2048 samples, five native seeds per budget. Validation-only selection requires both point-mean constraints (CE increase<=log(1.05), accuracy loss<=one point), plus crossed seed/bank bootstrap97.5%-CE increase<=log(1.05) and2.5%-accuracy difference>=-0.01. Ten thousand resamples use seed20261111. These are conservative engineering sensitivity constraints under exchangeable banks, not formal corpus-coverage guarantees. The selected T is written before any confirmation scoring.\n',
        '| Split | T | Perplexity | Relative to FP32 | Bootstrap upper ratio | Accuracy difference lower, pp | Point passed | All constraints passed |\n|---|---:|---:|---:|---:|---:|---|---|']
    for r in full:lines.append(f"| {r['split']} | {r['samples']} | {r['perplexity']:.3f} | {r['relative_perplexity']:.4f} | {r['bootstrap_relative_upper']:.4f} | {100*r['bootstrap_accuracy_difference_lower']:+.3f} | {r['point_tolerance_passed']} | {r['passes']} |")
    lines.extend([f"\nCalibration selected T={t}; calibration gate passed: {selection['gate_passed']}. Confirmation point-mean tolerance passed: {test['point_tolerance_passed']}. Confirmation resampling-margin constraints passed: {test['passes']}. These are separate reported outcomes, not further selection.\n",
        f"Confirmation native perplexity is {test['perplexity']:.3f} versus {float(fp['test']['perplexity']):.3f} FP32 ({100*(test['relative_perplexity']-1):+.2f}%). Its paired seed/bank sensitivity interval is [{test_full['bootstrap_relative_lower']:.4f},{test['bootstrap_relative_upper']:.4f}]; accuracy difference is {100*test['accuracy_difference']:+.3f} percentage points. Confirmation resamples use the predeclared seed20261112.\n",
        '## Conventional and programmed-mean controls\n',
        '| Split | Condition | Perplexity | Token accuracy |\n|---|---|---:|---:|'])
    for r in base:lines.append(f"| {r['split']} | {r['condition']} | {float(r['perplexity']):.3f} | {100*float(r['token_accuracy']):.3f}% |")
    lines.extend(['\nThe programmed-mean model diagnoses the persistent8-bit control error without sampling. It is conventional deterministic inference, not a native sample result. Affine4/8 preserve zero and use the same block128 grouping; these are numerical dequantized controls, not optimized packed kernels or exhaustive GPTQ/AWQ comparisons. A native quality result alone establishes no physical or memory advantage.\n',
        '## Sampling costs and verification\n',
        f"Per confirmation seed: {test['free_node_updates']:,} free-node evaluations, {12*t:,} rounds per eight-context bank batch, and {16*12*t:,} rounds with serial reuse over all16 batches. Sampling cost is {t/512:g} times the previous T512 study at equal text count. The entire declared calibration/confirmation study executes {resources['total_executed_study_free_node_evaluations']:,} free-node evaluations. Calibration work is not hidden in a per-inference number. These counters are not measured physical transitions or chip latency/energy.\n",
        f"There are {n:,} logical nodes and {free:,} free nodes. Group metadata uses {resources['range_metadata_bytes']:,} bytes; actual simulator arrays retain {resources['actual_field_bytes']:,} bytes each of float32 fields and uint32 codes, plus {resources['expanded_range_buffer_bytes']:,} bytes of expanded ranges and other program/state/activation/accumulator/unconverted-model costs. The prior hypothetical control/mode/range/bias subtotal is10.4063 bits per converted weight, versus affine4:4.3125 and affine8:8.34375. This is not one total storage bit per weight or achieved Python compression.\n",
        f"Independent audits verify {audit['native_runs']} native runs, {audit['operator_checkpoints']} operator checkpoints, exact separate hook replay of every native final output, eight independent model baseline replays, source/key/data lineage, conditional control-bias/sampling MSE, logical counts, and {audit['independent_selection_bootstrap_values']:,} independently reconstructed selection/confirmation bootstrap values. Three targeted selection checks also cover a favorable mean with excessive seed uncertainty and a bank-correlated accuracy failure.\n",
        'All failed budgets and earlier confirmation outcomes remain preserved. These test positions are consumed once scored and cannot become fresh evidence for a tuned successor. The long-horizon goal remains active for broader conversion, nonuniform/adaptive budgets, grouped partial-sum implementation, nonideal large operators, training and physical-system measurements.\n'])
    text='\n'.join(lines)+'\n'
    for a,b in {'the5%':'the 5%','uses128':'uses 128','and128':'and 128','windows,8,192':'windows, 8,192','All12':'All 12','blocks0/1/2':'blocks 0/1/2','THRML:21':'THRML: 21',',25.9223':', 25.9223','at8-bit':'at 8-bit','is1024/2048':'is 1024/2048','bootstrap97.5':'bootstrap 97.5','and2.5':'and 2.5','seed20261111':'seed 20261111','seed20261112':'seed 20261112','persistent8':'persistent 8','all16':'all 16','is10.4063':'is 10.4063'}.items():text=text.replace(a,b)
    (ROOT/'QUARTER_CONSERVATIVE_REPORT.md').write_text(text);print(json.dumps({'selection':selection,'confirmation':test_full,'resources':resources},indent=2))

if __name__=='__main__':main()
