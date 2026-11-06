"""Frozen quarter-model confirmation report and predeclared uncertainty analysis."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .confirmation_statistics import bootstrap_paired_difference

ROOT=Path('results/thrml_spw');OUT=ROOT/'quarter_confirmation'

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed']
    final=json.loads((OUT/'manifest.json').read_text());p=final['config'];summary=final['summary'];assert p['selected_samples']==512
    tokens=np.load(p['input_data_path'])['tokens'];base=np.load(OUT/'baseline_tokens.npz');names=p['conditions']
    losses={};accuracies={}
    for c in names:
        if c=='native_finite8':
            raw=[np.load(OUT/'runs'/f'confirmation_seed{s}'/'prefix_tokens.npz') for s in range(5)]
            losses[c]=np.stack([r['prefix12_nll'].mean(-1,dtype=np.float64) for r in raw]);accuracies[c]=np.stack([(r['prefix12_predictions']==tokens[:,1:]).mean(-1) for r in raw])
        else:
            losses[c]=np.broadcast_to(base[c+'_nll'].mean(-1,dtype=np.float64),(5,128));accuracies[c]=np.broadcast_to((base[c+'_predictions']==tokens[:,1:]).mean(-1),(5,128))
    pairs=[(c,'FP32') for c in names if c!='FP32']+ [('native_finite8',c) for c in ['affine4_block128','affine8_block128','programmed_mean8']]
    rows=[];draws={}
    for left,right in pairs:
        ce=bootstrap_paired_difference(losses[left],losses[right],seed=20261108);acc=bootstrap_paired_difference(accuracies[left],accuracies[right],seed=20261108)
        lo,hi=np.quantile(ce,[.025,.975]);alo,ahi=np.quantile(acc,[.025,.975]);key=left+'_minus_'+right
        draws[key+'_CE']=ce;draws[key+'_accuracy']=acc
        rows.append({'left':left,'right':right,'mean_cross_entropy_difference':float(np.mean(losses[left]-losses[right])),
            'relative_perplexity':float(np.exp(np.mean(losses[left]-losses[right]))),'bootstrap_relative_lower':float(np.exp(lo)),'bootstrap_relative_upper':float(np.exp(hi)),
            'bootstrap_CE_lower':float(lo),'bootstrap_CE_upper':float(hi),'accuracy_difference_pp':100*float(np.mean(accuracies[left]-accuracies[right])),
            'bootstrap_accuracy_lower_pp':100*float(alo),'bootstrap_accuracy_upper_pp':100*float(ahi),'resamples':10000,'seed':20261108,
            'scope':'Paired crossed seed/bank bootstrap sensitivity under exchangeable clusters, not distribution-free corpus coverage'})
    with (OUT/'paired_bootstrap.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'paired_bootstrap_draws.npz',**draws)
    fp=float(losses['FP32'].mean());ratios=[float(np.exp(losses[c].mean()-fp)) for c in names]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].bar(range(5),ratios,color=['black','tab:green','tab:purple','tab:gray','tab:blue'])
    axes[0].axhline(1.05,color='black',ls=':',label='Frozen 5% mean tolerance')
    axes[0].set(xticks=range(5),xticklabels=['FP32','Affine4','Affine8','Mean8','Native8 T512'],ylabel='Perplexity / FP32',title='25.92% native conversion: 8,192 reserved targets');axes[0].tick_params(axis='x',labelrotation=20);axes[0].legend(fontsize=7,loc='lower left')
    for i,c in enumerate(['affine4_block128','native_finite8']):
        r=next(r for r in rows if r['left']==c and r['right']=='FP32');mean=r['relative_perplexity'];axes[1].errorbar(i,mean,yerr=[[mean-r['bootstrap_relative_lower']],[r['bootstrap_relative_upper']-mean]],fmt='o',color=['tab:green','tab:blue'][i])
    axes[1].axhline(1.05,color='black',ls=':');axes[1].set(xticks=range(2),xticklabels=['Affine4','Native8 T512'],ylabel='Perplexity / FP32',title='Paired seed/bank resampling sensitivity')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'quarter_confirmation_evidence.png',dpi=180);plt.close(fig)
    native=next(r for r in rows if r['left']=='native_finite8' and r['right']=='FP32')
    run=json.loads((OUT/'runs/confirmation_seed0/run.json').read_text());controls=[json.loads(Path(s['metadata']).read_text())['controls'] for s in run['stages']]
    resources={'logical_nodes':sum(c['logical_pbits'] for c in controls),'free_nodes':sum(c['free_pbits'] for c in controls),
        'range_metadata_bytes':sum(c['metadata_bits'] for c in controls)//8,'actual_float32_field_bytes':sum(c['bias_storage_bytes_simulator'] for c in controls),
        'actual_uint32_code_bytes':sum(c['free_pbits'] for c in controls)*4,'expanded_range_buffer_bytes':sum(c['expanded_range_buffer_bytes_simulator'] for c in controls),
        'free_node_evaluations_per_seed':summary['free_node_updates'],'sampling_depth_per_bank_batch':12*512,'bank_batches':16,'serial_sampling_rounds_all_bank_batches':12*512*16,
        'scope':'Simulator logical/update counts and selected array subtotals, excluding program/state/accumulation/activation and unconverted model costs; no physical-chip area, latency or energy measurement'}
    (OUT/'resource_ledger.json').write_text(json.dumps(resources,indent=2)+'\n')
    lines=['# Frozen confirmation of quarter-model p-bit conversion\n',
        'All twelve projection matrices in DistilGPT-2 blocks0/1/2 use actual official THRML sampling: 21,233,664 normalized weight nodes, 25.9223% of unique model parameters. Eight-bit dimensionless Ising-b controls in [-4,4], input-block128 affine lower/span metadata and T512 were frozen after calibration. No settings changed in response to the development-test miss or confirmation scores. Biases, LayerNorm, embeddings/head and the last three blocks stay deterministic.\n',
        '## Provenance and fixed comparison\n',
        'The 128 reserved65-token windows contain8,192 targets and have no positional overlap with the original32 native test windows, the consumed first-block128-window confirmation or the documented four-window generic pilot. Token/source/offset/selection provenance was independently audited before scoring. These are new positions in the same WikiText-2 test corpus, not a new dataset or a pretraining-clean guarantee. The earlier calibration and development-test figures use different text and cannot be compared by absolute perplexity alone.\n',
        'The frozen conditions are FP32, conventional zero-preserving affine4/8 with blocks128, programmed-mean8, and native8/T512. Five native sampling seeds use16 banks of eight contexts. Every downstream student input is regenerated after all earlier native outputs and checked exactly. Linear accumulation precedes downstream nonlinearities.\n',
        '| Condition | Perplexity | Relative to FP32 | Token accuracy |\n|---|---:|---:|---:|']
    for c,ratio in zip(names,ratios):lines.append(f"| {c} | {np.exp(losses[c].mean()):.3f} | {ratio:.4f} | {100*accuracies[c].mean():.3f}% |")
    lines.extend([f"\nThe fixed native configuration {'meets' if summary['mean_quality_tolerance_passed'] else 'misses'} the point-mean criterion. Its perplexity increase is {100*(summary['relative_perplexity']-1):.2f}%, and accuracy loss is {100*(accuracies['FP32'].mean()-accuracies['native_finite8'].mean()):.3f} percentage points. The predeclared tolerance is at most5% perplexity increase and one-point accuracy loss. Both components and every outcome are retained; no new precision/budget is selected here.\n",
        f"Conditional five-seed normal relative interval: [{summary['relative_perplexity_lower']:.4f},{summary['relative_perplexity_upper']:.4f}]. Crossed paired seed/bank-bootstrap sensitivity interval: [{native['bootstrap_relative_lower']:.4f},{native['bootstrap_relative_upper']:.4f}]. Ten thousand resamples use seed20261108 and preserve eight-context sharing. They describe sensitivity under exchangeable bank clusters, not a distribution-free unseen-corpus guarantee. A point-mean pass or miss is distinct from interval coverage of the threshold.\n",
        '## Paired comparisons\n',
        '| Comparison | Relative perplexity | Paired bootstrap interval | Accuracy difference, pp |\n|---|---:|---:|---:|'])
    for r in rows:lines.append(f"| {r['left']} / {r['right']} | {r['relative_perplexity']:.4f} | [{r['bootstrap_relative_lower']:.4f},{r['bootstrap_relative_upper']:.4f}] | {r['accuracy_difference_pp']:+.3f} |")
    lines.extend(['\nProgrammed-mean8 runs the deterministic model with the exact expected weights induced by the programmed lattice. It diagnoses the T-to-infinity control floor; it is not a separate native stochastic execution. Conventional baselines use numerical dequantization, not optimized packed kernels or best-available GPTQ/AWQ methods. No physical advantage follows from a quality comparison.\n',
        '## Resources, checks and remaining scope\n',
        f"Each native seed uses {summary['free_node_updates']:,} free-node evaluations across sixteen eight-context bank batches, at6,144 sampling rounds per batch or98,304 for serial bank reuse over all confirmation text. Range metadata is {resources['range_metadata_bytes']:,} bytes. Actual arrays retain {resources['actual_float32_field_bytes']:,} bytes of float32 fields, the same number of uint32-code bytes and {resources['expanded_range_buffer_bytes']:,} bytes of expanded range arrays, before other costs.\n",
        'The earlier hypothetical control/mode/range/bias ledger is10.4063 digital bits per converted matrix weight, versus affine4:4.3125 and affine8:8.34375. It excludes stochastic states, program objects, accumulators and unconverted weights; it is not actual software compression or analog device area. Multiplication/accumulation remain modeled external JAX arithmetic. The new worker retains12 compiled operators instead of thrashing an eight-entry shared cache; CPU/CUDA exact-output tests validate that software change without changing samples or counts.\n',
        f"Independent evidence checks cover {audit['native_runs']} complete native cases, {audit['operator_checkpoints']} operator checkpoints, separate pre/post-hook replay of every native token array and four independently reproduced deterministic model conditions. Raw weights, student inputs, conditional references, native outputs, token losses, all source/key lineage, finite-field bias/variance decomposition, resource counts and bootstrap draws remain available.\n",
        'This confirmation text is now consumed and must not be described as fresh evidence for a successor tuned on its results. The long-horizon goal remains active for new development studies, broader conversion, nonideal large operators, improved training/accumulation and physical programming/readout costs. The representation and a quarter-model quality outcome do not prove full-model or hardware-system superiority.\n'])
    text='\n'.join(lines)+'\n'
    for a,b in {'blocks0/1/2':'blocks 0/1/2','reserved65':'reserved 65','contain8,192':'contain 8,192','original32':'original 32','first-block128':'first-block 128','blocks128':'blocks of 128','use16':'use 16','most5%':'most 5%','seed20261108':'seed 20261108','at6,144':'at 6,144','or98,304':'or 98,304','is10.4063':'is 10.4063','retains12':'retains 12'}.items():text=text.replace(a,b)
    (ROOT/'QUARTER_CONFIRMATION_REPORT.md').write_text(text);print(json.dumps({'summary':summary,'native_bootstrap':native,'resources':resources},indent=2))

if __name__=='__main__':main()
