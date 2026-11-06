"""Confirmation scores and the predeclared crossed seed/bank sensitivity analysis."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .confirmation_statistics import bootstrap_paired_difference

ROOT=Path('results/thrml_spw');OUT=ROOT/'transformer_confirmation'

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed']
    p=json.loads((OUT/'protocol.json').read_text());assert p['samples']==256 and p['selected_field_bits']==8
    tokens=np.load(p['input_data_path'])['tokens'];base=np.load(OUT/'baseline_tokens.npz')
    summaries=read(OUT/'summary.csv');baselines=read(OUT/'baseline_metrics.csv')
    losses={};correct={}
    for c in ('FP32','affine_zero_block128_4'):
        losses[c]=np.broadcast_to(base[c+'_nll'].mean(-1,dtype=np.float64),(5,128))
        correct[c]=np.broadcast_to((base[c+'_predictions']==tokens[:,1:]).mean(-1),(5,128))
    for c in ('native_float','native_finite8'):
        raw=[np.load(OUT/'runs'/f'{c}_seed{s}'/'prefix_tokens.npz') for s in range(5)]
        losses[c]=np.stack([r['prefix4_nll'].mean(-1,dtype=np.float64) for r in raw])
        correct[c]=np.stack([(r['prefix4_predictions']==tokens[:,1:]).mean(-1) for r in raw])
    pairs=[('affine_zero_block128_4','FP32'),('native_float','FP32'),('native_finite8','FP32'),('native_finite8','native_float'),('native_finite8','affine_zero_block128_4')]
    rows=[];draws={}
    for left,right in pairs:
        delta=bootstrap_paired_difference(losses[left],losses[right]);acc=bootstrap_paired_difference(correct[left],correct[right]);lo,hi=np.quantile(delta,[.025,.975]);alo,ahi=np.quantile(acc,[.025,.975]);name=f'{left}_minus_{right}'
        draws[name+'_CE']=delta;draws[name+'_accuracy']=acc
        rows.append({'left':left,'right':right,'mean_cross_entropy_difference':float(np.mean(losses[left]-losses[right])),
            'bootstrap_CE_lower':float(lo),'bootstrap_CE_upper':float(hi),'relative_perplexity':float(np.exp(np.mean(losses[left]-losses[right]))),
            'bootstrap_relative_lower':float(np.exp(lo)),'bootstrap_relative_upper':float(np.exp(hi)),
            'mean_accuracy_difference_pp':100*float(np.mean(correct[left]-correct[right])),
            'bootstrap_accuracy_lower_pp':100*float(alo),'bootstrap_accuracy_upper_pp':100*float(ahi),
            'resamples':10000,'seed':20261106,'scope':'Crossed paired seed/bank bootstrap sensitivity, not a finite-sample or corpus-population coverage guarantee'})
    with (OUT/'paired_bootstrap.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'paired_bootstrap_draws.npz',**draws)
    fp=next(r for r in baselines if r['condition']=='FP32');fp_ce=float(fp['cross_entropy'])
    names=['FP32','affine_zero_block128_4','native_float','native_finite8']
    ratios=[float(np.exp(losses[name].mean()-fp_ce)) for name in names]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].bar(range(4),ratios,color=['black','tab:green','tab:orange','tab:blue'])
    axes[0].axhline(1.05,color='black',ls=':',label='Frozen 5% point-mean tolerance')
    axes[0].set(xticks=range(4),xticklabels=['FP32','Affine4','Float T256','8-bit T256'],ylabel='Perplexity / FP32',title='Reserved 8,192 targets; first-block projections');axes[0].set_ylim(.98,max(1.06,max(ratios)+.02));axes[0].legend(fontsize=7)
    for i,name in enumerate(names[1:]):
        row=next(r for r in rows if r['left']==name and r['right']=='FP32');mean=row['relative_perplexity'];axes[1].errorbar(i,mean,yerr=[[mean-row['bootstrap_relative_lower']],[row['bootstrap_relative_upper']-mean]],fmt='o',color=['tab:green','tab:orange','tab:blue'][i])
    axes[1].axhline(1.05,color='black',ls=':');axes[1].set(xticks=range(3),xticklabels=['Affine4','Float T256','8-bit T256'],ylabel='Perplexity / FP32',title='Paired seed/bank-bootstrap sensitivity')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'transformer_confirmation_evidence.png',dpi=180);plt.close(fig)
    lines=['# Frozen confirmation on disjoint Transformer text windows\n',
        'This is a confirmation of the already selected first-block representation, not a precision search. All four DistilGPT-2 block0 matrix projections (7,077,888 weights, 8.64% of model parameters) run through official THRML. Later student inputs are regenerated after prior stochastic substitutions; linear outputs are accumulated before downstream nonlinearities. Other matrices, neural biases and LayerNorm stay deterministic.\n',
        '## Frozen data and conditions\n',
        'The reserved 128 nonoverlapping 65-token windows contain 8,192 next-token targets. Token positions do not overlap prior native test tiles or the documented older four-window generic pilot. Token/offset provenance and the old pilot score were independently audited before confirmation inference. These are new positions in the same WikiText-2 test corpus, not a new dataset or a pretraining-clean claim.\n',
        'The predeclared conditions are FP32, conventional zero-preserving affine4 with input blocks of 128, floating-field native T256, and selected 8-bit uniform-field native T256 in [-4,4]. Grouping is per-output input-block128; native conditions have five sampling seeds with eight contexts sharing each bank. No condition or sample count changes in response to confirmation scores. The protocol hashes the prior calibration selection and reserved tokens.\n',
        '| Condition | Perplexity | Relative to FP32 | Token accuracy |\n|---|---:|---:|---:|']
    for name,ratio in zip(names,ratios):lines.append(f"| {name} | {np.exp(losses[name].mean()):.3f} | {ratio:.4f} | {100*correct[name].mean():.3f}% |")
    lines.extend(['\nThe criterion is point-mean CE increase <=log(1.05) and token-accuracy loss <=1 percentage point; it is reported without further selection. Perplexity for native conditions is exp(mean cross-entropy across seeds), not a mean of perplexities.\n',
        '| Native condition | Mean tolerance passed | Conditional MC relative interval |\n|---|---|---:|'])
    for r in summaries:lines.append(f"| {r['condition']} | {r['mean_quality_tolerance_passed']} | [{float(r['relative_perplexity_lower']):.4f},{float(r['relative_perplexity_upper']):.4f}] |")
    lines.extend(['\n## Text and sampling sensitivity\n',
        'The conditional normal intervals describe five Monte Carlo seeds on fixed text. Additionally, the predeclared analysis resamples seed indices and sixteen banks of eight windows together, preserving paired conditions and within-bank sharing. Ten thousand resamples use seed 20261106. These percentile intervals measure sensitivity under exchangeable bank clusters, not a distribution-free guarantee for unseen corpora. A point-mean tolerance pass is distinct from whether the full resampling interval falls inside the tolerance.\n',
        '| Comparison | Relative perplexity | Paired bootstrap interval | Accuracy difference, pp |\n|---|---:|---:|---:|'])
    for r in rows:lines.append(f"| {r['left']} / {r['right']} | {r['relative_perplexity']:.4f} | [{r['bootstrap_relative_lower']:.4f},{r['bootstrap_relative_upper']:.4f}] | {r['mean_accuracy_difference_pp']:+.3f} |")
    updates=int(summaries[0]['free_node_updates'])
    lines.extend(['\nThis is a fixed confirmation outcome. Any future range/controller/budget changes require separate development data and another independent confirmation set; these scores cannot serve as fresh evidence for a tuned successor.\n',
        '## Costs and limits\n',
        f"Each native condition uses {updates:,} free-node evaluations per seed for the 128 windows. This is sixteen bank batches; sampling depth is 1,024 dependent rounds per eight-context batch, or 16,384 rounds across serial reuse of the same banks for all sixteen batches. Arithmetic includes external JAX multiplications and accumulation. These counts are not measured hardware transitions, silicon latency or energy.\n",
        'Range metadata remains 0.5 bits per matrix weight. Selected finite controls use 8 bits per free node plus endpoint modes and ranges; the prior hypothetical packed control/mode/range/bias subtotal is 10.4063 bits per matrix weight, versus 4.3125 for the conventional affine4 code/scale/zero-point/bias ledger. Actual software retains float32 fields, uint32 codes, expanded lower/span arrays, stochastic state, program objects and accumulation buffers. One logical p-bit per weight is not one total storage bit or a demonstrated systems advantage.\n',
        f"Independent verification covers {audit['confirmation_native_runs']} completed native runs, {audit['native_operator_checkpoints']} operator checkpoints, all native final-token outputs via separate pre/post-hook execution, and both conventional model baselines. Raw token arrays, stage inputs/conditional references/outputs, source and key signatures, checksum manifests, bootstrap draws and negative outcomes are retained.\n",
        'The long-horizon goal remains active: this is 8.64% matrix conversion on short-context windows, with ideal independent banks and modeled finite bias levels. Larger converted fractions, nonideal large operators, training improvements and physical controller/readout/accumulation costs remain open.\n'])
    (ROOT/'TRANSFORMER_CONFIRMATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'summaries':summaries,'paired_sensitivity':rows},indent=2))

if __name__=='__main__':main()
