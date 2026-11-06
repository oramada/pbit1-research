"""Descriptive paired diagnostics; no new quality gate or candidate selection."""
import csv,json
from pathlib import Path
import numpy as np
from .confirmation_statistics import bootstrap_paired_difference
OUT=Path('results/thrml_spw/transformer_calibration_floor');OLD=Path('results/thrml_spw/calibrated_transformer')


def main():
    p=json.loads((OUT/'protocol.json').read_text());tokens=np.load(p['tokens_path'])['tokens'];truth=tokens[:,1:];base=np.load(OLD/'baseline_tokens.npz');reference=float(base['FP32_nll'].astype(float).mean());reference_acc=float(np.mean(base['FP32_predictions']==truth))
    native={};preds={};rows=[]
    for label,root,condition in [('measured8',OLD,'cal4096'),('ideal8',OLD,'ideal'),('oracle8',OUT,'oracle8')]:
        raw=[np.load(root/'runs'/f'{condition}_seed{s}'/'prefix_tokens.npz') for s in range(5)]
        native[label]=np.stack([r['prefix12_nll'].astype(float).mean(-1) for r in raw]);preds[label]=np.stack([(r['prefix12_predictions']==truth).mean(-1) for r in raw])
        ce=native[label].mean();ac=preds[label].mean()
        rows.append({'condition':label,'joint_profiles':5,'cross_entropy':float(ce),'perplexity':float(np.exp(ce)),'relative_perplexity_FP32':float(np.exp(ce-reference)),'token_accuracy':float(ac),'accuracy_difference_FP32':float(ac-reference_acc),'CE_seed_normal95_margin':float(1.96*native[label].mean(-1).std(ddof=1)/np.sqrt(5))})
    comparisons=[];draws={}
    for label,left,right in [('oracle_minus_measured','oracle8','measured8'),('oracle_minus_ideal','oracle8','ideal8')]:
        ce=bootstrap_paired_difference(native[left],native[right],seed=2026114701);ac=bootstrap_paired_difference(preds[left],preds[right],seed=2026114701)
        delta=float(np.mean(native[left]-native[right]));adelta=float(np.mean(preds[left]-preds[right]))
        comparisons.append({'contrast':label,'CE_difference':delta,'perplexity_ratio':float(np.exp(delta)),'accuracy_difference':adelta,'crossed_CE_low':float(np.quantile(ce,.025)),'crossed_CE_high':float(np.quantile(ce,.975)),'crossed_accuracy_low':float(np.quantile(ac,.025)),'crossed_accuracy_high':float(np.quantile(ac,.975))})
        draws[label+'_CE']=ce;draws[label+'_accuracy']=ac
    np.savez_compressed(OUT/'paired_diagnostic_draws.npz',**draws)
    means=[]
    for label in p['mean_conditions']:
        seeds=[0] if label=='ideal8' else list(range(5));raw=[np.load(OUT/'mean_runs'/f'{label}_seed{s}.npz') for s in seeds];loss=[r['nll'].astype(float).mean() for r in raw];acc=[float(np.mean(r['predictions']==truth)) for r in raw]
        means.append({'condition':label,'device_profiles':len(seeds),'cross_entropy':float(np.mean(loss)),'perplexity':float(np.exp(np.mean(loss))),'relative_perplexity_FP32':float(np.exp(np.mean(loss)-reference)),'token_accuracy':float(np.mean(acc))})
    diagnostics=json.loads((OUT/'weight_diagnostics.json').read_text());keys=list(diagnostics[0]['moments']);totals={k:sum(r['moments'][k] for r in diagnostics) for k in keys};n=totals['weights']
    aggregate={'weights_across_device_profiles':n,'components_mean_squared':{k:v/n for k,v in totals.items() if k not in ['weights','SSE_closure_error']},'SSE_closure_error_sum':totals['SSE_closure_error'],
        'known_inverse_outside_bound_fraction':sum(r['true_inverse_bound_clipped'] for r in diagnostics)/n,'scope':'Signed-weight mean-square telescoping replacement effects plus signed cross terms, not additive causal percentages of neural-task loss.'}
    (OUT/'weight_error_summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    for name,rr in [('native_summary.csv',rows),('paired_diagnostics.csv',comparisons),('mean_summary.csv',means)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    (OUT/'analysis_convention.json').write_text(json.dumps({'method':'Inherited source-study crossed seed/bank diagnostic sensitivity','resampling_seed':2026114701,'resamples':10000,'bank_contexts':8,'quality_gates_applied':False,'candidate_selected':False,'scope':'Descriptive analysis of alreadyconsumedtext andpairedknownparameters, not freshconfirmation, formalcoverage or deployablecalibration.'},indent=2)+'\n')
    print(json.dumps({'native':rows,'mean':means,'contrasts':comparisons,'weight_error':aggregate},indent=2))


if __name__=='__main__':main()
