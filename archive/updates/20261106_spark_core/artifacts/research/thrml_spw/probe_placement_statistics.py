"""Frozen quality and paired descriptive resampling for fresh probe-placement test."""
import csv,json
from pathlib import Path
import numpy as np
from .confirmation_statistics import bootstrap_paired_difference
OUT=Path('results/thrml_spw/probe_placement')


def summarize(out=OUT):
    p=json.loads((out/'protocol.json').read_text());tokens=np.load(out/'tokens.npz')['tokens'];truth=tokens[:,1:];base=np.load(out/'baseline_tokens.npz');ref=base['FP32_nll'].astype(float).mean(-1);refac=(base['FP32_predictions']==truth).mean(-1)
    loss={};correct={};profiles=[];draws={}
    for method in p['methods']:
        rr=[np.load(out/'runs'/f'{method}_seed{s}'/'prefix_tokens.npz') for s in range(5)];loss[method]=np.stack([r['prefix12_nll'].astype(float).mean(-1) for r in rr]);correct[method]=np.stack([(r['prefix12_predictions']==truth).mean(-1) for r in rr])
        ce=bootstrap_paired_difference(loss[method],np.broadcast_to(ref,loss[method].shape),seed=2026116701);ac=bootstrap_paired_difference(correct[method],np.broadcast_to(refac,correct[method].shape),seed=2026116701)
        delta=float((loss[method]-ref).mean());adelta=float((correct[method]-refac).mean());upper=float(np.quantile(ce,.975));lower=float(np.quantile(ac,.025));point=bool(delta<=np.log(1.05) and adelta>=-.01)
        profiles.append({'method':method,'cross_entropy':float(loss[method].mean()),'perplexity':float(np.exp(loss[method].mean())),'relative_perplexity':float(np.exp(delta)),'token_accuracy':float(correct[method].mean()),'accuracy_difference':adelta,'point_passed':point,'bootstrap_CE_upper':upper,'bootstrap_relative_upper':float(np.exp(upper)),'bootstrap_accuracy_lower':lower,'passes':bool(point and upper<=np.log(1.05) and lower>=-.01)})
        draws[method+'_CE_vs_FP32']=ce;draws[method+'_accuracy_vs_FP32']=ac
    ce=bootstrap_paired_difference(loss['weight_fisher'],loss['fixed'],seed=2026116701);ac=bootstrap_paired_difference(correct['weight_fisher'],correct['fixed'],seed=2026116701)
    primary={'contrast':'weight_fisher-minus-fixed','CE_difference':float((loss['weight_fisher']-loss['fixed']).mean()),'perplexity_ratio':float(np.exp((loss['weight_fisher']-loss['fixed']).mean())),'accuracy_difference':float((correct['weight_fisher']-correct['fixed']).mean()),
        'CE_sensitivity_low':float(np.quantile(ce,.025)),'CE_sensitivity_high':float(np.quantile(ce,.975)),'accuracy_sensitivity_low':float(np.quantile(ac,.025)),'accuracy_sensitivity_high':float(np.quantile(ac,.975))}
    draws['paired_CE']=ce;draws['paired_accuracy']=ac;np.savez_compressed(out/'quality_draws.npz',**draws)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(profiles[0]));w.writeheader();w.writerows(profiles)
    (out/'paired_summary.json').write_text(json.dumps(primary,indent=2)+'\n')
    return profiles,primary


if __name__=='__main__':print(json.dumps(summarize(),indent=2))
