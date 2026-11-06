"""Fixed-prefix quality and crossed paired sensitivity, no choices from outcomes."""
import json
from pathlib import Path
import numpy as np
from .confirmation_statistics import bootstrap_paired_difference
from .transformer_block_gpu import write_csv,save_json
OUT=Path('results/thrml_spw/all_blocks')

def main():
    p=json.loads((OUT/'protocol.json').read_text());tokens=np.load(OUT/'tokens.npz')['tokens'];truth=tokens[:,1:];base=np.load(OUT/'baseline_tokens.npz');ref=base['FP32_nll'].astype(float).mean(-1);refac=(base['FP32_predictions']==truth).mean(-1);rows=[];loss={};acc={};draws={}
    raw=[np.load(OUT/'runs'/f'seed{s}'/'prefix_tokens.npz') for s in range(5)]
    for k in [12,24]:
        loss[k]=np.stack([r[f'prefix{k}_nll'].astype(float).mean(-1) for r in raw]);acc[k]=np.stack([(r[f'prefix{k}_predictions']==truth).mean(-1) for r in raw])
        ce=bootstrap_paired_difference(loss[k],np.broadcast_to(ref,loss[k].shape),seed=p['bootstrap_seed']);ac=bootstrap_paired_difference(acc[k],np.broadcast_to(refac,acc[k].shape),seed=p['bootstrap_seed']);draws[f'prefix{k}_CE_vs_FP32']=ce;draws[f'prefix{k}_accuracy_vs_FP32']=ac
        delta=float((loss[k]-ref).mean());ad=float((acc[k]-refac).mean());upper=float(np.quantile(ce,.975));lower=float(np.quantile(ac,.025));point=delta<=np.log(1.05) and ad>=-.01
        rows.append({'converted_projections':k,'converted_weights':k*1769472,'cross_entropy':float(loss[k].mean()),'perplexity':float(np.exp(loss[k].mean())),'relative_perplexity':float(np.exp(delta)),'token_accuracy':float(acc[k].mean()),'accuracy_difference':ad,'point_passed':bool(point),'bootstrap_relative_upper':float(np.exp(upper)),'bootstrap_accuracy_lower':lower,'passes':bool(point and upper<=np.log(1.05) and lower>=-.01)})
    ce=bootstrap_paired_difference(loss[24],loss[12],seed=p['bootstrap_seed']);ac=bootstrap_paired_difference(acc[24],acc[12],seed=p['bootstrap_seed']);draws['paired_CE']=ce;draws['paired_accuracy']=ac
    pair={'contrast':'all24-minus-first12-shared-native-prefix','CE_difference':float((loss[24]-loss[12]).mean()),'perplexity_ratio':float(np.exp((loss[24]-loss[12]).mean())),'accuracy_difference':float((acc[24]-acc[12]).mean()),'CE_sensitivity_low':float(np.quantile(ce,.025)),'CE_sensitivity_high':float(np.quantile(ce,.975)),'accuracy_sensitivity_low':float(np.quantile(ac,.025)),'accuracy_sensitivity_high':float(np.quantile(ac,.975))}
    write_csv(OUT/'summary.csv',rows);save_json(OUT/'paired_summary.json',pair);np.savez_compressed(OUT/'quality_draws.npz',**draws);return rows,pair
if __name__=='__main__':print(json.dumps(main(),indent=2))
