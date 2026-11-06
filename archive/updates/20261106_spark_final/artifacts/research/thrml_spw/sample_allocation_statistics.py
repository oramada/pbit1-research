"""Fixed paired held-out statistics, independent of allocation construction."""
import json
from pathlib import Path
import numpy as np
from .confirmation_statistics import bootstrap_paired_difference
from .transformer_block_gpu import write_csv,save_json
OUT=Path('results/thrml_spw/sample_allocation')
def main():
    p=json.loads((OUT/'protocol.json').read_text());truth=np.load(OUT/'tokens.npz')['test_tokens'][:,1:];base=np.load(OUT/'test_baseline_tokens.npz');ref=base['FP32_nll'].astype(float).mean(-1);refac=(base['FP32_predictions']==truth).mean(-1);loss={};acc={};summary=[];draws={}
    for condition in ['uniform','allocated']:
        raw=[np.load(OUT/'test/runs'/f'{condition}_seed{s}'/'prefix_tokens.npz') for s in range(5)];loss[condition]=np.stack([v['prefix24_nll'].astype(float).mean(-1) for v in raw]);acc[condition]=np.stack([(v['prefix24_predictions']==truth).mean(-1) for v in raw]);ce=bootstrap_paired_difference(loss[condition],np.broadcast_to(ref,loss[condition].shape),seed=p['bootstrap_seed']);ac=bootstrap_paired_difference(acc[condition],np.broadcast_to(refac,acc[condition].shape),seed=p['bootstrap_seed']);draws[condition+'_CE_vs_FP32']=ce;draws[condition+'_accuracy_vs_FP32']=ac;delta=float((loss[condition]-ref).mean());ad=float((acc[condition]-refac).mean());upper=float(np.quantile(ce,.975));lower=float(np.quantile(ac,.025));point=delta<=np.log(1.05) and ad>=-.01
        summary.append({'condition':condition,'cross_entropy':float(loss[condition].mean()),'perplexity':float(np.exp(loss[condition].mean())),'relative_perplexity':float(np.exp(delta)),'token_accuracy':float(acc[condition].mean()),'accuracy_difference':ad,'point_passed':bool(point),'bootstrap_relative_upper':float(np.exp(upper)),'bootstrap_accuracy_lower':lower,'passes':bool(point and upper<=np.log(1.05) and lower>=-.01)})
    ce=bootstrap_paired_difference(loss['allocated'],loss['uniform'],seed=p['bootstrap_seed']);ac=bootstrap_paired_difference(acc['allocated'],acc['uniform'],seed=p['bootstrap_seed']);draws['paired_CE']=ce;draws['paired_accuracy']=ac;pair={'contrast':'allocated-minus-uniform','CE_difference':float((loss['allocated']-loss['uniform']).mean()),'perplexity_ratio':float(np.exp((loss['allocated']-loss['uniform']).mean())),'accuracy_difference':float((acc['allocated']-acc['uniform']).mean()),'CE_sensitivity_low':float(np.quantile(ce,.025)),'CE_sensitivity_high':float(np.quantile(ce,.975)),'accuracy_sensitivity_low':float(np.quantile(ac,.025)),'accuracy_sensitivity_high':float(np.quantile(ac,.975))};write_csv(OUT/'test_summary.csv',summary);save_json(OUT/'paired_summary.json',pair);np.savez_compressed(OUT/'test_quality_draws.npz',**draws);return summary,pair
if __name__=='__main__':print(json.dumps(main(),indent=2))
