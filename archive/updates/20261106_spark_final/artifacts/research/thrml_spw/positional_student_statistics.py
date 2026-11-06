"""Paired five-seed/four-bank diagnostics for matched-update causal students."""
import json
import numpy as np
from .positional_student_common import OUT,sha,guard
from .fine_range_common import atomic_json
from .fine_range_common import atomic_npz

def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);assert json.loads((OUT/'model_audit.json').read_text())['passed'] and json.loads((OUT/'native_audit.json').read_text())['passed'];tokens=np.load(p['tokens_path'])['tokens'];base=np.load(OUT/'baseline_tokens.npz');loss={'FP32':np.broadcast_to(base['nll'],(5,32,64))};acc={'FP32':np.broadcast_to(base['predictions']==tokens[:,1:],(5,32,64))};summaries={};draws={};pairs=[]
    for condition in ['single','positional']:
        loss[condition]=np.stack([np.load(OUT/'runs'/f'{condition}_seed{s}'/'tokens.npz')['nll'] for s in p['sampling_seeds']]);acc[condition]=np.stack([np.load(OUT/'runs'/f'{condition}_seed{s}'/'tokens.npz')['predictions']==tokens[:,1:] for s in p['sampling_seeds']])
    for name in loss:
        ce=float(loss[name].astype(float).mean());summaries[name]={'cross_entropy':ce,'perplexity_exp_mean_CE':float(np.exp(ce)),'token_accuracy':float(acc[name].mean())}
    for index,(left,right) in enumerate([('single','FP32'),('positional','FP32'),('single','positional')]):
        name=left+'_minus_'+right;rng=np.random.default_rng(p['statistics_seed']+index);seed_indices=rng.integers(0,5,size=(10000,5));bank_indices=rng.integers(0,4,size=(10000,4));draws[name+'_seed_indices']=seed_indices;draws[name+'_bank_indices']=bank_indices;result={'contrast':name}
        for metric,data in [('CE',loss),('accuracy',acc)]:
            delta=data[left].astype(float)-data[right].astype(float);bank=delta.reshape(5,4,8,64).mean((2,3));values=bank[seed_indices[:,:,None],bank_indices[:,None,:]].mean((1,2));draws[name+'_'+metric]=values;result[metric]={'difference':float(delta.mean()),'sensitivity_interval':np.quantile(values,[.025,.975]).tolist()}
        pairs.append(result)
    atomic_npz(OUT/'paired_draws.npz',**draws);result={'summaries':summaries,'paired_comparisons':pairs,'paired_scalar_values':60000,'resampling':'10,000 crossed seed/bank draws per contrast; same selected banks for selected seeds. Five MC seeds/four8-context bank clusters, consumed validation only. Unadjusted descriptive sensitivity; not held-out generalization or a hardware confidence bound.','source_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'paired_draws_sha256':sha(OUT/'paired_draws.npz')};atomic_json(OUT/'statistics.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
