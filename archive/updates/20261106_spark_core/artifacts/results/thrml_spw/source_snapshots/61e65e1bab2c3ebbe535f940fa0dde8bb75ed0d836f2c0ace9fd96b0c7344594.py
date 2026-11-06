"""Independent raw-score, resource and all60,000 resampling-value checks."""
import json
import numpy as np
from .positional_student_common import OUT,sha,guard
from .block_cached_control_worker import atomic_json

def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);r=json.loads((OUT/'statistics.json').read_text());assert r['protocol_sha256']==sha(OUT/'protocol.json') and sha(OUT/'paired_draws.npz')==r['paired_draws_sha256'];truth=np.load(p['tokens_path'])['tokens'][:,1:];base=np.load(OUT/'baseline_tokens.npz');loss={'FP32':np.stack([base['nll']]*5).astype(float)};accuracy={'FP32':np.stack([base['predictions']==truth]*5).astype(float)};updates=0
    for name in ['single','positional']:
        losses=[];correct=[]
        for seed in range(5):
            folder=OUT/'runs'/f'{name}_seed{seed}';case=json.loads((folder/'run.json').read_text());assert sha(folder/'tokens.npz')==case['tokens_sha256'];raw=np.load(folder/'tokens.npz');losses.append(raw['nll'].astype(float));correct.append((raw['predictions']==truth).astype(float));ce=float(losses[-1].mean());assert case['scores']['cross_entropy']==ce and case['scores']['perplexity']==float(np.exp(ce)) and case['scores']['token_accuracy']==float(correct[-1].mean());cfg=next(c for c in p['conditions'] if c['name']==name);assert cfg['K']*cfg['T']==256 and case['native_node_updates']==4*42467328*256;updates+=case['native_node_updates']
        loss[name]=np.stack(losses);accuracy[name]=np.stack(correct)
    assert updates==434865438720
    for name in loss:
        ce=float(loss[name].mean());assert r['summaries'][name]=={'cross_entropy':ce,'perplexity_exp_mean_CE':float(np.exp(ce)),'token_accuracy':float(accuracy[name].mean())}
    raw=np.load(OUT/'paired_draws.npz');checked=0
    for index,(left,right) in enumerate([('single','FP32'),('positional','FP32'),('single','positional')]):
        name=left+'_minus_'+right;entry=next(c for c in r['paired_comparisons'] if c['contrast']==name);rng=np.random.default_rng(p['statistics_seed']+index);seeds=rng.integers(0,5,(10000,5));banks=rng.integers(0,4,(10000,4));np.testing.assert_array_equal(seeds,raw[name+'_seed_indices']);np.testing.assert_array_equal(banks,raw[name+'_bank_indices'])
        for metric,data in [('CE',loss),('accuracy',accuracy)]:
            delta=data[left]-data[right];means=np.array([[np.mean(delta[s,b*8:(b+1)*8,:]) for b in range(4)] for s in range(5)]);values=[]
            for ss,bb in zip(seeds,banks):values.append(float(np.mean([[means[s,b] for b in bb] for s in ss])))
            np.testing.assert_allclose(values,raw[name+'_'+metric],rtol=1e-14,atol=1e-15);np.testing.assert_allclose(entry[metric]['difference'],delta.mean(),rtol=1e-14,atol=1e-15);np.testing.assert_allclose(entry[metric]['sensitivity_interval'],np.quantile(values,[.025,.975]),rtol=1e-14,atol=1e-15);checked+=len(values)
    assert checked==60000;result={'passed':True,'independent_resampling_values_checked':checked,'all_raw_scores_and_update_counts_checked':True,'statistics_sha256':sha(OUT/'statistics.json'),'source_sha256':sha(__file__),'scope':'Consumed validation descriptive uncertainty only. No fresh quality gate, candidate selection, physical or Extropic architecture claim.'};atomic_json(OUT/'statistics_audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
