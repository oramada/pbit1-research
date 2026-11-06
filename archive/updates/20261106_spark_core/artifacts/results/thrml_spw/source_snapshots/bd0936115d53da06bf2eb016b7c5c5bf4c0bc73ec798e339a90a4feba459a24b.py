"""Recompute CPU calibration comparisons and validate all variable-T ledgers."""
import json,math
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    root=OUT/'cpu_vector_validation_v1';r=json.loads((root/'results.json').read_text());p=json.loads((root/'protocol.json').read_text());assert len(r['records'])==256
    with np.load(OUT/'training_diagnostic_windows.npz')as f:ids=f['input_ids'][:16,:257]
    values={}
    for row in r['records']:
        condition=row['profile']if row['seed']is None else f"{row['profile']}_seed{row['seed']}"
        loss=np.load(root/f"{condition}_row{row['row']}.npy");assert loss.shape==(256,)and np.isfinite(loss).all();assert abs(loss.mean()-row['mean_NLL'])<1e-12
        import hashlib
        assert hashlib.sha256(ids[row['row']].tobytes()).hexdigest()==row['input_sha256'];values[row['profile'],row['seed'],row['row']]=float(loss.mean())
    summary={}
    for name,policy in p['policies'].items():
        for seed in p['seeds']:
            parameters=json.loads((root/f'{name}_seed{seed}_parameters.json').read_text());assert len(parameters)==148
            assert sum(x['parameters']for x in parameters)==124439808
            for x in parameters:
                assert x['samples']==policy[x['name']]['samples'];assert x['group']==policy[x['name']].get('group',128);assert x['samples']<=32768;assert x['counter_dtype']==('uint16'if x['samples']<=65535 else'uint32');assert x['max_count']<=x['samples']
            cost=sum(x['parameters']*x['samples']for x in parameters);assert cost==r['costs'][name]
            variance=sum(x['variance_of_SSE']for x in parameters);error=sum(x['SSE_about_mean']-x['expected_SSE']for x in parameters);assert abs(error/math.sqrt(variance))<8
        baseline=np.mean([values['baseline',None,row]for row in range(8,16)])
        means=np.array([np.mean([values[name,seed,row]for row in range(8,16)])for seed in p['seeds']]);uniform=np.array([np.mean([values['uniform32768',seed,row]for row in range(8,16)])for seed in p['seeds']]);delta=means-uniform;half=2.7764451051977987*delta.std(ddof=1)/math.sqrt(5)
        ci=np.exp([delta.mean()-half,delta.mean()+half]);np.testing.assert_allclose(ci,r['summaries']['calibration'][name]['paired_MC95_t_PPL_ratio_to_uniform'],atol=1e-12,rtol=0)
        summary[name]={'ratio_to_CPU_baseline':float(np.exp(means.mean()-baseline)),'paired_ratio_to_uniform':float(np.exp(delta.mean())),'paired_MC95_ratio_interval':ci.tolist(),'observation_cost_ratio':cost/r['costs']['uniform32768']}
    out={'status':'256_loss_arrays_15_variable_T_points_and_paired_statistics_verified','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),'calibration':summary,'scope':p['scope'],'range_metadata_bits':{name:64*sum(x['range_groups']for x in json.loads((root/f'{name}_seed{p["seeds"][0]}_parameters.json').read_text()))for name in p['policies']}}
    (OUT/'cpu_vector_audit_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
