"""Recompute all profile-selection metrics and sample-cost ledgers from artifacts."""
import json,math
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    root=OUT/'profile_calibration_v1';result=json.loads((root/'results.json').read_text());p=json.loads((root/'protocol.json').read_text());assert result['protocol_sha256']==sha(root/'protocol.json')
    assert len(result['records'])==(2+5*4)*16
    original=json.loads((OUT/'confirm_layout_v1/results.json').read_text());prior={r['row']:r for r in original['records']if r['condition']=='BF16'}
    loss={};checks=0;replays=0
    for r in result['records']:
        a=np.load(root/f"{r['condition']}_row{r['row']}_NLL.npy",allow_pickle=False);assert a.shape==(256,)and np.isfinite(a).all();assert abs(a.mean()-r['metrics']['mean_NLL'])<1e-12
        loss[r['condition'],r['row']]=float(a.mean());checks+=1
        if r['profile']=='BF16'and r['row']>=8:
            assert r['logits_sha256']==prior[r['row']]['logits_sha256'];replays+=1
    passing=[];summaries={}
    for profile in p['profiles']:
        means=np.array([np.mean([loss[f'{profile}_seed{seed}',row]for row in range(8,16)])for seed in p['seeds']]);base=np.mean([loss['BF16',row]for row in range(8,16)]);delta=means-base;half=2.7764451051977987*np.std(delta,ddof=1)/math.sqrt(5);passed=bool(np.mean(delta)+half<=math.log(1.02))
        all_costs=[]
        for seed in p['seeds']:
            records=json.loads((root/f'{profile}_seed{seed}_parameters.json').read_text());assert len(records)==148 and sum(r['parameters']for r in records)==124439808
            n=sum(r['parameters']for r in records);obs=sum(r['samples']*r['parameters']for r in records);cost=obs/n/8192;all_costs.append(cost)
            variance=sum(r['variance_of_SSE']for r in records);error=sum(r['SSE_about_finite_control_mean_before_FP32_BF16_rounding']-r['expected_SSE']for r in records);assert abs(error/math.sqrt(variance))<8
            l=result['ledgers'][f'{profile}_seed{seed}'];assert l['observations']==obs and l['logical_pbits']==124439808
        assert len(set(all_costs))==1
        summaries[profile]={'PPL_ratio':float(np.exp(delta.mean())),'MC95_t_PPL_ratio_interval':np.exp([delta.mean()-half,delta.mean()+half]).tolist(),'qualifies':passed,'observation_cost_ratio':all_costs[0]}
        recorded=result['summaries']['calibration']['profiles'][profile];assert np.allclose(summaries[profile]['MC95_t_PPL_ratio_interval'],recorded['MC95_t_PPL_ratio_interval'],atol=1e-12,rtol=0)
        if passed:passing.append((all_costs[0],float(delta.mean()),profile))
    selected=min(passing)[2]if passing else None;assert selected==result['selected_profile']
    audit={'status':'passed_complete_cohort_loss_ledger_and_selection_recomputation','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),
        'selected_profile':selected,'saved_token_arrays_checked':checks,'prior_BF16_logit_hash_matches':replays,'calibration':summaries,'scope':'Selectiononlyonreusedtrainingcalibration. No benchmarkqualityclaim.'}
    (OUT/'profile_calibration_audit_v1.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
