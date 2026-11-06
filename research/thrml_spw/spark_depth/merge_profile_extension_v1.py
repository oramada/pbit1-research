"""Ten-seed calibration selection with old failed five-seed records preserved."""
import json,math
import numpy as np
from .prepare_v1 import OUT,sha

def main():
    parent_dir=OUT/'profile_calibration_v1';extra_dir=OUT/'profile_extension_v1';a=json.loads((parent_dir/'results.json').read_text());b=json.loads((extra_dir/'results.json').read_text());p=json.loads((extra_dir/'protocol.json').read_text())
    assert p['parent_five_seed_result_sha256']==sha(parent_dir/'results.json')
    assert len(b['records'])==(2+2*5)*16
    for result,folder in [(a,parent_dir),(b,extra_dir)]:
        for r in result['records']:
            loss=np.load(folder/f"{r['condition']}_row{r['row']}_NLL.npy",allow_pickle=False)
            assert loss.shape==(256,)and np.isfinite(loss).all()and abs(loss.mean()-r['metrics']['mean_NLL'])<1e-12
    baseline={r['row']:r for r in a['records']if r['profile']=='BF16'}
    for r in b['records']:
        if r['profile']=='BF16':assert r['logits_sha256']==baseline[r['row']]['logits_sha256']
    summaries={};passing=[]
    for profile in ['operator_body_T32768','operator_all_T32768']:
        records=[r for result in[a,b]for r in result['records']if r['profile']==profile and r['split']=='calibration']
        seeds=sorted({r['seed']for r in records});assert len(seeds)==10
        base=np.mean([baseline[i]['metrics']['mean_NLL']for i in range(8,16)])
        means=np.array([np.mean([r['metrics']['mean_NLL']for r in records if r['seed']==seed])for seed in seeds]);delta=means-base;half=2.262157162798205*delta.std(ddof=1)/math.sqrt(10);passes=bool(delta.mean()+half<=math.log(1.02))
        costs=[]
        for result,folder in [(a,parent_dir),(b,extra_dir)]:
            for seed in sorted({r['seed']for r in result['records']if r['profile']==profile}):
                rows=json.loads((folder/f'{profile}_seed{seed}_parameters.json').read_text());assert len(rows)==148
                n=sum(r['parameters']for r in rows);assert n==124439808;obs=sum(r['parameters']*r['samples']for r in rows);costs.append(obs/(n*8192))
                variance=sum(r['variance_of_SSE']for r in rows);error=sum(r['SSE_about_finite_control_mean_before_FP32_BF16_rounding']-r['expected_SSE']for r in rows);assert abs(error/math.sqrt(variance))<8
        assert len(set(costs))==1
        summaries[profile]={'seeds':seeds,'seed_NLL':means.tolist(),'PPL_ratio':float(np.exp(delta.mean())),'MC95_t9_PPL_ratio_interval':np.exp([delta.mean()-half,delta.mean()+half]).tolist(),'qualified':passes,'observation_cost_ratio':costs[0]}
        if passes:passing.append((costs[0],float(delta.mean()),profile))
    selected=min(passing)[2]if passing else None
    combined={'status':'ten_seed_calibration_extension_complete','source_sha256':sha(__file__),'parent_sha256':sha(parent_dir/'results.json'),'extension_sha256':sha(extra_dir/'results.json'),
        'selected_profile':selected,'summaries':summaries,'scope':'Exploratorycalibrationselection afterpreservedfive-seedfailure. Ten-seedCIs onlysamplinguncertainty. Officialbenchmarkunchangedandnotyetrescored. Lowestobservationcost amongthetwoextendedcandidates.'}
    output=OUT/'profile_extension_combined_v1.json';output.write_text(json.dumps(combined,indent=2)+'\n')
    audit={'status':'passed_all544_loss_array_checks_complete_ten_seeds_and_selection','result_sha256':sha(output),'selected_profile':selected,'source_sha256':sha(__file__)}
    (OUT/'profile_extension_combined_audit_v1.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(combined,indent=2))

if __name__=='__main__':main()
