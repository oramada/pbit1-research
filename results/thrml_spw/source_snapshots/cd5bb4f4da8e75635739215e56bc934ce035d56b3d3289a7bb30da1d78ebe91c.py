"""Audit closed benchmark cases from token arrays; summarize only full cohorts."""
from pathlib import Path
import argparse,json,math
import numpy as np
from .prepare_v1 import OUT,sha
from .causal_v1 import digest

def inspect_case(root,case,protocol,tokens,offsets,layout):
    folder=root/case['name'];record=json.loads((folder/'results.json').read_text());parameters=json.loads((folder/'parameters.json').read_text())
    assert record['case']==case and record['point_sha256']==parameters['point_sha256']
    n=0;total=0.;sums=np.zeros(62);counts=np.zeros(62,dtype=np.int64);maximum_allocation=0
    for index,w in enumerate(layout['windows']):
        path=folder/f'window{index:04d}.npz';r=json.loads((folder/f'window{index:04d}.json').read_text())
        begin=int(offsets[w['article']]);ids=tokens[begin+w['context_start']:begin+w['context_stop']][None]
        assert r['window']==w and r['point_sha256']==parameters['point_sha256']and r['input_sha256']==digest(ids)and r['arrays_sha256']==sha(path)
        first=w['target_start']-w['context_start'];last=w['target_stop']-w['context_start']
        with np.load(path,allow_pickle=False)as a:
            loss=a['token_NLL'];target=a['target_ids'];normalizer=a['log_normalizer'];logit=a['target_logit']
            assert np.array_equal(target,ids[0,first:last])and np.isfinite(loss).all()
            assert loss.shape==target.shape==(last-first,);np.testing.assert_array_equal(normalizer-logit,loss)
            total+=float(np.sum(loss,dtype=np.float64));n+=len(loss);sums[w['article']]+=float(loss.sum());counts[w['article']]+=len(loss)
        maximum_allocation=max(maximum_allocation,r['GPU_peak_allocated_bytes'])
    assert n==283225 and np.array_equal(counts,[a['target_count']for a in layout['articles']])
    assert abs(total/n-record['NLL'])<1e-12
    np.testing.assert_allclose(sums,record['article_NLL_sums'],rtol=0,atol=1e-10)
    assert np.array_equal(counts,record['article_target_counts'])
    rows=parameters['records'];cost=None
    package_path=Path('results/thrml_spw/broad_verification_20261105/remote_metadata/additional_parameter_packages_v1/gpt2/package.json')
    canonical=json.loads(package_path.read_text())['canonical_parameters'];shapes={r['name']:r['shape']for r in canonical}
    if case['kind']in ['selected','matched_flat']:
        assert [r['name']for r in rows]==[r['name']for r in canonical]
        assert len(rows)==148 and sum(r['parameters']for r in rows)==124439808
        assert all(r['samples']==32768 for r in rows)
        for r in rows:
            expected='storage_flat'if case['kind']=='matched_flat'else protocol['preferences'].get(r['name'],'storage_flat')
            assert r['parameters']==int(np.prod(shapes[r['name']]))
            if len(shapes[r['name']])!=2:expected='storage_flat'
            assert r['layout']==expected
        observations=sum(r['parameters']*r['samples']for r in rows);assert observations==4077643628544
        variance=sum(r['variance_of_SSE']for r in rows);error=sum(r['SSE_about_finite_control_mean_before_FP32_BF16_rounding']-r['expected_SSE']for r in rows);assert abs(error/math.sqrt(variance))<8
        cost={'logical_pbits':124439808,'observations_per_model_reconstruction':observations,'ideal_control_range_state_policy_bits_per_parameter':13+(64*sum(r['range_groups']for r in rows)+17*148)/124439808}
    else:assert rows==[]
    return {'case':case,'NLL':total/n,'PPL':math.exp(total/n),'article_NLL_sums':sums.tolist(),'article_target_counts':counts.tolist(),'targets':n,'cost':cost,'maximum_GPU_allocation_bytes':maximum_allocation,
        'case_results_sha256':sha(folder/'results.json'),'parameters_sha256':sha(folder/'parameters.json'),
        'scope':'Checks savedtokenloss arithmetic, inputidentity, coverage, cost and momentledgers. Doesnotindependentlyrecomputeallvocabularylogsumexp orneuralforwards.'}

def main():
    args=argparse.ArgumentParser();args.add_argument('--case');args=args.parse_args();root=OUT/'benchmark_v1';p=json.loads((root/'protocol.json').read_text())
    assert sha(__import__('pathlib').Path('research/thrml_spw/spark_depth/benchmark_v1.py'))==p['source_sha256']
    assert sha(p['data']['tokens'])==p['data']['tokens_sha256']and sha(p['data']['layout'])==p['data']['layout_sha256']
    with np.load(p['data']['tokens'],allow_pickle=False)as f:tokens=f['tokens'].copy();offsets=f['offsets'].copy()
    layout=json.loads(Path(p['data']['layout']).read_text());dest=OUT/'benchmark_audit_v1';dest.mkdir(exist_ok=True)
    cases=[x for x in p['cases']if x['name']==args.case]if args.case else p['cases'];assert cases
    reports=[]
    for case in cases:
        report=inspect_case(root,case,p,tokens,offsets,layout);reports.append(report)
        output=dest/(case['name']+'.json')
        if output.exists():assert json.loads(output.read_text())==report
        else:output.write_text(json.dumps(report,indent=2)+'\n')
    if args.case:
        print(json.dumps({'case':args.case,'status':'closed_case_saved_arrays_audited','PPL':reports[0]['PPL']}));return
    all_results=json.loads((root/'results.json').read_text());assert all_results['protocol_sha256']==sha(root/'protocol.json')and len(reports)==12
    baseline=reports[0];target_counts=np.array(baseline['article_target_counts']);base=np.array(baseline['article_NLL_sums']);rng=np.random.default_rng(2026110899)
    article_weights=rng.multinomial(62,np.full(62,1/62),size=10000);seed_weights=rng.multinomial(5,np.full(5,.2),size=10000)/5;denom=article_weights@target_counts
    stats={};arrays={}
    for kind in ['selected','matched_flat']:
        chosen=[r for r in reports if r['case']['kind']==kind];assert len(chosen)==5
        chosen.sort(key=lambda r:r['case']['seed']);delta=np.array([r['NLL']-baseline['NLL']for r in chosen]);half=2.7764451051977987*delta.std(ddof=1)/np.sqrt(5)
        raw=np.array([r['article_NLL_sums']for r in chosen]);arrays[kind]=raw
        boot=np.einsum('bs,ba,sa->b',seed_weights,article_weights,raw-base,optimize=True)/denom
        interval=np.exp([delta.mean()-half,delta.mean()+half]);primary=all_results['statistics'][kind]
        np.testing.assert_allclose(interval,primary['MC95_t_PPL_ratio_interval'],rtol=0,atol=1e-12)
        stats[kind]={'PPL_ratio':float(np.exp(delta.mean())),'MC95_t_PPL_ratio_interval':interval.tolist(),'primary_2pct_pass':bool(delta.mean()+half<=np.log(1.02)),
            'secondary_article_and_seed_bootstrap_PPL_ratio_interval':np.exp(np.quantile(boot,[.025,.975])).tolist()}
    paired=np.einsum('bs,ba,sa->b',seed_weights,article_weights,arrays['selected']-arrays['matched_flat'],optimize=True)/denom
    result={'status':'all12cases_saved_array_coverage_and_statistics_audited','source_sha256':sha(__file__),'benchmark_results_sha256':sha(root/'results.json'),'protocol_sha256':sha(root/'protocol.json'),
        'cases':reports,'statistics':stats,'selected_over_matched_flat_joint_bootstrap_PPL_interval':np.exp(np.quantile(paired,[.025,.975])).tolist(),'scope':'Fullsavedobservationaudit. Independentneuralreplayandnativefullmodelvalidation remainseparate.'}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
