"""Independent frequency-weighted bootstrap and adverse-case decision fixtures."""
from pathlib import Path
import hashlib,json,math,copy
import numpy as np
from .task_statistics_v1 import summarize


def oracle(gaps,sizes,parameters):
    r=np.asarray(gaps,np.float64);w=np.asarray(sizes,np.int64);n=parameters['replicates']
    rng=np.random.default_rng(parameters['key']);seed_draws=rng.integers(0,5,(n,5))
    unit_draws=rng.integers(0,len(w),(n,len(w)))
    sf=np.stack([np.bincount(x,minlength=5) for x in seed_draws])
    uf=np.stack([np.bincount(x,minlength=len(w)) for x in unit_draws])
    draws=np.einsum('bs,su,bu->b',sf,r,uf)/(5*(uf@w))
    means=np.array([math.fsum(row)/int(w.sum()) for row in r]);mean=math.fsum(means)/5
    se=math.sqrt(math.fsum((x-mean)**2 for x in means)/4/5)
    return means,mean,se,[mean-2.7764451051977987*se,mean+2.7764451051977987*se],np.quantile(draws,[.025,.975],method='linear')


def verify(report,gaps,sizes,parameters):
    a,mean,se,mc,boot=oracle(gaps,sizes,parameters)
    for actual,expected in [(report['per_seed_mean_gap'],a),(report['mean_gap'],mean),
        (report['MC_standard_error'],se),(report['MC95_t4_interval_conditional_on_data'],mc),
        (report['crossed_seed_data_bootstrap_percentile95_interval'],boot)]:
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    assert report['scored_observations_per_seed']==int(np.asarray(sizes).sum())
    assert report['data_units']==len(sizes) and report['bootstrap_replicates']==parameters['replicates']
    assert report['bootstrap_key']==parameters['key'] and report['margin']==parameters['margin']
    assert report['MC_upper_within_margin']==(mc[1]<=parameters['margin'])
    assert report['bootstrap_upper_within_margin']==(boot[1]<=parameters['margin'])
    assert report['quality_gate_met']==(mc[1]<=parameters['margin'] and boot[1]<=parameters['margin'])


def main():
    rng=np.random.default_rng(202611940);cases=[];retained=None
    for name,sizes,margin in [('unequal_article_sizes',[7,11,31,5,47,9,21],math.log(1.02)),
                             ('paired_reviews',[1]*31,.02)]:
        sizes=np.asarray(sizes,np.int64)
        for condition,offset,expected in [('zero',0,True),('small',.004,True),('degraded',.1,False)]:
            gaps=np.zeros((5,len(sizes))) if condition=='zero' else (offset+rng.normal(0,.0005,(5,len(sizes))))*sizes
            params={'margin':margin,'replicates':10000,'key':202611941}
            report=summarize(gaps,sizes,margin,10000,params['key'],bootstrap_batch=37)
            verify(report,gaps,sizes,params)
            assert report['quality_gate_met']==expected
            other=summarize(gaps,sizes,margin,10000,params['key'],bootstrap_batch=256)
            assert other==report,'Observer batch size changed frozen random draw plan'
            cases.append({'task':name,'case':condition,'gate':expected})
            if condition=='small':retained=(report,gaps,sizes,params)
    report,gaps,sizes,params=retained;rejected=[]
    for field in ['mean','interval','gate','observations','units']:
        bad=copy.deepcopy(report)
        if field=='mean':bad['mean_gap']+=.1
        elif field=='interval':bad['crossed_seed_data_bootstrap_percentile95_interval'][1]+=.1
        elif field=='gate':bad['quality_gate_met']=False
        elif field=='observations':bad['scored_observations_per_seed']+=1
        else:bad['data_units']+=1
        try:verify(bad,gaps,sizes,params)
        except AssertionError:rejected.append(field)
        else:raise AssertionError('Forged report accepted')
    rejected_inputs=[]
    for name,g,s in [('four_seeds',gaps[:4],sizes),('zero_cluster',gaps,np.zeros_like(sizes)),
                     ('nonfinite',gaps*np.nan,sizes),('mismatched_units',gaps,sizes[:-1])]:
        try:summarize(g,s,.02,10000,202611941)
        except ValueError:rejected_inputs.append(name)
        else:raise AssertionError('Invalid input accepted')
    out=Path('results/thrml_spw/broad_verification_20261105/task_statistics_v1_preflight');out.mkdir(exist_ok=False)
    result={'status':'unequalarticle_and_review_paired_fiveseed_bootstrap_oracles_batchinvariance_and_rejections_passed',
        'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('task_statistics_v1.py')]},
        'cases':cases,'forgeries_rejected':rejected,'invalid_inputs_rejected':rejected_inputs,
        'scope':'Synthetic arithmetic only. Dataunitweights, pairing, marginandseed protocols must be frozen before any real model quality scoring.'}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
