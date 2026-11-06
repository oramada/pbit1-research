"""Nontrivial synthetic article/MC covariance and forged-report checks."""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .holdout_native_statistics import summarize,STAGES
from .verify_holdout_native_statistics import verify_summary


def main():
    root=Path('results/thrml_spw/lfm3_1p7b');out=root/'holdout_statistics_auditor_preflight'
    out.mkdir(exist_ok=False)
    rng=np.random.default_rng(202610704)
    cs=np.repeat(rng.uniform(3,5,(1,96,64)),5,0)
    cl=np.repeat(rng.uniform(3,5,(1,16,256)),5,0)
    sets={'BF16':(cs+.1,cl+.1),'same_fit_BF16':(cs,cl),'nominal_controls':(cs-.01,cl-.01)}
    shock=np.linspace(-.025,.025,5)
    sa=rng.normal(0,.06,(24,));la=rng.normal(0,.08,(16,))
    sets['native']=(cs+.005+shock[:,None,None]+np.repeat(sa,4)[None,:,None]+rng.normal(0,.002,cs.shape),
        cl+.005+shock[:,None,None]+la[None,:,None]+rng.normal(0,.002,cl.shape))
    parameters={'NLL_margin':math.log(1.02),'bootstrap_replicates':10000,'bootstrap_key':202610400}
    report=summarize(sets,parameters);assert verify_summary(report,sets,parameters)
    rejected=[]
    for name in ['mean','MC_interval','bootstrap_interval','pooled_weights','ratio','gate','seed_vector','nan']:
        bad=copy.deepcopy(report);record=bad['paired_contrasts']['same_fit_BF16']['contrasts']['pooled']
        if name=='mean':record['mean_NLL_gap']+=.1
        elif name=='MC_interval':record['MC95_t4_interval_conditional_on_articles'][1]+=.1
        elif name=='bootstrap_interval':record['stratified_crossed_bootstrap_percentile95_interval'][1]+=.1
        elif name=='pooled_weights':bad['paired_contrasts']['same_fit_BF16']['pooled_target_weights']=[.9,.1]
        elif name=='ratio':record['PPL_ratio_exp_mean_gap']+=.1
        elif name=='gate':bad['quality_tolerance_met_in_both_strata_and_pooled']=not bad['quality_tolerance_met_in_both_strata_and_pooled']
        elif name=='seed_vector':record['per_seed_mean_NLL_gap'].pop()
        else:record['mean_NLL_gap']=float('nan')
        try:verify_summary(bad,sets,parameters)
        except (AssertionError,ValueError):rejected.append(name)
        else:raise AssertionError('Forged summary accepted')
    with (out/'synthetic_inputs.npz').open('xb') as f:
        np.savez_compressed(f,**{stage+'_'+label:v for stage,arrays in sets.items() for label,v in zip(['short','long'],arrays)})
    (out/'synthetic_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    names=['holdout_native_statistics.py','verify_holdout_native_statistics.py','preflight_verify_holdout_statistics.py','holdout_uncertainty.py','holdout_batch_layout.py']
    result={'status':'independent_frequency_bootstrap_MC_checks_and_eight_forged_reports_verified',
        'sources_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names},
        'parameters':parameters,'forged_reports_rejected':rejected,
        'scope':'Synthetic complete five-seed/40-article geometry, shared seed shocks, article and token variation. No actual heldout outcomes or model scoring; no interval coverage guarantee.'}
    with (out/'results.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
