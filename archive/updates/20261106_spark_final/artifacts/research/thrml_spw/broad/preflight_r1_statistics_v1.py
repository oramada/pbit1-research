"""Before-score cohort/paired uncertainty fixtures for the complete R1 analysis."""
from pathlib import Path
import copy,hashlib,json,math
import numpy as np
from .r1_statistics_v1 import condition_summary
from .verify_r1_statistics_v1 import verify_interval


def main():
    records=[];retained=None
    for task,sizes,margin,key in [('NLL',np.asarray([7,11,31,5,47,9,21]),math.log(1.02),202611500),
                                 ('classification_error',np.ones(257,np.int64),.02,202611600)]:
        base=4*sizes if task=='NLL' else np.zeros_like(sizes)
        controls={arm:{'unit_sizes':sizes.tolist(),'unit_metric_sums':base.tolist()} for arm in ['original','mean1','mean4','mean8','int2','int4','int8']}
        for name in ['zero','small','degraded']:
            native=[]
            for seed in range(5):
                values=base.astype(np.float64).copy()
                if task=='NLL':values+=({'zero':0,'small':.004,'degraded':.1}[name])*sizes
                elif name=='small':values[seed*7]=1
                elif name=='degraded':values[:80]=1
                native.append({'unit_sizes':sizes.tolist(),'unit_metric_sums':values.tolist(),'resources':{'fixed':1}})
            report=condition_summary(native,controls,margin,key)
            for arm in ['original','mean1']:
                gaps=np.stack([np.asarray(a['unit_metric_sums'])-np.asarray(controls[arm]['unit_metric_sums']) for a in native])
                assert verify_interval(report['paired_degradation'][arm],gaps,sizes,margin,key)
            expected=name!='degraded';assert report['primary_quality_gate']==expected
            records.append({'metric':task,'case':name,'dual_control_gate':expected})
            if name=='small':retained=(native,controls,report,margin,key,sizes)
    native,controls,report,margin,key,sizes=retained
    for length in [0,4,6]:
        bad=(native*2)[:length]
        try:condition_summary(bad,controls,margin,key)
        except ValueError:pass
        else:raise AssertionError('Incomplete/extra MC cohort accepted')
    bad=copy.deepcopy(controls);bad['mean1']['unit_sizes'][0]+=1
    try:condition_summary(native,bad,margin,key)
    except ValueError:pass
    else:raise AssertionError('Changed paired data units accepted')
    gaps=np.stack([np.asarray(a['unit_metric_sums']) for a in native]);forgeries=[]
    for field in ['mean_gap','MC_standard_error','data_units','quality_gate_met']:
        bad=copy.deepcopy(report['paired_degradation']['original'])
        if field=='quality_gate_met':bad[field]=False
        else:bad[field]+=.5
        try:verify_interval(bad,gaps,sizes,margin,key)
        except AssertionError:forgeries.append(field)
        else:raise AssertionError('Forged interval accepted')
    output=Path('results/thrml_spw/broad_verification_20261105/r1_statistics_v1_preflight');output.mkdir(exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('r1_statistics_v1.py'),Path(__file__).with_name('verify_r1_statistics_v1.py')]
    result={'status':'complete_cohorts_dual_control_weighted_bootstrap_oracle_and_forgery_rejections_passed',
        'cases':records,'forgeries_rejected':forgeries,'rejected_cohort_sizes':[0,4,6],'changed_data_units_rejected':True,
        'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'scope':'Synthetic metric sums, no real testtokens or taskquality. Original predeclared margins and5seeds unchanged.'}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
