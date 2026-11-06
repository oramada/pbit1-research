"""Post-hoc IID state-change proxy and pooled calibration diagnostics, no selection."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('results/thrml_spw/probe_placement')


def expected_recorded_changes(p,samples):
    p=np.asarray(p,float)
    if not isinstance(samples,int) or samples<1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Probabilities and positive integer samples required')
    # Each independent native call initializes False. First updated state changes
    # with probability p; consecutive IID draws differ with probability2p(1-p).
    return p+2*(samples-1)*p*(1-p)


def main():
    p=json.loads((ROOT/'protocol.json').read_text());design=json.loads((ROOT/'placement_designs.json').read_text());rows=[];profiles=[]
    for method in p['methods']:
        for seed in range(5):
            totalN=0;wmse=gainmse=offsetmse=predicted=calchanges=infchanges=0.
            for stage,c in enumerate(p['control_artifacts']):
                control=np.load(c['path']);n=control['codes'].size;meta=json.loads((ROOT/'calibration'/f'{method}_stage{stage}_seed{seed}.json').read_text());a=np.load(ROOT/'calibration'/f'{method}_stage{stage}_seed{seed}.npz')
                noise=np.random.default_rng(np.random.SeedSequence([2026116600,stage,seed])).standard_normal((2,n));gain=np.exp(.2*noise[1]-.02);offset=.2*noise[0];pair=meta['probe_codes'];cal=0.
                for code in pair:
                    command=float(np.float32(-4+code*8/255));field=(gain*command+offset).astype(np.float32);q=1/(1+np.exp(-2*field.astype(float)));cal+=float(expected_recorded_changes(q,2048).sum())
                command=(-4+a['corrected_codes'].astype(float)*8/255).astype(np.float32);field=(gain*command.astype(float)+offset).astype(np.float32);q=1/(1+np.exp(-2*field.astype(float)));inc=8*float(expected_recorded_changes(q,1024).sum())
                dd=design[stage]['fixed_baseline' if method=='fixed' else 'chosen'];pred=dd['nominal_first_order_weight_MSE']
                rows.append({'method':method,'seed':seed,'stage':stage,'weights':n,'weight_MSE':meta['realized_weight_MSE_vs_nominal_program'],'gain_RMSE':meta['gain_RMSE'],'offset_RMSE':meta['offset_RMSE'],'nominal_Fisher_predicted_MSE':pred,
                    'analytic_expected_recorded_calibration_changes':cal,'analytic_expected_recorded_inference_changes':inc})
                totalN+=n;wmse+=n*meta['realized_weight_MSE_vs_nominal_program'];gainmse+=n*meta['gain_RMSE']**2;offsetmse+=n*meta['offset_RMSE']**2;predicted+=n*pred;calchanges+=cal;infchanges+=inc
            assert totalN==21233664
            profiles.append({'method':method,'seed':seed,'weight_MSE':wmse/totalN,'gain_RMSE':float(np.sqrt(gainmse/totalN)),'offset_RMSE':float(np.sqrt(offsetmse/totalN)),'nominal_Fisher_predicted_MSE':predicted/totalN,
                'analytic_expected_recorded_calibration_changes':calchanges,'analytic_expected_recorded_inference_changes':infchanges,'calibration_evaluations':4096*totalN,'inference_evaluations':8*1024*totalN})
    summary=[]
    for method in p['methods']:
        r=[r for r in profiles if r['method']==method]
        summary.append({'method':method,'weight_MSE':float(np.mean([a['weight_MSE'] for a in r])),'gain_RMSE':float(np.sqrt(np.mean([a['gain_RMSE']**2 for a in r]))),'offset_RMSE':float(np.sqrt(np.mean([a['offset_RMSE']**2 for a in r]))),'nominal_Fisher_predicted_MSE':float(np.mean([a['nominal_Fisher_predicted_MSE'] for a in r])),
            'analytic_expected_recorded_calibration_changes_per_profile':float(np.mean([a['analytic_expected_recorded_calibration_changes'] for a in r])),'analytic_expected_recorded_inference_changes_per_profile':float(np.mean([a['analytic_expected_recorded_inference_changes'] for a in r])),
            'calibration_evaluations_per_profile':4096*21233664,'inference_evaluations_per_profile':8*1024*21233664})
    for name,data in [('calibration_stage_diagnostics.csv',rows),('calibration_profile_diagnostics.csv',profiles),('calibration_summary.csv',summary)]:
        with (ROOT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    result={'posthoc_diagnostic_only':True,'formula':'p + 2*(T-1)*p*(1-p) per independent False-initialized sample call; inference resets per each8contextbank',
        'scope':'Analytic expected differences between recorded IID simulator states, not measured native flips or physical thermal switching. Physical rates/readout/reset/programming costs not modeled; equal observations do not imply equal energy. No method selection or quality rules changed.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'summary':summary}
    (ROOT/'activity_proxy.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
