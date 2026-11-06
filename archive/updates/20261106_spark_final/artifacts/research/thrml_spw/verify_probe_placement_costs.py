"""Independent pooled error and post-hoc IID activity-proxy verification."""
import csv,json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha
ROOT=Path('results/thrml_spw/probe_placement')


def read(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))


def main():
    assert json.loads((ROOT/'evidence_audit.json').read_text())['passed'];p=json.loads((ROOT/'protocol.json').read_text());ds=json.loads((ROOT/'placement_designs.json').read_text());stage_rows=read('calibration_stage_diagnostics.csv');profile_rows=read('calibration_profile_diagnostics.csv');summary=read('calibration_summary.csv');assert len(stage_rows)==120 and len(profile_rows)==10 and len(summary)==2
    expected_profiles=[]
    for method in p['methods']:
        for seed in range(5):
            totalN=0;werr=gerr=derr=pred=calchanges=infchanges=0.
            for stage,c in enumerate(p['control_artifacts']):
                a=np.load(c['path']);n=a['codes'].size;cm=json.loads((ROOT/'calibration'/f'{method}_stage{stage}_seed{seed}.json').read_text());cp=np.load(ROOT/'calibration'/f'{method}_stage{stage}_seed{seed}.npz')
                z=np.random.default_rng(np.random.SeedSequence([2026116600,stage,seed])).standard_normal((2,n));gain=np.exp(.2*z[1]-.02);offset=.2*z[0]
                cal=0.
                for code in cm['probe_codes']:
                    b=float(np.float32(-4+code*8/255));h=(gain*b+offset).astype(np.float32);q=1/(1+np.exp(-2*h.astype(float)))
                    cal+=float(np.sum(q)+2*2047*np.sum(q*(1-q)))
                b=(-4+cp['corrected_codes'].astype(float)*8/255).astype(np.float32);h=(gain*b.astype(float)+offset).astype(np.float32);q=1/(1+np.exp(-2*h.astype(float)));inc=8*float(np.sum(q)+2*1023*np.sum(q*(1-q)))
                score=ds[stage]['fixed_baseline' if method=='fixed' else 'chosen']['nominal_first_order_weight_MSE'];r=next(r for r in stage_rows if r['method']==method and int(r['seed'])==seed and int(r['stage'])==stage)
                for key,value in {'weight_MSE':cm['realized_weight_MSE_vs_nominal_program'],'gain_RMSE':cm['gain_RMSE'],'offset_RMSE':cm['offset_RMSE'],'nominal_Fisher_predicted_MSE':score,'analytic_expected_recorded_calibration_changes':cal,'analytic_expected_recorded_inference_changes':inc}.items():np.testing.assert_allclose(float(r[key]),value,atol=1e-12,rtol=1e-12)
                totalN+=n;werr+=n*cm['realized_weight_MSE_vs_nominal_program'];gerr+=n*cm['gain_RMSE']**2;derr+=n*cm['offset_RMSE']**2;pred+=n*score;calchanges+=cal;infchanges+=inc
            vals={'weight_MSE':werr/totalN,'gain_RMSE':np.sqrt(gerr/totalN),'offset_RMSE':np.sqrt(derr/totalN),'nominal_Fisher_predicted_MSE':pred/totalN,'analytic_expected_recorded_calibration_changes':calchanges,'analytic_expected_recorded_inference_changes':infchanges,'calibration_evaluations':4096*totalN,'inference_evaluations':8*1024*totalN}
            r=next(r for r in profile_rows if r['method']==method and int(r['seed'])==seed)
            for k,v in vals.items():np.testing.assert_allclose(float(r[k]),v,atol=1e-12,rtol=1e-12)
            expected_profiles.append({'method':method,'seed':seed,**vals})
    for method in p['methods']:
        rr=[r for r in expected_profiles if r['method']==method];r=next(r for r in summary if r['method']==method)
        vals={'weight_MSE':np.mean([x['weight_MSE'] for x in rr]),'gain_RMSE':np.sqrt(np.mean([x['gain_RMSE']**2 for x in rr])),'offset_RMSE':np.sqrt(np.mean([x['offset_RMSE']**2 for x in rr])),'nominal_Fisher_predicted_MSE':np.mean([x['nominal_Fisher_predicted_MSE'] for x in rr]),
            'analytic_expected_recorded_calibration_changes_per_profile':np.mean([x['analytic_expected_recorded_calibration_changes'] for x in rr]),'analytic_expected_recorded_inference_changes_per_profile':np.mean([x['analytic_expected_recorded_inference_changes'] for x in rr]),'calibration_evaluations_per_profile':4096*21233664,'inference_evaluations_per_profile':8*1024*21233664}
        for k,v in vals.items():np.testing.assert_allclose(float(r[k]),v,atol=1e-12,rtol=1e-12)
    proxy=json.loads((ROOT/'activity_proxy.json').read_text());assert proxy['posthoc_diagnostic_only'] and proxy['source_sha256']==sha('research/thrml_spw/probe_placement_costs.py')
    r={'passed':True,'stage_diagnostics':120,'profile_diagnostics':10,'pooled_methods':2,'expected_recorded_state_change_formula_independently_recomputed':True,'source_sha256':sha(__file__),
        'scope':'Derived pooled error and analytic IID/reset-False state-change proxy only. No actual transition trace, physical switching, energy, rate or time measurement. Post-hoc supplement did not alter choices or primary quality rules.'}
    (ROOT/'cost_evidence_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
