"""Independent MILP optimum and validation-only coefficient verification (CPU)."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
OUT=Path('results/thrml_spw/sample_allocation')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads((OUT/'protocol.json').read_text());a=json.loads((OUT/'allocation.json').read_text());m=json.loads((OUT/'validation_manifest.json').read_text());assert json.loads((OUT/'validation_evidence_audit.json').read_text())['passed'] and not a['test_model_scores_used'];assert a['protocol_sha256']==sha(OUT/'protocol.json') and a['validation_manifest_sha256']==sha(OUT/'validation_manifest.json') and a['validation_evidence_audit_sha256']==sha(OUT/'validation_evidence_audit.json');assert sha(OUT/'validation_baseline.npz')==m['baseline_archive_sha256'] and m['execution']['protocol_sha256']==sha(OUT/'protocol.json');base=float(np.load(OUT/'validation_baseline.npz')['mean_nll'].astype(float).mean());coeff=[]
    for stage in range(24):
        loss=[]
        for seed in range(5):
            r=next(v for v in m['cases'] if v['stage']==stage and v['seed']==seed);assert sha(r['tokens'])==r['tokens_sha256'];loss.append(np.load(r['tokens'])['nll'].astype(float).mean())
        coeff.append(max(float(np.mean(loss)-base),1e-4))
    np.testing.assert_array_equal(coeff,a['coefficients']);units=np.array([c['weights']//589824 for c in p['control_artifacts']]);levels=np.array([1,2,4,8,16]);sample=np.array(a['samples']);assert all(t in [256,512,1024,2048,4096] for t in sample) and int(np.dot(units,sample//256))==288 and int(np.dot([c['weights'] for c in p['control_artifacts']],sample))==43486543872
    objective=(np.array(coeff)[:,None]/levels).reshape(-1);matrix=np.zeros((25,120))
    for stage in range(24):matrix[stage,stage*5:stage*5+5]=1;matrix[24,stage*5:stage*5+5]=units[stage]*levels
    target=np.r_[np.ones(24),288];result=milp(objective*1e8,integrality=np.ones(120),bounds=Bounds(np.zeros(120),np.ones(120)),constraints=LinearConstraint(matrix,target,target),options={'mip_rel_gap':0.});assert result.success and result.mip_gap==0;best=float(result.fun/1e8);actual=float(np.sum(np.array(coeff)/(sample/256)));np.testing.assert_allclose(actual,best,atol=1e-11,rtol=1e-10);np.testing.assert_allclose(a['objective'],actual,atol=1e-14,rtol=1e-12)
    for name,h in a['source_sha256'].items():assert sha(name)==h
    r={'passed':True,'allocation_sha256':sha(OUT/'allocation.json'),'all_coefficients_independently_recomputed_from_validation_losses':True,'independent_solver':'scipy.optimize.milp / HiGHS, binary choices, exact budget, zero relative gap','solver_mip_gap':float(result.mip_gap),'independent_optimal_objective':best,'selected_objective':actual,'weighted_updates_per_bank':43486543872,'no_test_scores_used':True,'source_sha256':sha(__file__),'scope':'Global optimum of this fixed validation surrogate only. Lexicographic tie behavior separately covered by exhaustive small-case tests. No statement about fresh model quality or physical energy.'};(OUT/'selection_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
