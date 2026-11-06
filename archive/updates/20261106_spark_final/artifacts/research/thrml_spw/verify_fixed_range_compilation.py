"""Check preserved outputs from independently started fixed-compiler processes."""
import json,hashlib,csv
from pathlib import Path
import numpy as np
ROOT=Path('results/thrml_spw/range_localization')
OUT=ROOT/'fixed_compilation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    protocol=json.loads((OUT/'protocol.json').read_text());p=json.loads((ROOT/'protocol.json').read_text());man=[json.loads((OUT/f'replicate{k}/manifest.json').read_text()) for k in [0,1]];original=json.loads((ROOT/'native_evidence_audit.json').read_text());prior={(r['condition'],r['stage']):r for r in original['rows'] if r['seed']==0};rows=[]
    expected={(c,k) for c in protocol['conditions'] for k in protocol['stages']};assert len(expected)==72
    for i,m in enumerate(man):
        assert m['completed'] and m['replicate']==i and m['worker_sha256']==protocol['worker_sha256']==sha('research/thrml_spw/range_fixed_replay_cohort.py') and m['protocol_sha256']==sha(OUT/'protocol.json');assert m['thrml_commit']==protocol['thrml_commit'] and m['XLA_FLAGS']==protocol['XLA_FLAGS'] and m['matmul_precision']=='highest';assert {(r['condition'],r['stage']) for r in m['rows']}==expected and len(m['rows'])==72
    right={(r['condition'],r['stage']):r for r in man[1]['rows']};total=0
    for left in man[0]['rows']:
        key=left['condition'],left['stage'];other=right[key];stage=left['stage'];target=np.load(ROOT/f'anchors/stage{stage}_target.npy');cond=next(c for c in p['native_conditions'] if c['name']==left['condition']);ctrl=p['control_artifacts'][str(cond['group'])][stage];values=[]
        for r in [left,other]:
            assert sha(r['output'])==r['output_sha256'] and r['target_sha256']==sha(ROOT/f'anchors/stage{stage}_target.npy') and r['inputs_sha256']==sha(ROOT/f'anchors/stage{stage}_inputs.npy') and r['controls_sha256']==ctrl['sha256']==sha(ctrl['path']);y=np.load(r['output']);assert y.dtype==np.float32 and y.shape==target.shape and np.isfinite(y).all();mse=float(np.mean((y.astype(float)-target)**2));np.testing.assert_allclose(mse,r['target_MSE'],rtol=1e-13,atol=1e-15);assert r['integer_counts_sha256']==prior[key]['integer_counts_sha256'] and r['public_integer_counts_match_prior'];assert r['native_node_updates']==3*cond['samples']*ctrl['weights'];total+=r['native_node_updates'];values.append(y)
            if not r['public_sum_exact']:
                yp=Path(r['output']);pub=np.load(yp.with_name(yp.stem+'_public.npy'));assert not np.array_equal(pub,y);assert float(np.max(np.abs(pub.astype(float)-y.astype(float))))==r['max_public_sum_difference']
            else:assert r['max_public_sum_difference']==0
        same=bool(np.array_equal(*values));np.testing.assert_allclose(left['old_target_MSE'],float(np.mean((np.load(ROOT/'probes'/f'{left["condition"]}_stage{stage}_seed0/output.npy').astype(float)-target)**2)),rtol=1e-13,atol=1e-15)
        rows.append({'stage':stage,'condition':left['condition'],'cross_process_exact':same,'cross_process_max_difference':float(np.max(np.abs(values[0].astype(float)-values[1].astype(float)))),'both_public_sums_exact':left['public_sum_exact'] and other['public_sum_exact'],'count_hashes_match_prior':True,'relative_MSE_change_from_original':left['target_MSE']/left['old_target_MSE']-1})
    diag=ROOT/'reproducibility_diagnostic';diagnostics=[json.loads((diag/name/'result.json').read_text()) for name in ['cold','prefix','autotune0_rep0/cold','autotune0_rep1/cold']];assert all(d['production_public_integer_counts_exact'] and d['same_executable_repeat']['exact'] for d in diagnostics);assert len({d['integer_counts_sha256'] for d in diagnostics})==1
    diagnostic_updates=sum(d['native_observed_updates'] for d in diagnostics);result={'completed':True,'passed':all(r['cross_process_exact'] and r['both_public_sums_exact'] for r in rows),'cases':72,'processes':2,'all_cross_process_exact':all(r['cross_process_exact'] for r in rows),'all_public_sums_exact':all(r['both_public_sums_exact'] for r in rows),'all_public_counts_match_previous':True,'cohort_native_updates':total,'diagnostic_native_updates':diagnostic_updates,'this_study_native_updates':total+diagnostic_updates,'max_abs_relative_MSE_change_from_original':max(abs(r['relative_MSE_change_from_original']) for r in rows),'source_sha256':sha(__file__),'rows':rows,'scope':'Numerical reproducibility check at seed0, all24 projections/three conditions. Separate prospective compiler configuration; original360-case default-compiler gate remains failed. No heldout full stochastic neural quality or physical-performance claim.'};(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
