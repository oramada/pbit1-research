"""Freeze a schedule from audited validation results, without reading test scores."""
import json
from pathlib import Path
import numpy as np
from .sample_allocation_math import allocate
from .transformer_block_gpu import sha,save_json,write_csv
OUT=Path('results/thrml_spw/sample_allocation')
def main():
    if (OUT/'allocation.json').exists():raise RuntimeError('Allocation already frozen')
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'validation_manifest.json').read_text());audit=json.loads((OUT/'validation_evidence_audit.json').read_text());assert audit['passed'] and not m['test_scores_computed'] and audit['validation_manifest_sha256']==sha(OUT/'validation_manifest.json');assert json.loads((OUT/'validation_native_audit.json').read_text())['passed'];assert sha(OUT/'validation_baseline.npz')==m['baseline_archive_sha256'] and m['execution']['protocol_sha256']==sha(OUT/'protocol.json');base=np.load(OUT/'validation_baseline.npz');ref=float(base['mean_nll'].astype(float).mean());coeff=[];rows=[]
    for stage in range(24):
        values=[]
        for seed in range(5):
            case=next(c for c in m['cases'] if c['stage']==stage and c['seed']==seed);assert sha(case['tokens'])==case['tokens_sha256'];values.append(float(np.load(case['tokens'])['nll'].astype(float).mean()))
        delta=float(np.mean(values)-ref);coefficient=max(delta,p['loss_floor']);coeff.append(coefficient);rows.append({'stage':stage,'module':p['targets_in_forward_order'][stage],'mean_isolated_CE':float(np.mean(values)),'mean_reference_CE':ref,'raw_CE_delta':delta,'floored_coefficient':coefficient,'seed_std_CE':float(np.std(values,ddof=1))})
    weights=[c['weights']//p['weight_unit'] for c in p['control_artifacts']];chosen=allocate(coeff,weights,288);samples=[256*t for t in chosen['levels']];cost=sum(c['weights']*t for c,t in zip(p['control_artifacts'],samples));assert cost==42467328*1024
    for row,t in zip(rows,samples):row['selected_samples']=t
    write_csv(OUT/'validation_sensitivities.csv',rows);r={'schema':'frozen-validation-only-allocation-v1','samples':samples,'coefficients':coeff,'weight_units':weights,'objective':chosen['objective'],'uniform_surrogate_objective':float(sum(coeff)/4),'exact_budget_units':chosen['budget_used'],'weighted_updates_per_bank':cost,'uniform_weighted_updates_per_bank':42467328*1024,'weighted_mean_samples':cost/42467328,'sum_samples_across_layers':sum(samples),'max_samples_per_layer':max(samples),'protocol_sha256':sha(OUT/'protocol.json'),'validation_manifest_sha256':sha(OUT/'validation_manifest.json'),'validation_evidence_audit_sha256':sha(OUT/'validation_evidence_audit.json'),'sensitivities_sha256':sha(OUT/'validation_sensitivities.csv'),'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['select_sample_allocation.py','sample_allocation_math.py']},'test_model_scores_used':False,'scope':'Exact optimization of a fixed noisy isolated-layer validation surrogate, not proof of neural improvement. Fresh test comparison and independent MILP audit required.'};save_json(OUT/'allocation.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
