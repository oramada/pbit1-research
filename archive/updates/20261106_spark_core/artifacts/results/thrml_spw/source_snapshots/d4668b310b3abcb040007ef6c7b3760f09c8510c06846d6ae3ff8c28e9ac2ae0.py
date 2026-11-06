"""Independent all-pair/seed-summary verification of the operator report tables."""
import csv,json
import numpy as np
from .positional_operator_common import OUT,sha,guard
from .fine_range_common import atomic_json

def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);audit=json.loads((OUT/'evidence_audit.json').read_text());assert audit['passed'];r=json.loads((OUT/'statistics.json').read_text());cases={(v['signature']['stage'],v['signature']['K'],v['signature']['T'],v['signature']['seed']):v for v in r['rows']};pairs=list(csv.DictReader((OUT/'paired_operator_ratios.csv').open()));summary=list(csv.DictReader((OUT/'summary.csv').open()));assert len(cases)==960 and len(pairs)==480 and len(summary)==4;observed_keys=set()
    for row in pairs:
        stage,seed,updates=int(row['stage']),int(row['seed']),int(row['updates_per_weight']);assert (stage,seed,updates) not in observed_keys;observed_keys.add((stage,seed,updates));single=cases[stage,1,updates,seed];multi=cases[stage,4,updates//4,seed]
        for name,value in {'single_T':updates,'positional_T':updates//4,'single_MSE':single['observed_target_MSE'],'positional_MSE':multi['observed_target_MSE'],'positional_over_single_MSE':multi['observed_target_MSE']/single['observed_target_MSE'],'predicted_ratio':multi['predicted_target_MSE']/single['predicted_target_MSE']}.items():np.testing.assert_allclose(float(row[name]),value,rtol=1e-14,atol=1e-15)
    for row in summary:
        updates=int(row['updates_per_weight']);seed_logs=[];theory=[];count=0
        for seed in range(5):seed_logs.append(np.mean([np.log(cases[layer,4,updates//4,seed]['observed_target_MSE'])-np.log(cases[layer,1,updates,seed]['observed_target_MSE']) for layer in range(24)]))
        for layer in range(24):
            multi=[cases[layer,4,updates//4,s] for s in range(5)];single=[cases[layer,1,updates,s] for s in range(5)];count+=np.mean([v['observed_target_MSE'] for v in multi])<np.mean([v['observed_target_MSE'] for v in single]);theory.append(np.log(multi[0]['predicted_target_MSE'])-np.log(single[0]['predicted_target_MSE']))
        mean=float(np.mean(seed_logs));se=float(np.std(seed_logs,ddof=1)/np.sqrt(5));expected={'geometric_MSE_ratio':np.exp(mean),'approx_seed_sensitivity_lower':np.exp(mean-1.96*se),'approx_seed_sensitivity_upper':np.exp(mean+1.96*se),'predicted_geometric_ratio':np.exp(np.mean(theory)),'layers_positional_mean_MSE_lower':count}
        for key,value in expected.items():np.testing.assert_allclose(float(row[key]),value,rtol=1e-13,atol=1e-15)
    assert observed_keys=={(l,s,t) for l in range(24) for s in range(5) for t in [4,16,64,256]};result={'passed':True,'paired_operator_rows_checked':480,'seed_summary_rows_checked':4,'native_audit_sha256':sha(OUT/'evidence_audit.json'),'statistics_sha256':sha(OUT/'statistics.json'),'paired_table_sha256':sha(OUT/'paired_operator_ratios.csv'),'summary_table_sha256':sha(OUT/'summary.csv'),'source_sha256':sha(__file__),'scope':'Independent arithmetic from already audited native MSEs. Approximate seed sensitivity on consumed validation only; not fresh quality or hardware evidence.'};atomic_json(OUT/'summary_audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
