"""Read-only Bernoulli moment and parameter-noise audit of completed primaries.

No additional neural outputs or sampling. An8sigma aggregate moment gate is
fixed before inspection to distinguish sampler faults from noise sensitivity.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from .r1_io_v1 import BASE,sha,program


def main():
    protocol=BASE/'r1_execution_v1/protocol.json';p=json.loads(protocol.read_text())
    reports=[];begin=time.monotonic()
    for model,spec in p['models'].items():
        package=json.loads(Path(spec['package']).read_text());seeds=spec['seeds_by_budget']['8192'];samples=8192
        roots=[Path(p['output'])/'cases'/f'{model}__K1_C8192_seed{s}' for s in seeds]
        for root in roots:
            a=json.loads((root/'independent_audit.json').read_text());r=root/'results.json'
            assert a['status']=='complete_R1_parameter_counts_resources_and_all_neural_batches_independently_replayed'
            assert a['results_sha256']==sha(r) and a['protocol_sha256']==sha(protocol)
        parameters=[];total_pred=total_actual=total_var=0.;n_total=0
        for record in package['canonical_parameters']:
            assert sha(record['program'])==record['program_sha256'];q=program(record['program']);n=record['elements']
            h=q['codes'].astype(np.float32)*np.float32(8/4095)-np.float32(4)
            prob=1/(1+np.exp(-2*h.astype(np.float64)))
            span=np.repeat(q['span'],128)[:n].astype(np.float64)
            low=np.repeat(q['lower'],128)[:n].astype(np.float64)
            target=low+span*prob;v=prob*(1-prob)/samples
            predicted=float(np.sum(span**2*v))
            variance_sse=float(np.sum(span**4*(2*v*v+prob*(1-prob)*(1-6*prob*(1-prob))/(samples**3))))
            observed=[]
            for root in roots:
                checkpoint=json.loads((root/f'parameter_{record["index"]:05d}.json').read_text())
                path=root/checkpoint['record']['output']['path'];assert sha(path)==checkpoint['record']['output']['sha256']
                with np.load(path,allow_pickle=False) as a:
                    assert int(a['samples'])==samples and int(a['bits'])==1 and int(a['index'])==record['index']
                    c=a['counts'].astype(np.float64).reshape(-1)
                observed.append(float(np.sum((span*(c/samples-prob))**2)))
            actual=math.fsum(observed)/5
            ratios=np.sqrt(span**2*v)/np.maximum(np.abs(target),1e-8)
            parameters.append({'name':record['name'],'elements':n,'shape':record['shape'],
                'predicted_sum_weight_noise_variance':predicted,'observed_mean_sum_squared_noise':actual,
                'observed_to_predicted_MSE':actual/predicted if predicted else None,
                'predicted_relative_RMSE':math.sqrt(predicted/max(float(np.sum(target*target)),1e-30)),
                'fraction_coordinates_noise_std_exceeds10percent_of_abs_mean':float(np.mean(ratios>.1)),
                'max_group_span':float(q['span'].max()),'expected_noise_std_max':float(np.max(np.sqrt(span**2*v)))})
            total_pred+=predicted;total_actual+=actual;total_var+=variance_sse/5;n_total+=n
        z=(total_actual-total_pred)/math.sqrt(total_var)
        reports.append({'model':model,'scalars':n_total,'seeds':seeds,
            'predicted_total_squared_noise':total_pred,'observed_mean_total_squared_noise':total_actual,
            'MSE_ratio':total_actual/total_pred,'aggregate_8sigma_moment_z':z,'moment_gate_met':abs(z)<=8,
            'parameters':parameters})
    out=BASE/'primary_count_moments_v1';out.mkdir(exist_ok=False)
    result={'status':'completed_primary_count_moments_and_parameter_noise_profile',
        'source_sha256':sha(__file__),'protocol_sha256':sha(protocol),'models':reports,
        'all_aggregate_moment_gates_met':all(r['moment_gate_met'] for r in reports),
        'elapsed_seconds':time.monotonic()-begin,
        'scope':'Read-only completed5seedcountfiles, exact Bernoulli second/fourth moments and independentcoordinate assumption. Profiles show numerical perturbation, not causal layer sensitivity or physicaldevice characterization.'}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**{k:v for k,v in result.items() if k!='models'},'models':[{k:v for k,v in r.items() if k!='parameters'} for r in reports]}))

if __name__=='__main__':main()
