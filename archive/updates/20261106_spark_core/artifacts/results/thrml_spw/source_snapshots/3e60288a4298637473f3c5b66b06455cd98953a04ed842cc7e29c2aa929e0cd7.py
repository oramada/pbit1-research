"""Synthetic full-binding fixtures for new high-T five-seed statistics."""
from pathlib import Path
import copy,json,math,os,tempfile
import numpy as np
from .r1_io_v1 import sha
from .highT_statistics_v1 import build,cohort
from .verify_highT_statistics_v1 import verify


def main():
    destination=Path('results/thrml_spw/broad_verification_20261105/highT_preparation_v1/statistics_preflight.json').resolve()
    original=Path.cwd();passed=[]
    with tempfile.TemporaryDirectory(prefix='pbit1-highT-statistics-')as tmp:
        os.chdir(tmp)
        try:
            verifier_old='research/thrml_spw/broad/verify_r1_case_v1.py';verifier_new='research/thrml_spw/broad/verify_highT_case_v1.py'
            for name in [verifier_old,verifier_new]:
                f=Path(name);f.parent.mkdir(parents=True,exist_ok=True);f.write_text('# Synthetic identity fixture only; no neural evidence.\n'+name)
            digest={name:sha(name)for name in [verifier_old,verifier_new]}
            seeds=list(range(202611800,202611805));status='complete_R1_execution_and_statistics_frozen_before_scoring'
            controls=['original','mean1','mean4','mean8','int2','int4','int8']
            parent={'status':status,'output':'parent','sources_sha256':{verifier_old:digest[verifier_old]},'inputs_sha256':{},
                'cases':{'gpt2__'+arm:{'model_key':'gpt2','mode':'control','arm':arm}for arm in controls}}
            Path('parent_protocol.json').write_text(json.dumps(parent))
            p={'status':status,'output':'new','sources_sha256':{verifier_new:digest[verifier_new]},'inputs_sha256':{},
                'control_protocol':'parent_protocol.json','control_protocol_sha256':sha('parent_protocol.json'),
                'control_inputs_sha256':{},'sampling_budget':262144,'logical_pbits_per_scalar':1,
                'models':{'gpt2':{'margin':math.log(1.02),'bootstrap_key':202611810,'seeds_by_budget':{'262144':seeds}}},
                'cases':{f'gpt2__K1_C262144_seed{s}':{'model_key':'gpt2','mode':'native','bits':1,'budget':262144,'samples':262144,'seed':s}for s in seeds}}
            sizes=np.full(62,4568,np.int64);sizes[-1]+=9;assert sizes.sum()==283225
            original_metrics=(2.9+np.linspace(0,.6,62))*sizes
            resources={'logical_pbits':124439808,'logical_retained_device_evaluations':124439808*262144,
                'retained_device_evaluations':130744320*262144,'warmup_device_evaluations':130744320}
            def write(slug,is_control,sums):
                owner=parent if is_control else p;root=Path(owner['output'])/'cases'/slug;root.mkdir(parents=True,exist_ok=True)
                produced={'model_point_sha256':'fixture-point-'+slug}
                (root/'results.json').write_text(json.dumps(produced))
                a={'status':'complete_R1_parameter_counts_resources_and_all_neural_batches_independently_replayed',
                    'results_sha256':sha(root/'results.json'),'source_sha256':digest[verifier_old if is_control else verifier_new],
                    'protocol_sha256':sha('parent_protocol.json'if is_control else'protocol.json'),'case_spec':owner['cases'][slug],
                    'case':slug,'model_key':'gpt2','synthetic_only':False,'unique_parameter_scalars':124439808,
                    'canonical_parameters':148,'model_point_sha256':produced['model_point_sha256'],'evaluation_batches':586,
                    'metric':'NLL','unit_metric_sums':np.asarray(sums).tolist(),'unit_sizes':sizes.tolist(),
                    'resources':{}if is_control else resources}
                (root/'independent_audit.json').write_text(json.dumps(a))
                return root
            for index,arm in enumerate(controls):
                root=write('gpt2__'+arm,True,original_metrics+index*.001*sizes)
                for name in ['results.json','independent_audit.json']:p['control_inputs_sha256'][str(root/name)]=sha(root/name)
            Path('protocol.json').write_text(json.dumps(p))
            for mean_gap,wanted in [(.002,True),(.1,False)]:
                for index,seed in enumerate(seeds):
                    write(f'gpt2__K1_C262144_seed{seed}',False,original_metrics+(mean_gap+(index-2)*.0002+.0001*np.cos(np.arange(62)))*sizes)
                report=build(p,Path('protocol.json'));assert report['quality_gate_met']==wanted;assert verify(p,Path('protocol.json'),report)
                passed.append('passing_complete_binding_cohort'if wanted else'failing_complete_binding_cohort')
            for count in [0,4,6]:
                q=copy.deepcopy(p);q['models']['gpt2']['seeds_by_budget']['262144']=list(range(count))
                try:cohort(q)
                except ValueError:passed.append(f'reject_{count}_seed_cohort')
                else:raise AssertionError('Incomplete/extra cohort accepted')
            for corruption in ['uncertainty_bound','success_flag','input_binding','sampling_budget']:
                altered=copy.deepcopy(report)
                if corruption=='uncertainty_bound':altered['condition']['paired_degradation']['original']['MC95_t4_interval_conditional_on_data'][1]=0
                elif corruption=='success_flag':altered['quality_gate_met']=True
                elif corruption=='input_binding':altered['inputs_sha256'][next(iter(altered['inputs_sha256']))]='0'*64
                else:altered['condition']['samples']=8192
                try:verify(p,Path('protocol.json'),altered)
                except (AssertionError,ValueError):passed.append('reject_'+corruption)
                else:raise AssertionError('Forged report accepted')
        finally:os.chdir(original)
    r={'status':'highT_complete_binding_success_failure_missing_cohort_and_forgery_fixtures_passed','checks':passed,
        'source_sha256':sha(__file__),'producer_sha256':sha(Path(__file__).with_name('highT_statistics_v1.py')),
        'verifier_sha256':sha(Path(__file__).with_name('verify_highT_statistics_v1.py')),
        'scope':'Pure synthetic metric/binding fixtures, not actual modeltaskscoring or neural evidence.'}
    with destination.open('x')as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
