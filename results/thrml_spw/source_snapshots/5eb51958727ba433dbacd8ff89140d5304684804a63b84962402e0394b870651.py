"""Final evidence audit; does not execute or alter scientific producers."""
from pathlib import Path
import hashlib,json,re,zipfile,xml.etree.ElementTree as ET

R=Path('results/thrml_spw/spark_depth_20261106')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 old=read(R/'goal_requirements_review_v1.json');rows=old['requirements']
 additions={
 'conditional_error_covariance':['lfm3_error_covariance_v2/results.json','lfm3_error_covariance_v2/covariances.npz','lfm3_error_covariance_v2_audit.json'],
 'sampling_allocation':['reduced_work_gpu_v1_audit.json','reduced_work_replay_v1/results.json'],
 'SDK_and_installability':['sdk_program_verification_v2.json','sdk_build_verification_v2.json','sdk_distribution_v2_tests.xml','sdk_distribution_v2_cli.json'],
 'resource_fairness_and_limits':['resource_ledger_final_v1.json'],
 'K1_fixes_and_cross_architecture_quality':['benchmark_audit_v1/results.json','benchmark_replay_v1/all_results.json','distilbert_benchmark_audit_v1/results.json','distilbert_replay_v1/results.json'],
 }
 notes={
 'conditional_error_covariance':'All 48 conditional covariance matrices checked against saved sufficient statistics and original trace diagonals; five training draws and two 64-channel subsets per block. Not a full hidden-state covariance or held-out claim.',
 'sampling_allocation':'Full 283225-target five-seed evaluation and independent all-six-point SDK/eighteen-window neural replay. 41.7% fewer nominal reconstruction observations, not measured inference speed or energy.',
 'SDK_and_installability':'Final 0.1.0a2.dev1 wheel installed separately; 23 tests, zero skips and native THRML CLI passed. Wheel runtime source bytes and artifact hashes rechecked.',
 'archive_and_final_report':'Consolidated report and extended local archive verified. Large raw arrays omitted with hashes; checkpoints/datasets restored under their licenses. Not published.',
 }
 for row in rows:
  paths=[e['path']for e in row['evidence']]+[str(R/p)for p in additions.get(row['requirement'],[])]
  if row['requirement']=='archive_and_final_report':paths+=['SPARK_RESEARCH.md','publish/pbit1-experiments/updates/20261106_spark_final/verify_archive.py']
  row['evidence']=[]
  for p in dict.fromkeys(paths):
   assert Path(p).is_file(),p
   row['evidence'].append({'path':p,'sha256':sha(p),'exists':True})
  row['assessment']='verified_with_scope'
  if row['requirement']in notes:row['scope_note']=notes[row['requirement']]
 summary=read(R/'cross_model_summary_v2/results.json')
 assert len(summary['models'])==3 and summary['all_three_tested_models_pass']
 for m in summary['models']:
  assert m['seed_count']==5 and m['logical_pbits_per_learned_parameter']==1
  assert m['MC95_perplexity_ratio_interval'][1]<=1.02
 ledger=read(R/'resource_ledger_final_v1.json')
 for row,folder in zip(ledger['rows'],['benchmark_v1','distilbert_benchmark_v1','lfm3_benchmark_v2','reduced_work_gpu_v1']):
  d=read(R/folder/'results.json');case=next(x for x in d['cases']if x['case']['kind']in ['selected','reduced'])
  pf=R/folder/case['case']['name']/'parameters.json';p=read(pf);records=p.get('records',p.get('parameters'))
  assert sha(pf)==row['source_parameter_sha256']
  assert sum(x['parameters']for x in records)==row['unique_parameters']
  assert sum(x['range_groups']for x in records)*8==row['FP32_range_metadata_bytes']
  assert sum(x['parameters']*x['samples']for x in records)==row['nominal_independent_Bernoulli_observations_per_reconstruction']
  assert row['maximum_saved_GPU_allocation_bytes']<=2*1024**3
 build=read(R/'sdk_build_verification_v2.json')
 for a in build['artifacts']:assert sha(a['path'])==a['sha256']
 assert sha('publish/pbit1/src/pbit1/program.py')==build['source_program_sha256']
 suites=ET.parse(R/'sdk_distribution_v2_tests.xml').getroot()
 assert sum(int(s.get('tests',0))for s in suites.iter('testsuite'))==23
 assert all(int(s.get(k,0))==0 for s in suites.iter('testsuite')for k in ['failures','errors','skipped'])
 wheel=R/'sdk_distribution_v2/pbit1-0.1.0a2.dev1-py3-none-any.whl'
 with zipfile.ZipFile(wheel)as z:
  for p in Path('publish/pbit1/src/pbit1').glob('*.py'):assert z.read('pbit1/'+p.name)==p.read_bytes(),p
 links=[]
 for x in re.findall(r'\]\(([^)]+)\)',Path('SPARK_RESEARCH.md').read_text()):
  if not x.startswith(('https:','http:','#')):
   assert Path(x).exists(),x
   links.append(x)
 final=Path('publish/pbit1-experiments/updates/20261106_spark_final')
 manifest=read(final/'FILES.sha256.json')
 assert set(manifest)=={str(p.relative_to(final))for p in final.rglob('*')if p.is_file()and p.name!='FILES.sha256.json'}
 for p,digest in manifest.items():assert sha(final/p)==digest,p
 out={'goal_completion':'verified_research_objective_with_explicit_scope','requirements':rows,'remaining':[],
 'previous_review_sha256':sha(R/'goal_requirements_review_v1.json'),'source_sha256':sha(__file__),
 'checks':{'three_complete_model_cohorts':True,'four_resource_ledgers_recomputed':True,'all_wheel_runtime_modules_match_source':True,'installed_tests':23,'installed_test_skips':0,'report_local_links':len(links),'archive_files_verified_before_final_audit_append':len(manifest)},
 'fine_tuning_decision':old['fine_tuning_decision'],'physical_hardware_boundary':old['physical_hardware_boundary'],
 'scope':'Research investigation complete with success regimes, failure mechanism and fundamental finite-format counterexample. No universal numerical BF16 equivalence, physical hardware efficiency, unseen-task guarantee or release/publication is established. Prior inspected benchmarks are disclosed.'}
 (R/'goal_requirements_review_v2.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out['checks']))
if __name__=='__main__':main()
