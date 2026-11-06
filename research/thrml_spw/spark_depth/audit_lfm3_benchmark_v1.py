"""Independent saved-output, coverage, resource and statistical audit for LFM3 v2."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from .prepare_v1 import OUT,sha


def coverage(windows,offsets):
 lengths=np.diff(offsets);seen=[np.zeros(int(n),np.uint8)for n in lengths]
 for w in windows:
  a=w['article'];assert 0<=a<len(lengths)
  assert 0<=w['context_start']<w['target_start']<w['target_stop']<=w['context_stop']<=lengths[a]
  assert w['context_stop']-w['context_start']<=1024
  seen[a][w['target_start']:w['target_stop']]+=1
 for v in seen:assert v[0]==0 and np.all(v[1:]==1),'Missing or duplicate target'
 return lengths-1


def primary_statistics(values,baseline):
 values=np.asarray(values,np.float64);assert values.shape==(5,)and np.isfinite(values).all()
 delta=values-baseline;half=2.7764451051977987*np.std(delta,ddof=1)/np.sqrt(5)
 return {'PPL_ratio':float(np.exp(delta.mean())),'MC95_t4_PPL_ratio_interval':np.exp([delta.mean()-half,delta.mean()+half]).tolist(),'pass':bool(delta.mean()+half<=np.log(1.02))}


def main():
 root=OUT/'lfm3_benchmark_v2';r=json.loads((root/'results.json').read_text());p=json.loads((root/'protocol.json').read_text())
 assert r['protocol_sha256']==sha(root/'protocol.json')
 assert p['samples']==32768 and p['parameters']==1728801536 and p['control_bits']==12
 assert [c.get('seed')for c in p['cases'][2:]]==list(range(2026111901,2026111906)) and len(p['cases'])==7
 calibration=OUT/'lfm3_calibration_v2/results.json';assert p['calibration_sha256']==sha(calibration)
 assert p['selected_policies']==json.loads(calibration.read_text())['policies'];policies=p['selected_policies']
 train_file=OUT/'lfm3_training_depth_v1/results.json';assert r['training_qualification_sha256']==sha(OUT/'lfm3_training_depth_analysis_v1/results.json')
 assert json.loads(OUT.joinpath('lfm3_training_depth_analysis_v1/results.json').read_text())['results_sha256']==sha(train_file)
 assert json.loads(train_file.read_text())['MC95_t4_PPL_ratio_interval'][1]<=1.02
 data=OUT/'lfm3_data_v1';assert sha(data/'test_tokens.npz')==p['test_tokens_sha256']and sha(data/'test_layout.json')==p['test_layout_sha256']
 with np.load(data/'test_tokens.npz')as f:tokens=f['tokens'].copy();offsets=f['offsets'].copy()
 windows=json.loads((data/'test_layout.json').read_text())['windows'];assert len(windows)==p['windows']==607
 expected_counts=coverage(windows,offsets);assert expected_counts.sum()==p['targets']==294442 and len(expected_counts)==62
 original=json.loads((OUT/'lfm3_full_preflight_v2_audit.json').read_text())['weight_point_sha256'];cases=[];article_losses=[];sample_moments=[]
 for case_index,case in enumerate(p['cases']):
  folder=root/case['name'];point=json.loads((folder/'parameters.json').read_text());assert len(point['point_sha256'])==64
  if case['kind']=='original':assert point['point_sha256']==original
  elif case['kind']=='selected':
   records=point['records'];assert len(records)==len(policies)==286
   for record,policy in zip(records,policies):
    assert record['name']==policy['name']and record['parameters']==policy['parameters']and record['samples']==32768
    assert record['group']==policy['selected_group']and record['layout']==policy['selected_layout']
    assert record['range_groups']==math.ceil(policy['parameters']/policy['selected_group'])
    assert record['counter_dtype']=='uint16'and 0<=record['max_count']<=32768
    assert len(record['counts_sha256'])==len(record['BF16_sha256'])==64
    assert all(np.isfinite(record[k])and record[k]>=0 for k in ['SSE_about_mean','expected_SSE','variance_of_SSE'])
   actual=sum(x['SSE_about_mean']for x in records);expected=sum(x['expected_SSE']for x in records);variance=sum(x['variance_of_SSE']for x in records)
   z=(actual-expected)/np.sqrt(variance);assert abs(z)<8,'Aggregate sampled-weight variance inconsistent with independent-count model'
   sample_moments.append({'seed':case['seed'],'observed_SSE':actual,'expected_SSE':expected,'SSE_standardized_error':float(z)})
  else:assert point['records']==[]
  assert {x.name for x in folder.glob('window*.npz')}=={f'window{i:04d}.npz'for i in range(607)}
  assert {x.name for x in folder.glob('window*.json')}=={f'window{i:04d}.json'for i in range(607)}
  sums=np.zeros(62);counts=np.zeros(62,np.int64);peak=0
  for index,w in enumerate(windows):
   path=folder/f'window{index:04d}.npz';receipt=json.loads(path.with_suffix('.json').read_text())
   base=int(offsets[w['article']]);ids=tokens[base+w['context_start']:base+w['context_stop']]
   assert receipt['window']==w and receipt['point_sha256']==point['point_sha256']and receipt['input_sha256']==hashlib.sha256(ids.tobytes()).hexdigest()
   assert receipt['arrays_sha256']==sha(path)and len(receipt['BF16_scored_logits_sha256'])==64
   assert 0<receipt['GPU_peak_allocated_bytes']<=p['GPU_cap_bytes'];peak=max(peak,receipt['GPU_peak_allocated_bytes'])
   with np.load(path)as f:loss=f['token_NLL'];assert loss.shape==(w['target_stop']-w['target_start'],)and np.isfinite(loss).all()and(loss>=0).all()
   sums[w['article']]+=float(loss.sum());counts[w['article']]+=len(loss)
  np.testing.assert_array_equal(counts,expected_counts)
  saved=json.loads((folder/'results.json').read_text());mean=float(sums.sum()/counts.sum())
  assert r['cases'][case_index]==saved
  np.testing.assert_allclose(saved['NLL'],mean,rtol=1e-14);np.testing.assert_allclose(saved['PPL'],np.exp(mean),rtol=1e-14)
  np.testing.assert_array_equal(saved['article_target_counts'],counts);np.testing.assert_allclose(saved['article_NLL_sums'],sums,rtol=1e-14)
  assert saved['case']==case and saved['point_sha256']==point['point_sha256']and saved['GPU_peak_allocated_bytes']==peak
  cases.append({'case':case,'NLL':mean,'PPL':float(np.exp(mean)),'targets':int(counts.sum())});article_losses.append(sums)
  print('audited',case['name'],flush=True)
 values=np.array([c['NLL']for c in cases[2:]]);statistics=primary_statistics(values,cases[0]['NLL'])
 np.testing.assert_allclose(statistics['PPL_ratio'],r['PPL_ratio'],rtol=1e-14);np.testing.assert_allclose(statistics['MC95_t4_PPL_ratio_interval'],r['MC95_t4_PPL_ratio_interval'],rtol=1e-14);assert statistics['pass']==r['pass']
 # Same article sample for baseline/all model draws preserves the comparison pairing.
 a=np.array(article_losses);rng=np.random.default_rng(2026112301);boot=[]
 for _ in range(5000):
  articles=rng.integers(0,62,62);seeds=rng.integers(0,5,5)+2
  delta=(a[seeds][:,articles].mean(0).sum()-a[0,articles].sum())/expected_counts[articles].sum();boot.append(np.exp(delta))
 output={'status':'all7_cases_all294442_targets_saved_outputs_and_statistics_audited','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),'cases':cases,
  'statistics':statistics,'joint_seed_article_bootstrap_95':np.percentile(boot,[2.5,97.5]).tolist(),'bootstrap_seed':2026112301,'bootstrap_replicates':5000,
  'finite_controller_mean_PPL_ratio':float(np.exp(cases[1]['NLL']-cases[0]['NLL'])),'sampling_PPL_ratio_relative_to_controller_mean':float(np.exp(values.mean()-cases[1]['NLL'])),
  'sample_moments':sample_moments,'scope':'Saved-output/statistical audit. Full-model neural and weight-program replay remains separately required. '+p['scope']}
 dest=OUT/'lfm3_benchmark_audit_v1';dest.mkdir(exist_ok=True);(dest/'results.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(statistics))
if __name__=='__main__':main()
