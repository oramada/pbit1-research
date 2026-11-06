"""Audit all lower-work GPT2 outputs, budgets and unchanged reference predictions."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from .prepare_v1 import OUT,sha
from .audit_lfm3_benchmark_v1 import coverage,primary_statistics


def main():
 root=OUT/'reduced_work_gpu_v1';r=json.loads((root/'results.json').read_text());p=json.loads((root/'protocol.json').read_text());plan=json.loads((OUT/'reduced_work_GPU_plan_v1.json').read_text());assert r['protocol_sha256']==sha(root/'protocol.json')and p['plan_sha256']==sha(OUT/'reduced_work_GPU_plan_v1.json')
 assert p['policies']==plan['parameters'];policies=p['policies'];assert len(p['cases'])==6 and [x.get('seed')for x in p['cases'][1:]]==plan['seeds']
 data=p['data'];assert sha(data['tokens'])==data['tokens_sha256']and sha(data['layout'])==data['layout_sha256']
 with np.load(data['tokens'])as f:tokens=f['tokens'].copy();offsets=f['offsets'].copy()
 windows=json.loads(Path(data['layout']).read_text())['windows'];assert len(windows)==586;expected_counts=coverage(windows,offsets);assert expected_counts.sum()==283225
 N=sum(x['parameters']for x in policies);observations=sum(x['parameters']*x['samples']for x in policies);fraction=observations/(N*32768)
 assert N==124439808 and observations==2378245865472 and fraction==plan['observation_fraction_of_uniform32768']<=.60
 assert max(x['samples']for x in policies)==32768
 metadata=8*sum(math.ceil(x['parameters']/x['group'])for x in policies);uniform_metadata=8*sum(math.ceil(x['parameters']/128)for x in policies)
 assert metadata-uniform_metadata==53088
 summaries=[]
 for ci,case in enumerate(p['cases']):
  folder=root/case['name'];params=json.loads((folder/'parameters.json').read_text())
  if case['kind']=='reduced':
   assert len(params['records'])==len(policies)==148
   for x,y in zip(params['records'],policies):
    assert x['name']==y['name']and x['parameters']==y['parameters']and x['samples']==y['samples']and x['group']==y['group']and x['layout']==y['layout']
    assert x['counter_dtype']=='uint16'and 0<=x['max_count']<=x['samples']
  else:
   original=json.loads((OUT/'benchmark_v1/BF16/parameters.json').read_text());assert params==original
  sums=np.zeros(62);counts=np.zeros(62,np.int64);peak=0
  for wi,w in enumerate(windows):
   file=folder/f'window{wi:04d}.npz';receipt=json.loads(file.with_suffix('.json').read_text());assert receipt['arrays_sha256']==sha(file)and receipt['point_sha256']==params['point_sha256']and receipt['window']==w
   start=int(offsets[w['article']]);ids=tokens[start+w['context_start']:start+w['context_stop']][None];assert receipt['input_sha256']==hashlib.sha256(ids.tobytes()).hexdigest()
   with np.load(file)as f:
    loss=f['token_NLL'].copy();assert loss.shape==(w['target_stop']-w['target_start'],)and np.isfinite(loss).all()
    np.testing.assert_array_equal(loss,f['log_normalizer']-f['target_logit']);np.testing.assert_array_equal(f['target_ids'],ids[0,w['target_start']-w['context_start']:w['target_stop']-w['context_start']])
    if case['kind']=='original':
     with np.load(OUT/'benchmark_v1/BF16'/f'window{wi:04d}.npz')as old:
      for key in f.files:np.testing.assert_array_equal(f[key],old[key])
   sums[w['article']]+=float(loss.sum());counts[w['article']]+=len(loss);peak=max(peak,receipt['GPU_peak_allocated_bytes']);assert peak<=2*1024**3
  np.testing.assert_array_equal(counts,expected_counts);saved=json.loads((folder/'results.json').read_text());assert r['cases'][ci]==saved
  mean=float(sums.sum()/counts.sum());np.testing.assert_allclose(mean,saved['NLL'],rtol=1e-14);np.testing.assert_allclose(sums,saved['article_NLL_sums'],rtol=1e-14);np.testing.assert_array_equal(counts,saved['article_target_counts'])
  summaries.append({'case':case,'NLL':mean,'PPL':float(np.exp(mean))});print('AUDITED',case['name'],flush=True)
 stats=primary_statistics(np.array([x['NLL']for x in summaries[1:]]),summaries[0]['NLL']);assert stats['pass']==r['pass'];np.testing.assert_allclose(stats['PPL_ratio'],r['PPL_ratio'],rtol=1e-14);np.testing.assert_allclose(stats['MC95_t4_PPL_ratio_interval'],r['MC95_t4_PPL_ratio_interval'],rtol=1e-14)
 output={'status':'all6_cases_283225targets_budgets_and_original_BF16_arrays_audited','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),'cases':summaries,'statistics':stats,'observations':observations,'observation_fraction':fraction,'maximum_samples':32768,'extra_range_metadata_bytes':53088,'scope':p['scope']+' Neural/count replay is separate.'}
 (OUT/'reduced_work_gpu_v1_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(stats))
if __name__=='__main__':main()
