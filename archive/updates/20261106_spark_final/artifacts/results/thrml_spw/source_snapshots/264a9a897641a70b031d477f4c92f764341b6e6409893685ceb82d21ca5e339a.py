"""Complete training input, parameter-record and count-moment audit for LFM3."""
import json,math
import numpy as np
from .prepare_v1 import OUT,sha

def main():
 root=OUT/'lfm3_training_depth_v1';p=json.loads((root/'protocol.json').read_text());r=json.loads((root/'results.json').read_text());assert r['protocol_sha256']==sha(root/'protocol.json')
 assert p['selection_sha256']==sha(OUT/'lfm3_calibration_v2/results.json')and p['training_data_sha256']==sha(OUT/'lfm3_data_v1/training_windows.npz')
 policies=json.loads((OUT/'lfm3_calibration_v2/results.json').read_text())['policies'];assert sum(x['parameters']for x in policies)==1728801536
 with np.load(OUT/'lfm3_data_v1/training_windows.npz')as f:inputs=f['input_ids'].copy()
 checks=[]
 for index,case in enumerate(p['cases']):
  folder=root/case['name'];record=json.loads((folder/'parameters.json').read_text());losses=[]
  for row in range(32):
   with np.load(folder/f'row{row:02d}.npz')as f:
    np.testing.assert_array_equal(f['input_ids'],inputs[row]);loss=f['token_NLL'];assert loss.shape==(256,)and np.isfinite(loss).all()and(loss>=0).all();losses.extend(loss)
  summary=json.loads((folder/'results.json').read_text());assert summary==r['cases'][index];np.testing.assert_allclose(np.mean(losses),summary['NLL'],rtol=1e-14);assert summary['point_sha256']==record['point_sha256']
  moment=None
  if case['kind']=='selected':
   assert len(record['records'])==286
   for x,y in zip(record['records'],policies):
    assert x['name']==y['name']and x['parameters']==y['parameters']and x['layout']==y['selected_layout']and x['group']==y['selected_group']and x['samples']==32768
    assert x['range_groups']==math.ceil(y['parameters']/y['selected_group'])and x['counter_dtype']=='uint16'and 0<=x['max_count']<=32768
    assert len(x['counts_sha256'])==len(x['BF16_sha256'])==64
   observed=sum(x['SSE_about_mean']for x in record['records']);expected=sum(x['expected_SSE']for x in record['records']);variance=sum(x['variance_of_SSE']for x in record['records']);z=(observed-expected)/np.sqrt(variance);assert abs(z)<8
   moment={'observed_SSE':observed,'expected_SSE':expected,'standardized_error':float(z)}
  else:assert record['records']==[]
  checks.append({'case':case['name'],'input_rows_verified':32,'targets':8192,'point_sha256':record['point_sha256'],'moments':moment})
 output={'status':'all7_training_cases_inputs_scores_and_286_parameter_records_audited','source_sha256':sha(__file__),'result_sha256':sha(root/'results.json'),'checks':checks,'scope':'Count-hash/neural replay is separate and is being performed by causal v2. This audit checks saved inputs, losses, policy records, sample budgets and aggregate moments.'}
 dest=OUT/'lfm3_training_audit_v2';dest.mkdir(exist_ok=True);(dest/'results.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))
if __name__=='__main__':main()
