"""Derived static-cache quality control; parent audits must be complete."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
OUT=Path('results/thrml_spw/boolean_classifier_cached_reuse');PARENT=Path('results/thrml_spw/boolean_classifier')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
    e=json.loads((OUT/'execution_protocol.json').read_text())
    for p,h in e['source_sha256'].items():assert sha(p)==h
    assert e['selection_protocol_sha256']==sha(OUT/'protocol.json')
    a=json.loads((PARENT/'native_audit.json').read_text());s=json.loads((PARENT/'statistics_audit.json').read_text());assert a['passed'] and s['passed'];assert a['producer_manifest_sha256']==sha(PARENT/'manifest.json') and s['statistics_sha256']==sha(PARENT/'statistics.json');assert (PARENT/'completion.json').exists()
    for path,h in json.loads((PARENT/'completion.json').read_text())['artifact_sha256'].items():assert sha(path)==h
    assert sha(OUT/'selected_counts.npz')==json.loads((OUT/'protocol.json').read_text())['selected_counts_sha256']
    return e

def main():
    check();assert not (OUT/'evaluation.json').exists();p=json.loads((OUT/'protocol.json').read_text());prep=json.loads(Path('results/thrml_spw/boolean_classifier_preparation_v2/preparation.json').read_text());d=np.load(prep['data_path']);x=d['test_inputs'].astype(float).reshape(1250,8,64);labels=d['test_labels'];counts=np.load(OUT/'selected_counts.npz');parent=np.load(PARENT/'statistical_arrays.npz');values=np.empty((3,5,10,2));preds=np.empty((5,3,10,10000),np.uint8);losses=np.empty((5,3,10,10000),float);rows=[]
    for row in p['selected']:
        assert sha(row['parent_path'])==row['parent_sha256'];mi,bi,si,t=row['model'],row['budget_index'],row['seed'],row['T'];model=np.load(prep['models'][mi]['controls_path']);w0=counts['weight_0'][mi,bi,si].astype(float)/t;w1=counts['weight_1'][mi,bi,si].astype(float)/t;z=(np.maximum(x@w0+model['bias_0'],0)@w1+model['bias_1']).reshape(10000,10);np.testing.assert_array_equal(z[:8],np.load(row['parent_path'])['logits'][:8]);pred=z.argmax(1).astype(np.uint8);v=z-z.max(1,keepdims=True);loss=np.log(np.exp(v).sum(1))-v[np.arange(10000),labels];values[bi,mi,si]=[np.mean(pred==labels),loss.mean()];preds[mi,bi,si]=pred;losses[mi,bi,si]=loss;rows.append({'model':mi,'budget_index':bi,'seed':si,'T':t,'all_logits_byte_sha256':hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest(),'first8_parent_logits_exact':True})
    models,seeds=parent['bootstrap_models'],parent['bootstrap_seeds'];diff=values-parent['values'][0];draws={};summary=[]
    for bi,t in enumerate([98,392,1568]):
        dist=diff[bi][models[:,:,None],seeds].mean((1,2));draws[f'budget{bi}']=dist;summary.append({'T':t,'cached_mean':values[bi].mean((0,1)).tolist(),'per_bank_mean':parent['values'][0,bi].mean((0,1)).tolist(),'cached_minus_per_bank':diff[bi].mean((0,1)).tolist(),'descriptive_95_interval':np.quantile(dist,[.025,.975],axis=0).T.tolist()})
    np.savez_compressed(OUT/'derived_predictions_losses.npz',predictions=preds,losses=losses);np.savez_compressed(OUT/'statistics.npz',values=values,**draws)
    result={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_protocol_sha256':sha(OUT/'protocol.json'),'execution_protocol_sha256':sha(OUT/'execution_protocol.json'),'parent_completion_sha256':sha(PARENT/'completion.json'),'parent_statistical_arrays_sha256':sha(PARENT/'statistical_arrays.npz'),'selected_counts_sha256':sha(OUT/'selected_counts.npz'),'predictions_losses_sha256':sha(OUT/'derived_predictions_losses.npz'),'statistics_sha256':sha(OUT/'statistics.npz'),'metric_order':['accuracy_fraction','mean_cross_entropy_nats'],'rows':rows,'summaries':summary,'new_native_updates':0,'scope':'All150 fixed first-bank cache controls; reused parent sampled trajectories/seeds, not new independent replications or fresh test data. Counterfactual sampling amortization requires retained F64 matrices and conventional dense arithmetic. No original cohort metric/gate/resource receipt is changed.'};(OUT/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n');print('Static cache evaluation complete; independent derived audit pending',flush=True)
if __name__=='__main__':main()
