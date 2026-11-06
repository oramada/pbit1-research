"""All predeclared full CNN outcomes and paired descriptive uncertainty."""
import json
import numpy as np
from results.thrml_spw.fashion_cnn_native.cohort.common import OUT,CONTROLS,sha,write,save,guard,metrics
def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);m=json.loads((OUT/'manifest.json').read_text());audit=json.loads((OUT/'native_audit.json').read_text());assert m['complete'] and audit['passed'] and audit['producer_manifest_sha256']==sha(OUT/'manifest.json');expected={(mi,t,seed) for mi in range(5) for t in p['samples'] for seed in range(10)};assert {(r['model'],r['T'],r['seed']) for r in m['rows']}==expected and len(m['rows'])==150
    labels=np.load(OUT/'test_data.npz')['labels'];controls=np.load(OUT/'control_logits.npz');base=np.empty((10,5,2));br=[]
    for ci,name in enumerate(CONTROLS):
        for mi in range(5):
            v=metrics(controls[f'c{ci}_model{mi}'],labels);base[ci,mi]=v;br.append({'control':name,'model':mi,'accuracy':float(v[0]),'mean_CE':float(v[1])})
    values=np.empty((3,5,10,2));rows=[]
    for r in m['rows']:
        assert sha(r['path'])==r['sha256'];v=metrics(np.load(r['path'])['logits'],labels);bi=p['samples'].index(r['T']);values[bi,r['model'],r['seed']]=v;rows.append({'name':r['name'],'model':r['model'],'seed':r['seed'],'T':r['T'],'accuracy':float(v[0]),'mean_CE':float(v[1]),'clipped_coordinates':r['clipped_coordinates']})
    rng=np.random.default_rng(p['bootstrap_seed']);bm=rng.integers(0,5,size=(5000,5));bs=rng.integers(0,10,size=(5000,5,10));contrasts=[];dist={};summaries=[]
    for bi,t in enumerate(p['samples']):
        summaries.append({'T':t,'mean':values[bi].mean((0,1)).tolist(),'model_means':values[bi].mean(1).tolist(),'clipped_coordinates':np.sum([r['clipped_coordinates'] for r in m['rows'] if r['T']==t],axis=0).tolist()})
        for ci,name in enumerate(CONTROLS):
            delta=values[bi]-base[ci,:,None,:];z=delta[bm[:,:,None],bs].mean((1,2));dist[f'budget{bi}_control{ci}']=z;contrasts.append({'T':t,'budget_index':bi,'control':name,'control_index':ci,'difference':delta.mean((0,1)).tolist(),'descriptive_95_interval':np.quantile(z,[.025,.975],axis=0).T.tolist()})
    save(OUT/'statistical_arrays.npz',values=values,baseline_metrics=base,bootstrap_models=bm,bootstrap_seeds=bs,**dist)
    result={'manifest_sha256':sha(OUT/'manifest.json'),'native_audit_sha256':sha(OUT/'native_audit.json'),'controls_sha256':sha(OUT/'controls.json'),'test_data_sha256':sha(OUT/'test_data.npz'),'control_logits_sha256':sha(OUT/'control_logits.npz'),'statistical_arrays_sha256':sha(OUT/'statistical_arrays.npz'),'rows':rows,'controls':br,'summaries':summaries,'control_summary':{name:base[ci].mean(0).tolist() for ci,name in enumerate(CONTROLS)},'contrasts':contrasts,'metric_order':['accuracy_fraction','mean_CE_nats'],'scope':p['statistics']};write(OUT/'statistics.json',result);print('All150CNN outcomes and30contrasts retained; independent audit pending',flush=True)
if __name__=='__main__':main()
