"""Fixed paired-model/independent-MC matched-resource contrasts."""
import json
import numpy as np
from results.thrml_spw.fashion_cnn_native.word_comparison.common import OUT,PARENT,sha,write,save,guard,parent_closed,metrics
from results.thrml_spw.fashion_cnn_native.cohort.common import CONTROLS
def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);parent_closed();m=json.loads((OUT/'manifest.json').read_text());audit=json.loads((OUT/'native_audit.json').read_text());assert m['complete'] and audit['passed'] and audit['manifest_sha256']==sha(OUT/'manifest.json');assert {(r['model'],r['T'],r['seed']) for r in m['rows']}=={(mi,t,seed) for mi in range(5) for t in p['word_samples'] for seed in range(10)}
    y=np.load(PARENT/'test_data.npz')['labels'];parent=np.load(PARENT/'statistical_arrays.npz');plugin=np.load(OUT/'mean_logits.npz');mean=np.array([metrics(plugin[f'model{mi}'],y) for mi in range(5)]);values=np.empty((3,5,10,2));rows=[]
    for r in m['rows']:
        assert sha(r['path'])==r['sha256'];v=metrics(np.load(r['path'])['logits'],y);bi=p['word_samples'].index(r['T']);values[bi,r['model'],r['seed']]=v;rows.append({'name':r['name'],'model':r['model'],'T':r['T'],'seed':r['seed'],'accuracy':float(v[0]),'mean_CE':float(v[1])})
    bm=parent['bootstrap_models'];bs_single=parent['bootstrap_seeds'];bs_word=np.random.default_rng(p['bootstrap_word_seed']).integers(0,10,size=(5000,5,10));dist={};contrasts=[];summaries=[]
    for bi,t in enumerate(p['word_samples']):
        word_draws=values[bi][bm[:,:,None],bs_word].mean((1,2));summaries.append({'T_word':t,'T_single':4*t,'mean':values[bi].mean((0,1)).tolist(),'model_means':values[bi].mean(1).tolist(),'clipped_coordinates':np.sum([r['clipped_coordinates'] for r in m['rows'] if r['T']==t],axis=0).tolist()})
        for name in p['contrasts']:
            if name=='word_minus_single':reference=parent['values'][bi];ref_draws=reference[bm[:,:,None],bs_single].mean((1,2));point=values[bi].mean((0,1))-reference.mean((0,1))
            else:
                reference=mean if name=='word_minus_nominal_word_mean' else parent['baseline_metrics'][CONTROLS.index('FP32_F64' if name=='word_minus_FP32' else 'ACT8_INT4_minmax')];ref_draws=reference[bm].mean(1);point=values[bi].mean((0,1))-reference.mean(0)
            z=word_draws-ref_draws;dist[f'{name}_budget{bi}']=z;contrasts.append({'contrast':name,'budget_index':bi,'difference':point.tolist(),'descriptive_95_interval':np.quantile(z,[.025,.975],axis=0).T.tolist()})
    save(OUT/'statistical_arrays.npz',values=values,mean_metrics=mean,bootstrap_word_seeds=bs_word,**dist);write(OUT/'statistics.json',{'manifest_sha256':sha(OUT/'manifest.json'),'native_audit_sha256':sha(OUT/'native_audit.json'),'parent_statistics_sha256':sha(PARENT/'statistics.json'),'parent_arrays_sha256':sha(PARENT/'statistical_arrays.npz'),'mean_logits_sha256':sha(OUT/'mean_logits.npz'),'mean_weights_sha256':sha(OUT/'mean_weights.npz'),'statistical_arrays_sha256':sha(OUT/'statistical_arrays.npz'),'rows':rows,'summaries':summaries,'mean_control_summary':mean.mean(0).tolist(),'contrasts':contrasts,'metric_order':['accuracy_fraction','mean_CE_nats'],'scope':p['statistics']});print('All matched-resource word contrasts retained; independent audit pending',flush=True)
if __name__=='__main__':main()
