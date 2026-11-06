"""All prespecified integer8 quality comparisons and deterministic ACT8 controls."""
import json
import numpy as np
from results.thrml_spw.integer_activation_native.cohort_common import OUT,PARENT,sha,guard,parent_closed,atomic_json,atomic_npz
def activation_codes(x,lower,span):
    return np.rint(255*np.clip((np.asarray(x,float)-lower)/span,0,1)).astype(np.uint8)
NAMES=['ACT8_FP32_weights','ACT8_INT2_weights','ACT8_INT4_weights','ACT8_INT8_weights','ACT8_programmed_weight_mean']
def metrics(z,y):
    z=np.asarray(z,float);v=z-z.max(1,keepdims=True);return np.array([np.mean(z.argmax(1)==y),np.mean(np.log(np.exp(v).sum(1))-v[np.arange(len(y)),y])])
def control_forward(x,model,weights):
    h=x
    for layer,w in enumerate(weights):
        low=float(model['input_lower'] if layer==0 else model['hidden_lower']);span=float(model['input_span'] if layer==0 else model['hidden_span']);code=activation_codes(h,low,span);y=low*w.sum(0)+(span/255)*np.matmul(code.astype(float),w)+model[f'bias_{layer}'];h=np.maximum(y,0) if layer==0 else y
    return h.reshape(10000,10)
def main():
    p=json.loads((OUT/'protocol.json').read_text());prep=guard(p);parent_closed();m=json.loads((OUT/'manifest.json').read_text());a=json.loads((OUT/'evidence_audit.json').read_text());assert a['passed'] and a['producer_manifest_sha256']==sha(OUT/'manifest.json');d=np.load(prep['data_path']);x=d['test_inputs'].astype(float).reshape(1250,8,64);labels=d['test_labels'];parent=np.load(PARENT/'statistical_arrays.npz');values=np.empty((3,5,10,2));rows=[]
    for r in m['rows']:
        assert sha(r['path'])==r['sha256'];bi=p['samples'].index(r['T']);v=metrics(np.load(r['path'])['logits'],labels);values[bi,r['model'],r['seed']]=v;rows.append({'name':r['name'],'model':r['model'],'seed':r['seed'],'T':r['T'],'accuracy':v[0],'mean_CE':v[1]})
    baseline=np.empty((5,5,2));bl={};controls=[]
    for mi,item in enumerate(prep['models']):
        model=np.load(item['controls_path']);w=[model[f'weight_{i}'].astype(float) for i in [0,1]]
        for ci,name in enumerate(NAMES):
            if ci==0:ww=w
            elif ci==4:ww=[np.exp(-np.logaddexp(0,-2*model[f'weight_field_{i}'].astype(float))) for i in [0,1]]
            else:
                bit=[2,4,8][ci-1];ww=[np.rint(a*(2**bit-1))/(2**bit-1) for a in w]
            z=control_forward(x,model,ww);bl[f'control{ci}_model{mi}']=z;v=metrics(z,labels);baseline[ci,mi]=v;controls.append({'condition':name,'model':mi,'accuracy':v[0],'mean_CE':v[1]})
    atomic_npz(OUT/'control_logits.npz',**bl);bm,bs=parent['bootstrap_models'],parent['bootstrap_seeds'];draws={};summaries=[];contrasts=[]
    for bi,t in enumerate(p['samples']):
        z=values[bi];summaries.append({'T':t,'mean':z.mean((0,1)).tolist(),'model_means':z.mean(1).tolist(),'hidden_clipped_coordinates':sum(r['hidden_clipped_coordinates'] for r in m['rows'] if r['T']==t)})
        refs={'integer8_minus_R1':parent['values'][0,bi],'integer8_minus_R3_equal_updates':parent['values'][1,bi],'integer8_minus_FP32':parent['baseline_metrics'][0,:,None,:]}
        for name,ref in refs.items():
            delta=z-ref;dist=delta[bm[:,:,None],bs].mean((1,2));key=f'{name}_budget{bi}';draws[key]=dist;contrasts.append({'contrast':name,'budget_index':bi,'difference':delta.mean((0,1)).tolist(),'descriptive_95_interval':np.quantile(dist,[.025,.975],axis=0).T.tolist()})
    atomic_npz(OUT/'statistical_arrays.npz',values=values,control_metrics=baseline,**draws)
    result={'producer_manifest_sha256':sha(OUT/'manifest.json'),'count_audit_sha256':sha(OUT/'evidence_audit.json'),'parent_statistics_sha256':sha(PARENT/'statistics.json'),'parent_statistical_arrays_sha256':sha(PARENT/'statistical_arrays.npz'),'rows':rows,'controls':controls,'summaries':summaries,'contrasts':contrasts,'control_summary':{name:baseline[ci].mean(0).tolist() for ci,name in enumerate(NAMES)},'metric_order':['accuracy_fraction','CE_nats'],'control_logits_sha256':sha(OUT/'control_logits.npz'),'statistical_arrays_sha256':sha(OUT/'statistical_arrays.npz'),'scope':'All5 models/10 paired MC trajectories, consumed10000-image test data. Parent5000 bootstrap index draws reused, five model units; no fresh independent replication or new pass/fail gate. ACT8 control plug-in means are not expectations of nonlinear stochastic networks. R3 contrast includes different activation coding and sample rounds at equal active updates, not a pure isolated-noise intervention.'};atomic_json(OUT/'statistics.json',result);print('All150 integer8 case metrics and25 ACT8 controls summarized; independent statistics audit pending',flush=True)
if __name__=='__main__':main()
