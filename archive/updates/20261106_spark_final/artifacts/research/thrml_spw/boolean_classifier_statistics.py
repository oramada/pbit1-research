"""All prespecified classifier summaries, controls and paired uncertainty."""
import json
import numpy as np
from .boolean_classifier_common import OUT,sha,guard,metrics,encode,fields
from .fine_range_common import atomic_json,atomic_npz
BASES=['FP32_weights_F64_readout','stored_FP32_reference','INT2','INT4','INT8','programmed_mean_R1','programmed_mean_R3']
def baselines(x,model):
    w=[model[f'weight_{i}'].astype(float) for i in range(2)];b=[model[f'bias_{i}'].astype(float) for i in range(2)]
    run=lambda ww:np.maximum(x@ww[0]+b[0],0)@ww[1]+b[1]
    out={'FP32_weights_F64_readout':run(w),'stored_FP32_reference':model['reference_test_logits']}
    for bit in [2,4,8]:out[f'INT{bit}']=run([np.rint(a*(2**bit-1))/(2**bit-1) for a in w])
    pw=[np.exp(-np.logaddexp(0,-2*fields(encode(a)).astype(float))) for a in w];out['programmed_mean_R1']=run(pw)
    return out,pw

def main():
    p=json.loads((OUT/'protocol.json').read_text());a=guard(p);m=json.loads((OUT/'manifest.json').read_text());audit=json.loads((OUT/'native_audit.json').read_text());assert m['complete'] and audit['passed'] and audit['producer_manifest_sha256']==sha(OUT/'manifest.json')
    d=np.load(a['data_path']);x=d['test_inputs'].astype(float);labels=d['test_labels'];low=a['input_lower'];span=a['input_span'];base=np.empty((7,5,2));baseline_arrays={};baseline_rows=[]
    for mi in range(5):
        model=np.load(a['models'][mi]['model_path']);control=np.load(a['models'][mi]['controls_path']);out,pw=baselines(x,model)
        qa=np.exp(-np.logaddexp(0,-2*fields(encode(np.clip((x-low)/span,0,1))).astype(float)));h=np.maximum((low+span*qa)@pw[0]+model['bias_0'],0);hlo=float(control['hidden_lower']);hspan=float(control['hidden_span'])
        qh=np.exp(-np.logaddexp(0,-2*fields(encode(np.clip((h-hlo)/hspan,0,1))).astype(float)));out['programmed_mean_R3']=(hlo+hspan*qh)@pw[1]+model['bias_1']
        for bi,name in enumerate(BASES):
            v=metrics(out[name],labels);base[bi,mi]=[v['accuracy'],v['mean_cross_entropy']];baseline_arrays[f'c{bi}_model{mi}']=out[name];baseline_rows.append({'condition':name,'model':mi,**v})
    atomic_npz(OUT/'baseline_logits.npz',**baseline_arrays)
    values=np.empty((2,3,5,10,2));rows=[]
    for row in m['rows']:
        assert sha(row['path'])==row['sha256'];logits=np.load(row['path'])['logits'];v=metrics(logits,labels);r=['R1','R3'].index(row['regime']);ti=p['samples'][row['regime']].index(row['T']);values[r,ti,row['model'],row['seed']]=[v['accuracy'],v['mean_cross_entropy']];rows.append({'name':row['name'],'regime':row['regime'],'T':row['T'],'model':row['model'],'seed':row['seed'],'hidden_clipped_coordinates':row['hidden_clipped_coordinates'],**v})
    rng=np.random.default_rng(1357520);models=rng.integers(0,5,size=(5000,5));seeds=rng.integers(0,10,size=(5000,5,10));distributions={};contrasts=[];summaries=[]
    for ti in range(3):
        for ri,r in enumerate(['R1','R3']):
            z=values[ri,ti];summaries.append({'regime':r,'T':p['samples'][r][ti],'mean_accuracy':float(z[:,:,0].mean()),'mean_cross_entropy':float(z[:,:,1].mean()),'model_mean_metrics':z.mean(1).tolist()})
        for name,z in [('R3_minus_R1',values[1,ti]-values[0,ti]),('R1_minus_FP32',values[0,ti]-base[0,:,None,:]),('R3_minus_FP32',values[1,ti]-base[0,:,None,:])]:
            dist=z[models[:,:,None],seeds].mean((1,2));key=f'{name}_budget{ti}';distributions[key]=dist;contrasts.append({'contrast':name,'budget_index':ti,'difference':z.mean((0,1)).tolist(),'descriptive_95_interval':np.quantile(dist,[.025,.975],axis=0).T.tolist()})
    atomic_npz(OUT/'statistical_arrays.npz',values=values,baseline_metrics=base,bootstrap_models=models,bootstrap_seeds=seeds,**distributions)
    result={'producer_manifest_sha256':sha(OUT/'manifest.json'),'native_audit_sha256':sha(OUT/'native_audit.json'),'rows':rows,'baseline_rows':baseline_rows,'summaries':summaries,'contrasts':contrasts,'baseline_summary':{n:base[i].mean(0).tolist() for i,n in enumerate(BASES)},'metric_order':['accuracy_fraction','cross_entropy_nats'],'bootstrap_draws':5000,'bootstrap_seed':1357520,'baseline_logits_sha256':sha(OUT/'baseline_logits.npz'),'statistical_arrays_sha256':sha(OUT/'statistical_arrays.npz'),'scope':'Five model units, ten MC seeds each, consumed10000-image test set fixed in descriptive hierarchical bootstrap. Expected-weight/activation plug-in controls are not expectations of the stochastic nonlinear network. Quantization is uniform[0,1], dequantized F64 software. No native integer-kernel or physical advantage.'}
    atomic_json(OUT/'statistics.json',result);print('All prespecified summaries written; independent statistics audit pending',flush=True)
if __name__=='__main__':main()
