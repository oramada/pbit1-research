"""Independent full-model baselines, paired statistics and all-block evidence audit."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .transformer_block_gpu import sha,save_json
OUT=Path('results/thrml_spw/all_blocks_clean')
def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def resample(delta):
    d=np.asarray(delta,float).reshape(5,4,8).mean(-1);rng=np.random.default_rng(2026116801);si=rng.integers(0,5,(10000,5));bi=rng.integers(0,4,(10000,4));return np.array([d[si[i]][:,bi[i]].mean() for i in range(10000)])
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)

def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['config']==p and m['native_full_forward_runs']==5 and m['native_operator_runs']==120 and m['native_probe_windows']==0 and m['first12_prefix_shared_without_extra_sampling'];assert sha(OUT/'protocol.json')==m['execution']['design_sha256']
    for name,h in {**p['source_sha256'],**m['execution']['source_sha256']}.items():assert sha(name)==h
    tokens=np.load(OUT/'tokens.npz')['tokens'];truth=tokens[:,1:];assert sha(OUT/'tokens.npz')==p['tokens_sha256'];rawbase=np.load(OUT/'baseline_tokens.npz');baserows=rows('baseline_metrics.csv');native=rows('native_metrics.csv');prefix=rows('prefix_metrics.csv');summary=rows('summary.csv');assert len(baserows)==5 and len(native)==5 and len(prefix)==125 and len(summary)==2
    model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];original=[a.weight.detach().clone() for a in modules];weights=[w.numpy().copy() for w in original]
    for condition in ['FP32','affine4_all24','affine8_all24','programmed_mean8_first12','programmed_mean8_all24']:
        try:
            if condition!='FP32':
                for stage,module in enumerate(modules):
                    if condition=='programmed_mean8_first12' and stage>=12:continue
                    w=weights[stage]
                    if condition.startswith('programmed'):
                        c=p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);b=(-4+a['codes'].astype(float)*8/255).astype(np.float32);q=1/(1+np.exp(-2*b.astype(float)));ww=(np.repeat(a['lower'],128,0)+np.repeat(a['span'],128,0)*q).astype(np.float32)
                    else:
                        bits=4 if condition=='affine4_all24' else 8;ww=np.empty_like(w)
                        for first in range(0,len(w),128):
                            z=w[first:first+128];lo=np.minimum(z.min(0),0);hi=np.maximum(z.max(0),0);scale=(hi-lo)/((1<<bits)-1);safe=np.where(scale==0,1,scale);zero=np.clip(np.rint(-lo/safe),0,(1<<bits)-1);code=np.clip(np.rint(z/safe)+zero,0,(1<<bits)-1);ww[first:first+128]=(code-zero)*safe
                    with torch.no_grad():module.weight.copy_(torch.from_numpy(ww))
            nll=[];pred=[]
            with torch.inference_mode():
                for row in tokens:
                    ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];target=ids[:,1:];nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
            nll,pred=np.stack(nll),np.stack(pred);np.testing.assert_array_equal(nll,rawbase[condition+'_nll']);np.testing.assert_array_equal(pred,rawbase[condition+'_predictions']);rr=next(r for r in baserows if r['condition']==condition);ce=float(nll.astype(float).mean())
            for k,v in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':np.mean(pred==truth)}.items():close(float(rr[k]),v)
            print(f'Independent all-block baseline {condition}',flush=True)
        finally:
            with torch.no_grad():
                for module,w in zip(modules,original):module.weight.copy_(w)
    loss={12:[],24:[]};acc={12:[],24:[]};updates=0
    for seed in range(5):
        folder=OUT/'runs'/f'seed{seed}';r=json.loads((folder/'run.json').read_text());raw=np.load(folder/'prefix_tokens.npz');assert r['status']=='complete' and sha(folder/'prefix_tokens.npz')==r['token_archive_sha256'] and r['protocol_sha256']==sha(OUT/'protocol.json') and r['exact_prefix_input_checks']==9600 and len(r['stages'])==24;np.testing.assert_array_equal(raw['prefix0_nll'],rawbase['FP32_nll']);np.testing.assert_array_equal(raw['prefix0_predictions'],rawbase['FP32_predictions']);cu=0
        for stage,s in enumerate(r['stages']):
            assert s['stage']==stage;meta=json.loads(Path(s['metadata']).read_text());assert sha(s['output'])==meta['output_sha256'] and s['inputs_sha256']==meta['signature']['inputs_sha256'];assert s['prior_native_output_sha256']==[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)];cu+=meta['free_node_updates']
        assert cu==r['free_node_updates']==4*1024*42467328;updates+=cu
        for k in range(25):
            ce=float(raw[f'prefix{k}_nll'].astype(float).mean());ac=float(np.mean(raw[f'prefix{k}_predictions']==truth));rr=next(v for v in prefix if int(v['sampling_seed'])==seed and int(v['converted_prefix'])==k)
            for key,val in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':ac}.items():close(float(rr[key]),val)
            if k in [12,24]:loss[k].append(raw[f'prefix{k}_nll'].astype(float).mean(-1));acc[k].append((raw[f'prefix{k}_predictions']==truth).mean(-1))
        for key,value in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':ac}.items():close(r['scores'][key],value)
        rr=next(v for v in native if int(v['sampling_seed'])==seed)
        for key in ['cross_entropy','perplexity','token_accuracy']:close(float(rr[key]),r['scores'][key])
    assert updates==m['native_free_node_updates'];draws=np.load(OUT/'quality_draws.npz');ref=rawbase['FP32_nll'].astype(float).mean(-1);refac=(rawbase['FP32_predictions']==truth).mean(-1)
    for k in [12,24]:
        loss[k],acc[k]=np.stack(loss[k]),np.stack(acc[k]);ce=resample(loss[k]-ref);ac=resample(acc[k]-refac);close(draws[f'prefix{k}_CE_vs_FP32'],ce);close(draws[f'prefix{k}_accuracy_vs_FP32'],ac);delta=float((loss[k]-ref).mean());ad=float((acc[k]-refac).mean());point=delta<=np.log(1.05) and ad>=-.01;rr=next(v for v in summary if int(v['converted_projections'])==k)
        assert int(rr['converted_weights'])==sum(c['weights'] for c in p['control_artifacts'][:k])
        for key,val in {'cross_entropy':loss[k].mean(),'perplexity':np.exp(loss[k].mean()),'relative_perplexity':np.exp(delta),'token_accuracy':acc[k].mean(),'accuracy_difference':ad,'bootstrap_relative_upper':np.exp(np.quantile(ce,.975)),'bootstrap_accuracy_lower':np.quantile(ac,.025)}.items():close(float(rr[key]),val)
        assert (rr['point_passed']=='True')==point and (rr['passes']=='True')==(point and np.quantile(ce,.975)<=np.log(1.05) and np.quantile(ac,.025)>=-.01)
        mm=next(v for v in m['summary'] if v['converted_projections']==k)
        for key,value in rr.items():
            if key in ['passes','point_passed']:assert mm[key]==(value=='True')
            else:close(float(value),mm[key])
    ce=resample(loss[24]-loss[12]);ac=resample(acc[24]-acc[12]);close(draws['paired_CE'],ce);close(draws['paired_accuracy'],ac);pair=json.loads((OUT/'paired_summary.json').read_text());assert pair==m['paired_contrast']
    for key,val in {'CE_difference':(loss[24]-loss[12]).mean(),'perplexity_ratio':np.exp((loss[24]-loss[12]).mean()),'accuracy_difference':(acc[24]-acc[12]).mean(),'CE_sensitivity_low':np.quantile(ce,.025),'CE_sensitivity_high':np.quantile(ce,.975),'accuracy_sensitivity_low':np.quantile(ac,.025),'accuracy_sensitivity_high':np.quantile(ac,.975)}.items():close(pair[key],val)
    for name in ['native_path_audit.json','independent_model_audit.json','preparation_audit.json']:assert json.loads((OUT/name).read_text())['passed']
    audit={'passed':True,'baselines_independently_replayed':5,'prefix_metric_rows_checked':125,'paired_prefixes_per_seed':[12,24],'all_60000_resampling_values_recomputed':True,'all_five_sampling_seeds_retained':True,'native_free_node_updates_checked':updates,'raw_native_operator_paths_and_exact_input_hashes_checked':120,'source_sha256':sha(__file__),'scope':'Pinned original-model baselines, all raw task metrics and paired descriptive quality rules independently recomputed; every native operator separately public-replayed and first12/all24 integration independently checked. Five seeds, four banks, one checkpoint, ideal finite controls; no full-model or physical claim.'};save_json(OUT/'evidence_audit.json',audit);print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
