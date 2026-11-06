"""Independent held-out baselines, scores, paired bootstrap and resource audit."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .transformer_block_gpu import sha,save_json
OUT=Path('results/thrml_spw/sample_allocation')
def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def close(x,y):np.testing.assert_allclose(x,y,atol=1e-12,rtol=1e-12)
def bootstrap(delta):
    d=np.asarray(delta,float).reshape(5,4,8).mean(-1);rng=np.random.default_rng(2026117002);si=rng.integers(0,5,(10000,5));bi=rng.integers(0,4,(10000,4));return np.array([d[si[i]][:,bi[i]].mean() for i in range(10000)])
def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);p=json.loads((OUT/'protocol.json').read_text());a=json.loads((OUT/'allocation.json').read_text());m=json.loads((OUT/'test_manifest.json').read_text());assert m['native_model_cases']==10 and m['native_operators']==240 and not m['test_scores_used_for_selection'];assert sha(OUT/'protocol.json')==m['execution']['protocol_sha256'] and sha(OUT/'allocation.json')==m['execution']['allocation_sha256'] and sha(OUT/'selection_audit.json')==m['execution']['selection_audit_sha256'];assert sha(OUT/'tokens.npz')==p['tokens_sha256']
    for name,h in {**p['source_sha256'],**a['source_sha256'],**m['execution']['source_sha256']}.items():assert sha(name)==h
    tokens=np.load(OUT/'tokens.npz')['test_tokens'];truth=tokens[:,1:];base=np.load(OUT/'test_baseline_tokens.npz');br=rows('test_baseline_metrics.csv');nr=rows('test_native_metrics.csv');pr=rows('test_prefix_metrics.csv');summary=rows('test_summary.csv');draws=np.load(OUT/'test_quality_draws.npz');assert len(br)==4 and len(nr)==10 and len(pr)==250 and len(summary)==2
    model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];original=[v.weight.detach().clone() for v in modules];weights=[v.numpy().copy() for v in original]
    for condition in ['FP32','affine4_all24','affine8_all24','programmed_mean8_all24']:
        try:
            if condition!='FP32':
                for stage,module in enumerate(modules):
                    w=weights[stage]
                    if condition=='programmed_mean8_all24':
                        c=p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];z=np.load(c['path']);b=(-4+z['codes'].astype(float)*8/255).astype(np.float32);q=1/(1+np.exp(-2*b.astype(float)));ww=(np.repeat(z['lower'],128,0)+np.repeat(z['span'],128,0)*q).astype(np.float32)
                    else:
                        bits=4 if condition=='affine4_all24' else 8;ww=np.empty_like(w)
                        for first in range(0,len(w),128):
                            z=w[first:first+128];lo=np.minimum(z.min(0),0);hi=np.maximum(z.max(0),0);scale=(hi-lo)/((1<<bits)-1);safe=np.where(scale==0,1,scale);zero=np.clip(np.rint(-lo/safe),0,(1<<bits)-1);code=np.clip(np.rint(z/safe)+zero,0,(1<<bits)-1);ww[first:first+128]=(code-zero)*safe
                    with torch.no_grad():module.weight.copy_(torch.from_numpy(ww))
            nll=[];pred=[]
            with torch.inference_mode():
                for row in tokens:
                    ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
            nll,pred=np.stack(nll),np.stack(pred);np.testing.assert_array_equal(nll,base[condition+'_nll']);np.testing.assert_array_equal(pred,base[condition+'_predictions']);ce=float(nll.astype(float).mean());rr=next(v for v in br if v['condition']==condition)
            for key,value in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':np.mean(pred==truth)}.items():close(float(rr[key]),value)
            print(f'Independent test baseline {condition}',flush=True)
        finally:
            with torch.no_grad():
                for module,w in zip(modules,original):module.weight.copy_(w)
    losses={};acc={};total=0
    for condition in ['uniform','allocated']:
        ll=[];cc=[]
        for seed in range(5):
            folder=OUT/'test/runs'/f'{condition}_seed{seed}';r=json.loads((folder/'run.json').read_text());assert r['execution']==m['execution'] and r['status']=='complete' and r['exact_prefix_input_checks']==9600 and r['condition']==condition and r['seed']==seed;assert sha(folder/'prefix_tokens.npz')==r['token_archive_sha256'];raw=np.load(folder/'prefix_tokens.npz');np.testing.assert_array_equal(raw['prefix0_nll'],base['FP32_nll']);np.testing.assert_array_equal(raw['prefix0_predictions'],base['FP32_predictions']);updates=0
            for stage,s in enumerate(r['stages']):
                assert s['stage']==stage;meta=json.loads(Path(s['metadata']).read_text());t=1024 if condition=='uniform' else a['samples'][stage];assert s['samples']==meta['signature']['samples']==t and s['inputs_sha256']==meta['signature']['inputs_sha256'] and sha(s['output'])==meta['output_sha256'];assert s['prior_native_output_sha256']==[sha(folder/f'stage{k}_native_output.npz') for k in range(stage)];assert meta['free_node_updates']==4*t*p['control_artifacts'][stage]['weights'];updates+=meta['free_node_updates']
            assert updates==r['free_node_updates']==4*42467328*1024;total+=updates
            for k in range(25):
                ce=float(raw[f'prefix{k}_nll'].astype(float).mean());ac=float(np.mean(raw[f'prefix{k}_predictions']==truth));rr=next(v for v in pr if v['condition']==condition and int(v['seed'])==seed and int(v['converted_prefix'])==k)
                for key,value in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':ac}.items():close(float(rr[key]),value)
            rr=next(v for v in nr if v['condition']==condition and int(v['seed'])==seed);assert int(rr['free_node_updates'])==updates
            for key,value in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':ac}.items():close(float(rr[key]),value);close(r['scores'][key],value)
            ll.append(raw['prefix24_nll'].astype(float).mean(-1));cc.append((raw['prefix24_predictions']==truth).mean(-1))
        losses[condition]=np.stack(ll);acc[condition]=np.stack(cc)
    assert total==m['native_observed_updates']==10*4*1024*42467328;ref=base['FP32_nll'].astype(float).mean(-1);refac=(base['FP32_predictions']==truth).mean(-1)
    for condition in ['uniform','allocated']:
        ce=bootstrap(losses[condition]-ref);ac=bootstrap(acc[condition]-refac);close(draws[condition+'_CE_vs_FP32'],ce);close(draws[condition+'_accuracy_vs_FP32'],ac);delta=float((losses[condition]-ref).mean());ad=float((acc[condition]-refac).mean());point=delta<=np.log(1.05) and ad>=-.01;rr=next(v for v in summary if v['condition']==condition)
        values={'cross_entropy':losses[condition].mean(),'perplexity':np.exp(losses[condition].mean()),'relative_perplexity':np.exp(delta),'token_accuracy':acc[condition].mean(),'accuracy_difference':ad,'bootstrap_relative_upper':np.exp(np.quantile(ce,.975)),'bootstrap_accuracy_lower':np.quantile(ac,.025)}
        for key,value in values.items():close(float(rr[key]),value);close(next(v for v in m['summary'] if v['condition']==condition)[key],value)
        assert (rr['point_passed']=='True')==point and (rr['passes']=='True')==(point and np.quantile(ce,.975)<=np.log(1.05) and np.quantile(ac,.025)>=-.01)
        stored=next(v for v in m['summary'] if v['condition']==condition);assert stored['point_passed']==point and stored['passes']==(point and np.quantile(ce,.975)<=np.log(1.05) and np.quantile(ac,.025)>=-.01)
    ce=bootstrap(losses['allocated']-losses['uniform']);ac=bootstrap(acc['allocated']-acc['uniform']);close(draws['paired_CE'],ce);close(draws['paired_accuracy'],ac);pair=json.loads((OUT/'paired_summary.json').read_text());assert pair==m['paired_contrast']
    for key,value in {'CE_difference':(losses['allocated']-losses['uniform']).mean(),'perplexity_ratio':np.exp((losses['allocated']-losses['uniform']).mean()),'accuracy_difference':(acc['allocated']-acc['uniform']).mean(),'CE_sensitivity_low':np.quantile(ce,.025),'CE_sensitivity_high':np.quantile(ce,.975),'accuracy_sensitivity_low':np.quantile(ac,.025),'accuracy_sensitivity_high':np.quantile(ac,.975)}.items():close(pair[key],value)
    for name in ['preparation_audit.json','selection_audit.json','validation_evidence_audit.json','test_model_audit.json','test_native_audit.json']:assert json.loads((OUT/name).read_text())['passed']
    native=json.loads((OUT/'test_native_audit.json').read_text());r={'passed':True,'test_model_cases':10,'native_operators':240,'baselines_independently_replayed':4,'prefix_metric_rows_checked':250,'all_60000_bootstrap_values_recomputed':True,'exact_equal_weighted_updates_both_methods':True,'production_node_updates_checked':total,'all_public_F32_native_outputs_bitwise_exact':native['all_public_F32_outputs_bitwise_exact'],'descriptive_2e_5_failures_retained':native['descriptive_2e_5_failures'],'allocation_frozen_from_validation_only':True,'source_sha256':sha(__file__),'scope':'Independent original-model baselines, raw labels/scores/quality gates and paired resampling/resource counts, after exact public native/model path audits. Surrogate optimum not proof of task gain; five seeds/four banks/one checkpoint,51.84% conversion, ideal devices. No physical superiority.'};save_json(OUT/'evidence_audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
