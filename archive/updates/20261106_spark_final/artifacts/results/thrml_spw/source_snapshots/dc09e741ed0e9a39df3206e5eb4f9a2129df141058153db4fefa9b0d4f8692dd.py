"""Independent fresh task baselines, paired gates, counters and combined evidence."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .verify_transformer_control import sha
ROOT=Path('results/thrml_spw/probe_placement')


def read(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))


def bootstrap(delta):
    d=np.asarray(delta,float).reshape(5,8,8).mean(-1);rng=np.random.default_rng(2026116701);si=rng.integers(0,5,(10000,5));bi=rng.integers(0,8,(10000,8))
    return np.array([d[si[i]][:,bi[i]].mean() for i in range(10000)])


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    p=json.loads((ROOT/'protocol.json').read_text());m=json.loads((ROOT/'manifest.json').read_text());assert m['config']==p and m['native_forward_runs']==10 and m['native_operator_runs']==120 and m['native_probe_windows']==240
    assert sha(ROOT/'protocol.json')==m['execution']['design_sha256'] and sha(ROOT/'tokens.npz')==p['tokens_sha256'] and sha(ROOT/'placement_designs.json')==p['placement_designs_sha256']
    for n,h in {**p['source_sha256'],**m['execution']['source_sha256']}.items():assert sha(n)==h
    tokens=np.load(ROOT/'tokens.npz')['tokens'];truth=tokens[:,1:];baseraw=np.load(ROOT/'baseline_tokens.npz');baserows=read('baseline_metrics.csv');rows=read('native_metrics.csv');prefix=read('prefix_metrics.csv');summary=read('summary.csv');draws=np.load(ROOT/'quality_draws.npz')
    assert len(rows)==10 and len(prefix)==130 and len(baserows)==4 and len(summary)==2
    assert {(r['method'],int(r['sampling_seed'])) for r in rows}=={(v,s) for v in ['fixed','weight_fisher'] for s in range(5)}
    model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];original=[a.weight.detach().clone() for a in modules];weights=[w.numpy().copy() for w in original]
    for condition in ['FP32','affine4_block128','affine8_block128','programmed_mean8']:
        try:
            if condition!='FP32':
                for stage,module in enumerate(modules):
                    w=weights[stage]
                    if condition=='programmed_mean8':
                        c=p['control_artifacts'][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);b=(-4+a['codes'].astype(float)*8/255).astype(np.float32);q=1/(1+np.exp(-2*b.astype(float)));ww=(np.repeat(a['lower'],128,0)+np.repeat(a['span'],128,0)*q).astype(np.float32)
                    else:
                        bits=4 if condition=='affine4_block128' else 8;ww=np.empty_like(w)
                        for first in range(0,len(w),128):
                            z=w[first:first+128];lo=np.minimum(z.min(0),0);hi=np.maximum(z.max(0),0);scale=(hi-lo)/((1<<bits)-1);safe=np.where(scale==0,1,scale);zero=np.clip(np.rint(-lo/safe),0,(1<<bits)-1);code=np.clip(np.rint(z/safe)+zero,0,(1<<bits)-1);ww[first:first+128]=(code-zero)*safe
                    with torch.no_grad():module.weight.copy_(torch.from_numpy(ww))
            nll=[];pred=[]
            with torch.inference_mode():
                for row in tokens:
                    ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];target=ids[:,1:];nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
            np.testing.assert_array_equal(np.stack(nll),baseraw[condition+'_nll']);np.testing.assert_array_equal(np.stack(pred),baseraw[condition+'_predictions'])
            ce=float(np.mean(np.stack(nll).astype(float)));ac=float(np.mean(np.stack(pred)==truth));rr=next(r for r in baserows if r['condition']==condition)
            for k,v in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':ac}.items():np.testing.assert_allclose(float(rr[k]),v,atol=1e-12,rtol=1e-13)
            print(f'Independent fresh baseline {condition}',flush=True)
        finally:
            with torch.no_grad():
                for module,w in zip(modules,original):module.weight.copy_(w)
    ref=baseraw['FP32_nll'].astype(float).mean(-1);refac=(baseraw['FP32_predictions']==truth).mean(-1);loss={};correct={};updates=0
    for method in p['methods']:
        nl=[];pr=[]
        for seed in range(5):
            folder=ROOT/'runs'/f'{method}_seed{seed}';r=json.loads((folder/'run.json').read_text());raw=np.load(folder/'prefix_tokens.npz');assert r['status']=='complete' and sha(folder/'prefix_tokens.npz')==r['token_archive_sha256'] and r['protocol_sha256']==sha(ROOT/'protocol.json') and r['exact_prefix_input_checks']==64*78 and len(r['stages'])==12
            total=0
            for stage,s in enumerate(r['stages']):
                meta=json.loads(Path(s['metadata']).read_text());assert sha(s['output'])==meta['output_sha256'] and s['inputs_sha256']==meta['signature']['inputs_sha256'];assert s['prior_native_output_sha256']==[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)]
                assert meta['signature']['method']==method and meta['signature']['stage']==stage and meta['signature']['seed']==seed and meta['signature']['samples']==1024;total+=meta['free_node_updates']
            assert total==r['free_node_updates']==8*1024*21233664;updates+=total
            for stage in range(13):
                nll,pred=raw[f'prefix{stage}_nll'],raw[f'prefix{stage}_predictions'];assert nll.shape==pred.shape==(64,64) and np.isfinite(nll).all();ce=float(nll.astype(float).mean());ac=float(np.mean(pred==truth));rr=next(a for a in prefix if a['method']==method and int(a['sampling_seed'])==seed and int(a['converted_prefix'])==stage)
                for k,v in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':ac}.items():np.testing.assert_allclose(float(rr[k]),v,atol=1e-12,rtol=1e-13)
                if stage==0:np.testing.assert_array_equal(nll,baseraw['FP32_nll']);np.testing.assert_array_equal(pred,baseraw['FP32_predictions'])
                if stage==12:
                    rr=next(a for a in rows if a['method']==method and int(a['sampling_seed'])==seed)
                    for k,v in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':ac}.items():np.testing.assert_allclose(float(rr[k]),v,atol=1e-12,rtol=1e-13);assert r['scores'][k]==v
            nl.append(raw['prefix12_nll'].astype(float).mean(-1));pr.append((raw['prefix12_predictions']==truth).mean(-1))
        loss[method]=np.stack(nl);correct[method]=np.stack(pr);delta=loss[method]-ref;adelta=correct[method]-refac;ce=bootstrap(delta);ac=bootstrap(adelta)
        np.testing.assert_allclose(draws[method+'_CE_vs_FP32'],ce,atol=1e-15,rtol=1e-13);np.testing.assert_allclose(draws[method+'_accuracy_vs_FP32'],ac,atol=1e-15,rtol=1e-13)
        up=float(np.quantile(ce,.975));lower=float(np.quantile(ac,.025));point=bool(delta.mean()<=np.log(1.05) and adelta.mean()>=-.01);passing=bool(point and up<=np.log(1.05) and lower>=-.01)
        values={'cross_entropy':loss[method].mean(),'perplexity':np.exp(loss[method].mean()),'relative_perplexity':np.exp(delta.mean()),'token_accuracy':correct[method].mean(),'accuracy_difference':adelta.mean(),'point_passed':point,'bootstrap_CE_upper':up,'bootstrap_relative_upper':np.exp(up),'bootstrap_accuracy_lower':lower,'passes':passing};rr=next(r for r in summary if r['method']==method);mm=next(r for r in m['profiles'] if r['method']==method)
        for k,v in values.items():
            if isinstance(v,bool):assert rr[k]==str(v) and mm[k]==v
            else:np.testing.assert_allclose(float(rr[k]),v,atol=1e-14,rtol=1e-13);np.testing.assert_allclose(mm[k],v,atol=1e-14,rtol=1e-13)
    ce=bootstrap(loss['weight_fisher']-loss['fixed']);ac=bootstrap(correct['weight_fisher']-correct['fixed']);np.testing.assert_allclose(draws['paired_CE'],ce,atol=1e-15,rtol=1e-13);np.testing.assert_allclose(draws['paired_accuracy'],ac,atol=1e-15,rtol=1e-13)
    primary=json.loads((ROOT/'paired_summary.json').read_text());delta=float((loss['weight_fisher']-loss['fixed']).mean());adelta=float((correct['weight_fisher']-correct['fixed']).mean())
    for k,v in {'CE_difference':delta,'perplexity_ratio':np.exp(delta),'accuracy_difference':adelta,'CE_sensitivity_low':np.quantile(ce,.025),'CE_sensitivity_high':np.quantile(ce,.975),'accuracy_sensitivity_low':np.quantile(ac,.025),'accuracy_sensitivity_high':np.quantile(ac,.975)}.items():np.testing.assert_allclose(primary[k],v,atol=1e-14,rtol=1e-13);np.testing.assert_allclose(m['primary'][k],v,atol=1e-14,rtol=1e-13)
    cal=list((ROOT/'calibration').glob('*.json'));assert len(cal)==120 and sum(json.loads(c.read_text())['free_node_evaluations'] for c in cal)==m['native_calibration_free_evaluations']==2*5*4096*21233664
    assert updates==m['native_inference_free_evaluations']==2*5*8*1024*21233664
    prep=json.loads((ROOT/'preparation_audit.json').read_text());assert prep['passed'] and prep['candidate_scores_checked_with_independent_inverse_Fisher']==1536
    modelaudit=json.loads((ROOT/'independent_model_audit.json').read_text());nativeaudit=json.loads((ROOT/'native_path_audit.json').read_text());assert modelaudit['passed'] and modelaudit['regenerated_input_hashes_exact']==120 and nativeaudit['passed'] and nativeaudit['native_operator_runs']==120 and nativeaudit['native_probe_windows']==240
    audit={'passed':True,'fresh_model_runs':10,'native_operator_runs':120,'native_probe_windows':240,'new_text_targets':4096,'independent_baselines':4,'bootstrap_values_recomputed':60000,'all_input_hashes_independently_regenerated':True,'all_native_counts_fits_and_outputs_independently_replayed':True,'all_task_scores_fixed_quality_rules_and_equal_budgets_recomputed':True,'source_sha256':sha(__file__),'manifest_sha256':sha(ROOT/'manifest.json'),
        'scope':'Fresh positions/devices for pre-frozen nominal weight-only probe design; no task selection or oracle in estimator. Unit-Fisher prediction is approximate, counts matched not hardware energy. Only25.92% ofonecheckpoint converted.'}
    (ROOT/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
