"""Independent full-model baselines, raw scores, bootstrap rules and resource audit."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM
from .verify_transformer_control import sha
OUT=Path('results/thrml_spw/calibrated_transformer')


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def summarize_raw(nll,pred,baseline_nll,baseline_pred,tokens):
    losses=nll.astype(float).mean(-1);ref=baseline_nll.astype(float).mean(-1);accuracy=(pred==tokens[None,:,1:]).mean(-1);refacc=(baseline_pred==tokens[:,1:]).mean(-1)
    delta=losses-ref;adelta=accuracy-refacc
    rg=np.random.default_rng(2026114701);si=rg.integers(0,5,(10000,5));bi=rg.integers(0,8,(10000,8))
    # Independently cross resampled model-state seeds with whole8-context banks.
    ld=delta.reshape(5,8,8).mean(-1);aa=adelta.reshape(5,8,8).mean(-1)
    ce=np.array([ld[si[i]][:,bi[i]].mean() for i in range(10000)])
    ac=np.array([aa[si[i]][:,bi[i]].mean() for i in range(10000)])
    upper=float(np.quantile(ce,.975));lower=float(np.quantile(ac,.025));point=bool(delta.mean()<=np.log(1.05) and adelta.mean()>=-.01)
    r={'samples':1024,'cross_entropy_mean':float(losses.mean()),'cross_entropy_mc95_margin':float(1.96*losses.mean(-1).std(ddof=1)/np.sqrt(5)),
        'perplexity':float(np.exp(losses.mean())),'relative_perplexity':float(np.exp(delta.mean())),'token_accuracy_mean':float(accuracy.mean()),'accuracy_difference':float(adelta.mean()),
        'point_tolerance_passed':point,'bootstrap_CE_upper':upper,'bootstrap_relative_upper':float(np.exp(upper)),'bootstrap_accuracy_difference_lower':lower,
        'passes':bool(point and upper<=np.log(1.05) and lower>=-.01),'bootstrap_seed':2026114701,'bootstrap_resamples':10000}
    return r,ce,ac


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['config']==p and m['native_forward_runs']==15 and m['native_operator_runs']==180 and m['native_probe_windows']==120
    assert sha(OUT/'protocol.json')==m['execution']['prepared_design_sha256'] and sha(OUT/'tokens.npz')==p['tokens_sha256'];tokens=np.load(OUT/'tokens.npz')['tokens']
    for n,h in m['execution']['source_sha256'].items():assert sha(n)==h
    rows=read(OUT/'native_metrics.csv');prefixes=read(OUT/'prefix_metrics.csv');base=read(OUT/'baseline_metrics.csv');summaries=read(OUT/'summary.csv');assert len(rows)==15 and len(prefixes)==195 and len(base)==4 and len(summaries)==3
    assert {(r['condition'],int(r['sampling_seed'])) for r in rows}=={(c,s) for c in ['ideal','uncalibrated','cal4096'] for s in range(5)}
    baseraw=np.load(OUT/'baseline_tokens.npz');snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}"
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];original=[a.weight.detach().clone() for a in modules]
    weights=np.load(p['master_weights_path']);assert sha(p['master_weights_path'])==p['master_weights_sha256']
    controls=[]
    for s,c in enumerate(p['control_artifacts']):
        assert sha(c['path'])==c['sha256'];a=np.load(c['path']);controls.append(a)
        np.testing.assert_array_equal(modules[s].weight.detach().numpy(),weights[f'weight_{s}']);np.testing.assert_array_equal(modules[s].bias.detach().numpy(),a['bias'])
    for condition in ['FP32','affine4_block128','affine8_block128','allfree_programmed_mean8']:
        try:
            if condition!='FP32':
                for s,module in enumerate(modules):
                    w=weights[f'weight_{s}'];a=controls[s]
                    if condition=='allfree_programmed_mean8':
                        b=(-4+a['codes'].astype(float)*8/255).astype(np.float32);q=1/(1+np.exp(-2*b.astype(float)));expected=(np.repeat(a['lower'],128,0)+np.repeat(a['span'],128,0)*q).astype(np.float32)
                    else:
                        # Independently implement the affine-zero block control.
                        bits=4 if condition=='affine4_block128' else 8;expected=np.empty_like(w)
                        for start in range(0,len(w),128):
                            z=w[start:start+128];lo=z.min(0);hi=z.max(0);scale=(hi-lo)/((1<<bits)-1);safe=np.where(scale==0,1,scale);zero=np.rint(-lo/safe)
                            codes=np.clip(np.rint(z/safe)+zero,0,(1<<bits)-1);expected[start:start+128]=(codes-zero)*safe
                    with torch.no_grad():module.weight.copy_(torch.from_numpy(expected))
            nn=[];pp=[]
            with torch.inference_mode():
                for row in tokens:
                    ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];target=ids[:,1:]
                    nn.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').numpy());pp.append(logits.argmax(-1).numpy()[0])
            np.testing.assert_array_equal(np.stack(nn),baseraw[f'{condition}_nll']);np.testing.assert_array_equal(np.stack(pp),baseraw[f'{condition}_predictions'])
            print(f'Independent full-model baseline {condition}',flush=True)
        finally:
            with torch.no_grad():
                for module,w in zip(modules,original):module.weight.copy_(w)
    total=0
    for condition in ['ideal','uncalibrated','cal4096']:
        nl=[];pr=[]
        for seed in range(5):
            folder=OUT/'runs'/f'{condition}_seed{seed}';run=json.loads((folder/'run.json').read_text());raw=np.load(folder/'prefix_tokens.npz');assert run['status']=='complete' and run['exact_prefix_input_checks']==64*78
            assert len(run['stages'])==12 and run['design_sha256']==sha(OUT/'protocol.json');updates=0
            for s,stage in enumerate(run['stages']):
                meta=json.loads(Path(stage['metadata']).read_text());assert sha(stage['output'])==meta['output_sha256'] and sha(stage['inputs'])==meta['signature']['input_sha256'] and sha(stage['reference'])==meta['signature']['reference_sha256']
                assert stage['prior_native_output_sha256']==[sha(folder/f'stage{i}_native_output.npy') for i in range(s)]
                updates+=meta['free_node_updates']
            assert run['free_node_updates']==updates==8*1024*21233664;total+=updates
            for stage in range(13):
                nll=raw[f'prefix{stage}_nll'];pred=raw[f'prefix{stage}_predictions'];assert nll.shape==pred.shape==(64,64) and np.isfinite(nll).all()
                score={'cross_entropy':float(nll.astype(float).mean()),'perplexity':float(np.exp(nll.astype(float).mean())),'token_accuracy':float(np.mean(pred==tokens[:,1:]))}
                rr=next(r for r in prefixes if r['condition']==condition and int(r['sampling_seed'])==seed and int(r['converted_prefix'])==stage)
                for k,v in score.items():np.testing.assert_allclose(float(rr[k]),v,atol=1e-12,rtol=1e-12)
                if stage==12:
                    rr=next(r for r in rows if r['condition']==condition and int(r['sampling_seed'])==seed)
                    for k,v in score.items():np.testing.assert_allclose(float(rr[k]),v,atol=1e-12,rtol=1e-12)
                    for k,v in score.items():assert run['scores'][k]==v
                if stage==0:
                    np.testing.assert_array_equal(nll,baseraw['FP32_nll']);np.testing.assert_array_equal(pred,baseraw['FP32_predictions'])
            nl.append(raw['prefix12_nll']);pr.append(raw['prefix12_predictions'])
        r,ce,ac=summarize_raw(np.stack(nl),np.stack(pr),baseraw['FP32_nll'],baseraw['FP32_predictions'],tokens)
        saved=np.load(OUT/f'{condition}_uncertainty_draws.npz');np.testing.assert_allclose(saved['CE_difference'],ce,atol=1e-15);np.testing.assert_allclose(saved['accuracy_difference'],ac,atol=1e-15)
        rr=next(a for a in summaries if a['condition']==condition);mm=next(a for a in m['profiles'] if a['condition']==condition)
        for k,v in r.items():
            if isinstance(v,bool):assert rr[k]==str(v) and mm[k]==v
            else:np.testing.assert_allclose(float(rr[k]),v,atol=1e-14,rtol=1e-13);np.testing.assert_allclose(mm[k],v,atol=1e-14,rtol=1e-13)
    assert total==m['total_native_inference_free_evaluations'] and m['total_native_calibration_free_evaluations']==5*4096*21233664
    hooks=json.loads((OUT/'independent_hook_audit.json').read_text());assert hooks['passed'] and hooks['completed_runs_audited']==15
    native=json.loads((OUT/'native_path_audit.json').read_text());assert native['passed'] and native['native_operator_runs']==180 and native['native_probe_windows']==120
    correction=json.loads((OUT/'postprocessing_correction.json').read_text());assert correction['reporting_only'] and not correction['new_native_sampling_performed'] and not correction['raw_cases_changed']
    audit={'passed':True,'native_full_quarter_runs':15,'native_operator_runs':180,'native_probe_windows':120,'new_text_targets':4096,'all_baselines_scores_prefix_links_bootstrap_draws_and_fixed_gates_recomputed':True,'all_final_model_inputs_and_losses_independently_hook_replayed':True,'all_native_nodes_independently_replayed':True,'source_sha256':sha(__file__),
        'scope':'Complete selected quarter-model forward, every native probe/operator and all scored targets on64 new short windows. Synthetic static devices and normalized signed weights. Reporting shape correction retained. No full-model, long-context, physical-cost or architecture-superiority claim.'}
    (OUT/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
