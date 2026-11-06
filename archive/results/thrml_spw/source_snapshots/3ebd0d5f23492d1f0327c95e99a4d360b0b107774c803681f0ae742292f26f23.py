"""Independent FP32 anchors, range metrics and all mean-weight neural forwards."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .transformer_block_gpu import sha,save_array,save_json
OUT=Path('results/thrml_spw/range_localization')
def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'analytic_manifest.json').read_text());assert m['execution']['protocol_sha256']==sha(OUT/'protocol.json');tokens=np.load(OUT/'validation_tokens.npz')['tokens'];raw=np.load(OUT/'mean_model_tokens.npz');model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];original=[v.weight.detach().clone() for v in modules];weights=[v.numpy().copy() for v in original];biases=[v.bias.detach().numpy().copy() for v in modules];captured=[[] for _ in modules];hooks=[]
    for stage,module in enumerate(modules):
        def before(module,args,_stage=stage):captured[_stage].append(args[0].detach().numpy().copy())
        hooks.append(module.register_forward_pre_hook(before))
    with (OUT/'mean_model_metrics.csv').open() as f:meanrows=list(csv.DictReader(f))
    def evaluate():
        nll=[];pred=[]
        with torch.inference_mode():
            for row in tokens:
                ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
        return np.stack(nll),np.stack(pred)
    try:loss,pred=evaluate()
    finally:
        for hook in hooks:hook.remove()
    np.testing.assert_array_equal(loss,raw['FP32_nll']);np.testing.assert_array_equal(pred,raw['FP32_predictions']);scratch=OUT/'analytic_audit_work';scratch.mkdir(exist_ok=True);rows={}
    with (OUT/'analytic_operator_metrics.csv').open() as f:
        for r in csv.DictReader(f):rows[int(r['stage']),int(r['group'])]=r
    assert len(rows)==96;indices=np.array([i*65+j for i in range(32) for j in [16,48]]);inequalities=0
    for stage,(w,anchor) in enumerate(zip(weights,m['anchors'])):
        x=np.concatenate(captured[stage]).reshape(-1,w.shape[0])[indices];xp=scratch/'inputs.npy';save_array(xp,x);assert sha(xp)==anchor['inputs_sha256']==sha(anchor['inputs']);xp.unlink();reference=x.astype(float)@w.astype(float)+biases[stage];np.testing.assert_array_equal(reference,np.load(anchor['target']));assert sha(anchor['target'])==anchor['target_sha256'];previous=None;basevar=None;coarse=None
        for group in [128,64,32,16]:
            c=p['control_artifacts'][str(group)][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);lo=np.repeat(a['lower'],group,0).astype(float);sp=np.repeat(a['span'],group,0).astype(float);b=(-4+a['codes'].astype(float)*8/255).astype(np.float32);q=1/(1+np.exp(-2*b.astype(float)));mu=lo+sp*q;variance=sp**2*q*(1-q);view=w.astype(float).reshape(-1,group,w.shape[1]);l=np.repeat(view.min(1),group,0);u=np.repeat(view.max(1),group,0);ideal=(w.astype(float)-l)*(u-w.astype(float))
            if previous is not None:assert np.all(ideal<=previous);inequalities+=w.size
            previous=ideal
            if basevar is None:basevar=variance
            mean=x.astype(float)@mu+biases[stage];one=float(np.mean((x.astype(float)**2)@variance));bias=float(np.mean((mean-reference)**2));pred512=bias+one/512;pred1024=bias+one/1024
            if group==128:coarse=pred1024
            rr=rows[stage,group];values={'ideal_weight_variance_mean':ideal.mean(),'actual_weight_variance_mean':variance.mean(),'programmed_weight_MSE':np.mean((mu-w.astype(float))**2),'mean_Y_bias_MSE':bias,'one_draw_Y_variance':one,'predicted_target_MSE_T512':pred512,'predicted_target_MSE_T1024':pred1024,'halfT_MSE_ratio_vs_coarse1024':pred512/coarse}
            for k,v in values.items():np.testing.assert_allclose(float(rr[k]),v,atol=1e-13,rtol=1e-11)
            assert int(rr['finite_variance_increases_vs_g128_coordinates'])==np.count_nonzero(variance>basevar) and int(rr['parameter_payload_bytes'])==sum(a[k].nbytes for k in ['codes','lower','span','bias'])
        print(f'Independent all-weight variance/anchor audit stage{stage}',flush=True)
    for group in [128,64,32,16]:
        try:
            with torch.no_grad():
                for module,c in zip(modules,p['control_artifacts'][str(group)]):
                    a=np.load(c['path']);lo=np.repeat(a['lower'],group,0).astype(float);sp=np.repeat(a['span'],group,0).astype(float);b=(-4+a['codes'].astype(float)*8/255).astype(np.float32);q=1/(1+np.exp(-2*b.astype(float)));module.weight.copy_(torch.from_numpy((lo+sp*q).astype(np.float32)))
            loss,pred=evaluate();np.testing.assert_array_equal(loss,raw[f'g{group}_nll']);np.testing.assert_array_equal(pred,raw[f'g{group}_predictions']);ce=float(loss.astype(float).mean());rr=next(v for v in meanrows if v['condition']==f'programmed_mean_g{group}')
            for k,v in {'cross_entropy':ce,'perplexity':np.exp(ce),'token_accuracy':np.mean(pred==tokens[:,1:])}.items():np.testing.assert_allclose(float(rr[k]),v,atol=1e-12,rtol=1e-12)
            print(f'Independent programmed-mean neural model g{group}',flush=True)
        finally:
            with torch.no_grad():
                for module,w in zip(modules,original):module.weight.copy_(w)
    assert inequalities==127401984
    for name,h in m['execution']['source_sha256'].items():assert sha(name)==h
    r={'passed':True,'anchors_exact_from_independent_hooks':24,'true_F64_operator_targets_exact':24,'analytic_rows_checked':96,'nested_coordinate_inequalities_checked':inequalities,'mean_neural_forward_conditions_exact':5,'source_sha256':sha(__file__),'scope':'Independent original-model hooks, source weights and controller codes verify all ranges/variances/targets and every deterministic mean-model token loss/prediction. Mean network is not expectation of stochastic network; no new heldout quality or full stochastic NN evidence.'};save_json(OUT/'analytic_evidence_audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
