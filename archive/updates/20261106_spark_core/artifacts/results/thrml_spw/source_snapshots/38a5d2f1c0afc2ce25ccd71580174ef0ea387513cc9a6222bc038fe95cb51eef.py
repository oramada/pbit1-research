"""Full-weight range variance analysis and deterministic mean-model validation."""
import json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .sequential_transformer import SequentialRunner
from .localized_controls import expand,decode,ideal_variance
from .transformer_block_gpu import sha,save_json,save_array,write_csv
OUT=Path('results/thrml_spw/range_localization')
def main():
    p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'preparation_audit.json').read_text())['passed'];sources={str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['range_localization_analytic.py','localized_controls.py','sequential_transformer.py','transformer_block_gpu.py']};execution={'protocol_sha256':sha(OUT/'protocol.json'),'source_sha256':sources};ep=OUT/'analytic_execution.json'
    if ep.exists():assert json.loads(ep.read_text())==execution
    else:save_json(ep,execution)
    if (OUT/'analytic_manifest.json').exists():
        m=json.loads((OUT/'analytic_manifest.json').read_text());assert m['execution']==execution and sha(OUT/'mean_model_tokens.npz')==m['mean_model_tokens_sha256']
        for r in m['anchors']:assert sha(r['inputs'])==r['inputs_sha256'] and sha(r['target'])==r['target_sha256']
        print(json.dumps({'cached_complete':True,'new_native_samples':False}));return
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);tokens=np.load(OUT/'validation_tokens.npz')['tokens'];r=SequentialRunner(AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval(),p['targets_in_forward_order']);start=time.monotonic();fp=r.evaluate(tokens);raw={'FP32_nll':fp['nll'],'FP32_predictions':fp['predictions']};means=[{'condition':'FP32',**fp['scores']}];old=np.load('results/thrml_spw/sample_allocation/validation_baseline.npz');np.testing.assert_array_equal(fp['nll'],old['FP32_nll']);np.testing.assert_array_equal(fp['predictions'],old['FP32_predictions']);folder=OUT/'anchors';folder.mkdir(exist_ok=True);anchors=[];stats=[];comparisons=0
    indices=np.array([i*65+j for i in range(32) for j in p['anchor_positions']])
    for stage,w in enumerate(r.weights):
        value=r.evaluate(tokens,capture=stage);np.testing.assert_array_equal(value['nll'],fp['nll']);x=value['inputs'].reshape(-1,w.shape[0])[indices];target=x.astype(float)@w.astype(float)+r.biases[stage];xp=folder/f'stage{stage}_inputs.npy';yp=folder/f'stage{stage}_target.npy';save_array(xp,x);save_array(yp,target);anchors.append({'stage':stage,'inputs':str(xp),'inputs_sha256':sha(xp),'target':str(yp),'target_sha256':sha(yp),'input_shape':list(x.shape),'target_shape':list(target.shape)});previous=None;basevar=None;coarse=None
        for g in [128,64,32,16]:
            c=p['control_artifacts'][str(g)][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);lo,sp=expand(a,g);field=decode(a['codes']);q=1/(1+np.exp(-2*field.astype(float)));mean=lo.astype(float)+sp.astype(float)*q;variance=sp.astype(float)**2*q*(1-q);ideal=ideal_variance(w,g)
            if previous is not None:assert np.all(ideal<=previous);comparisons+=w.size
            previous=ideal
            if basevar is None:basevar=variance
            ym=x.astype(float)@mean+r.biases[stage];one=float(np.mean(x.astype(float)**2@variance));bias=float(np.mean((ym-target)**2));pred512=bias+one/512;pred1024=bias+one/1024
            if g==128:coarse=pred1024
            stats.append({'stage':stage,'module':p['targets_in_forward_order'][stage],'group':g,'weights':int(w.size),'groups':int(a['lower'].size),'parameter_payload_bytes':c['parameter_payload_bytes'],'group_flag_bytes':c['group_flag_bytes'],'ideal_weight_variance_mean':float(ideal.mean()),'actual_weight_variance_mean':float(variance.mean()),'finite_variance_increases_vs_g128_coordinates':int(np.count_nonzero(variance>basevar)),'programmed_weight_MSE':float(np.mean((mean-w.astype(float))**2)),'mean_Y_bias_MSE':bias,'one_draw_Y_variance':one,'predicted_target_MSE_T512':pred512,'predicted_target_MSE_T1024':pred1024,'halfT_MSE_ratio_vs_coarse1024':pred512/coarse})
        print(f'All-weight nested ranges and FP32 anchors checked stage{stage}/23',flush=True)
    for g in [128,64,32,16]:
        try:
            with torch.no_grad():
                for module,c in zip(r.modules,p['control_artifacts'][str(g)]):
                    a=np.load(c['path']);lo,sp=expand(a,g);q=1/(1+np.exp(-2*decode(a['codes']).astype(float)));module.weight.copy_(torch.from_numpy((lo.astype(float)+sp.astype(float)*q).astype(np.float32)))
            value=r.evaluate(tokens);raw[f'g{g}_nll']=value['nll'];raw[f'g{g}_predictions']=value['predictions'];means.append({'condition':f'programmed_mean_g{g}',**value['scores']})
            if g==128:np.testing.assert_array_equal(value['nll'],old['mean_nll']);np.testing.assert_array_equal(value['predictions'],old['mean_predictions'])
            print(f'Conventional programmed-mean validation g{g}: {value["scores"]}',flush=True)
        finally:r.restore()
    np.savez_compressed(OUT/'mean_model_tokens.npz',**raw);write_csv(OUT/'mean_model_metrics.csv',means);write_csv(OUT/'analytic_operator_metrics.csv',stats);assert comparisons==3*42467328
    m={'execution':execution,'anchors':anchors,'all_coordinate_nested_ideal_variance_passed':True,'coordinate_inequalities_checked':comparisons,'mean_model_forward_conditions':5,'mean_model_tokens_sha256':sha(OUT/'mean_model_tokens.npz'),'analytic_metrics_sha256':sha(OUT/'analytic_operator_metrics.csv'),'elapsed_seconds':time.monotonic()-start,'native_samples_collected':0,'new_heldout_test_quality_computed':False,'scope':'Deterministic expected-weight model diagnostics on consumed validation, not expectation of a stochastic network. Ideal range theorem and actual finite controller variance analyzed separately; anchors are real FP32 inputs for full-size native operator validation.'};save_json(OUT/'analytic_manifest.json',m);print(json.dumps({k:v for k,v in m.items() if k!='anchors'},indent=2))
if __name__=='__main__':main()
