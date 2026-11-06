"""Independent original-model, label/metric and public-native validation audit."""
import json,subprocess,csv
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .gpu_environment import cuda_environment
from .transformer_block_gpu import sha,save_json,save_array
OUT=Path('results/thrml_spw/sample_allocation')

def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'validation_manifest.json').read_text());assert m['native_operators']==120 and not m['test_scores_computed'];assert sha(OUT/'tokens.npz')==p['tokens_sha256'] and sha(OUT/'validation_baseline.npz')==m['baseline_archive_sha256']
    for name,h in m['execution']['source_sha256'].items():assert sha(name)==h
    tokens=np.load(OUT/'tokens.npz')['validation_tokens'];truth=tokens[:,1:];model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];base=np.load(OUT/'validation_baseline.npz');bm=json.loads((OUT/'validation_baseline.json').read_text())
    def evaluate():
        losses=[];predictions=[]
        with torch.inference_mode():
            for i,row in enumerate(tokens):
                nonlocal case
                case=i;ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];losses.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').numpy());predictions.append(logits.argmax(-1).numpy()[0])
        return np.stack(losses),np.stack(predictions)
    case=0
    for kind in ['FP32','mean']:
        if kind=='mean':
            with torch.no_grad():
                for module,c in zip(modules,p['control_artifacts']):
                    assert sha(c['path'])==c['sha256'];a=np.load(c['path']);field=(-4+a['codes'].astype(float)*8/255).astype(np.float32);q=1/(1+np.exp(-2*field.astype(float)));w=(np.repeat(a['lower'],128,0)+np.repeat(a['span'],128,0)*q).astype(np.float32);module.weight.copy_(torch.from_numpy(w))
        nll,pred=evaluate();np.testing.assert_array_equal(nll,base[kind+'_nll']);np.testing.assert_array_equal(pred,base[kind+'_predictions']);scores={'cross_entropy':float(nll.astype(float).mean()),'perplexity':float(np.exp(nll.astype(float).mean())),'token_accuracy':float(np.mean(pred==truth))};rr=bm['FP32' if kind=='FP32' else 'programmed_mean8_all24']
        for key,value in scores.items():np.testing.assert_allclose(rr[key],value,atol=1e-12,rtol=1e-12)
    log=(OUT/'validation_native_audit_worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.sample_allocation_native_audit_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment());scratch=OUT/'validation/audit_work';scratch.mkdir(exist_ok=True);results=[]
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Auditor exited {worker.poll()}')
        r=json.loads(line)
        if r['status']=='error':raise RuntimeError(str(r))
        return r
    with (OUT/'validation_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    try:
        assert reply()['status']=='ready'
        for record in m['cases']:
            stage,seed=record['stage'],record['seed'];assert sha(record['output'])==record['output_sha256'] and sha(record['tokens'])==record['tokens_sha256'];yp=np.load(record['output'],mmap_mode='r');captured=[]
            def before(module,args):captured.append(args[0].detach().numpy().copy())
            def after(module,args,value):return torch.from_numpy(np.asarray(yp[case:case+1]).copy()).to(dtype=value.dtype,device=value.device)
            hooks=[modules[stage].register_forward_pre_hook(before),modules[stage].register_forward_hook(after)]
            try:nll,pred=evaluate()
            finally:
                for hook in hooks:hook.remove()
            raw=np.load(record['tokens']);np.testing.assert_array_equal(nll,raw['nll']);np.testing.assert_array_equal(pred,raw['predictions']);scores={'cross_entropy':float(nll.astype(float).mean()),'perplexity':float(np.exp(nll.astype(float).mean())),'token_accuracy':float(np.mean(pred==truth))};rr=next(v for v in rows if int(v['stage'])==stage and int(v['seed'])==seed)
            for key,value in scores.items():np.testing.assert_allclose(record['scores'][key],value,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(float(rr[key]),value,atol=1e-12,rtol=1e-12)
            xp=scratch/'regenerated_inputs.npy';save_array(xp,np.concatenate(captured));assert sha(xp)==record['inputs_sha256'];worker.stdin.write(json.dumps({'inputs':str(xp),'input_sha256':sha(xp),'output':record['output']})+'\n');worker.stdin.flush();assert reply()['bitwise_exact'];xp.unlink();results.append({'stage':stage,'seed':seed,'exact_input_hash':True,'exact_token_losses_predictions':True});print(f'Validation audit stage{stage} seed{seed}: exact publicF32 path and labels',flush=True)
        worker.stdin.write('{"action":"finish"}\n');worker.stdin.flush();assert reply()['operators']==120
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    assert len(results)==120 and len(rows)==120;assert sum(c['observed_node_updates'] for c in m['cases'])==m['native_observed_updates']==5*4*256*42467328
    r={'passed':True,'baseline_model_paths_exact':2,'isolated_model_cases_exact':120,'native_inputs_regenerated_exact':120,'all_raw_losses_predictions_scores_checked':True,'native_audit_sha256':sha(OUT/'validation_native_audit.json'),'validation_manifest_sha256':sha(OUT/'validation_manifest.json'),'source_sha256':sha(__file__),'test_model_scores_computed':False,'rows':results,'scope':'Unchanged original-model hooks and independently reconstructed programmedmean weights verify all val baselines, every isolated case, all labels/scores and exact native input hashes. Public native count/F32/envelope replay separate. No test text model scoring.'};save_json(OUT/'validation_evidence_audit.json',r);print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
