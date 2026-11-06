"""Single-pass causal stochastic student comparison; no repeated prefix forwards."""
import json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .positional_student_common import OUT,sha,guard,capacity,check_prerequisites
from .fine_range_common import atomic_json,atomic_npy
from .fine_range_common import atomic_npz
from .packed_float_outputs import load_output
from .gpu_environment import cuda_environment

def scores(loss,pred,tokens):
    ce=float(loss.astype(float).mean());return {'cross_entropy':ce,'perplexity':float(np.exp(ce)),'token_accuracy':float(np.mean(pred==tokens[:,1:]))}

def main():
    p=json.loads((OUT/'protocol.json').read_text());check_prerequisites(p);capacity(p);assert not (OUT/'manifest.json').exists();torch.set_num_threads(8);torch.use_deterministic_algorithms(True);tokens=np.load(p['tokens_path'])['tokens'];model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(e['module']) for e in p['matrices']];original=[m.forward for m in modules];work=OUT/'work';work.mkdir(exist_ok=True);xp=work/'current_inputs.npy';start=time.monotonic();base_loss=[];base_pred=[]
    def token_metrics(logits,ids):
        shifted=logits[:,:-1];return F.cross_entropy(shifted.reshape(-1,shifted.shape[-1]),ids[:,1:].reshape(-1),reduction='none').reshape(8,64).numpy(),shifted.argmax(-1).numpy()
    with torch.inference_mode():
        for first in range(0,32,8):
            ids=torch.from_numpy(tokens[first:first+8]);loss,pred=token_metrics(model(input_ids=ids,use_cache=False).logits,ids);base_loss.append(loss);base_pred.append(pred)
    base_loss=np.concatenate(base_loss);base_pred=np.concatenate(base_pred);baseline=OUT/'baseline_tokens.npz'
    if baseline.exists():
        old=np.load(baseline);np.testing.assert_array_equal(old['nll'],base_loss);np.testing.assert_array_equal(old['predictions'],base_pred)
    else:atomic_npz(baseline,nll=base_loss,predictions=base_pred)
    env=cuda_environment();env['XLA_FLAGS']='--xla_gpu_autotune_level=0';log=(OUT/'worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.positional_student_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env);atomic_json(OUT/'LIVE_CHECKPOINT.json',{'phase':'native_student_production','parent_pid':__import__('os').getpid(),'worker_pid':worker.pid,'started_scoring':True,'protocol_sha256':sha(OUT/'protocol.json')})
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Native student worker exited {worker.poll()}')
        r=json.loads(line)
        if r['status']=='error':raise RuntimeError(str(r))
        return r
    cases=[]
    try:
        assert reply()['status']=='ready'
        for seed in p['sampling_seeds']:
            for cfg in p['conditions']:
                name=cfg['name'];folder=OUT/'runs'/f'{name}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);run=folder/'run.json'
                if run.exists():
                    old=json.loads(run.read_text());assert old['protocol_sha256']==sha(OUT/'protocol.json') and sha(folder/'tokens.npz')==old['tokens_sha256']
                    for bank in old['bank_calls']:assert sha(bank['output'])==bank['output_sha256']
                    cases.append(old);continue
                records=[];losses=[];predictions=[]
                for bank,first in enumerate(range(0,32,8)):
                    order=[]
                    for stage,module in enumerate(modules):
                        def replaced(inputs,*args,_stage=stage,**kwargs):
                            assert inputs.shape==(8,65,p['matrices'][_stage]['shape'][0]);order.append(_stage);atomic_npy(xp,inputs.detach().numpy());path=folder/f'stage{_stage}_bank{bank}.npz';request={'stage':_stage,'seed':seed,'bank':bank,'condition':name,'input':str(xp),'input_sha256':sha(xp),'output':str(path)};worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();answer=reply();meta=json.loads(Path(answer['metadata']).read_text());assert sha(path)==meta['output_sha256'];records.append({'stage':_stage,'bank':bank,'input_sha256':sha(xp),'output':str(path),'output_sha256':meta['output_sha256'],'metadata':answer['metadata'],'native_node_updates':meta['native_node_updates']});return torch.from_numpy(load_output(path)).to(dtype=inputs.dtype,device=inputs.device)
                        module.forward=replaced
                    try:
                        with torch.inference_mode():
                            ids=torch.from_numpy(tokens[first:first+8]);loss,pred=token_metrics(model(input_ids=ids,use_cache=False).logits,ids)
                    finally:
                        for module,forward in zip(modules,original):module.forward=forward
                    assert order==list(range(24));losses.append(loss);predictions.append(pred);print(f'Student {name} seed{seed} bank{bank}: causal24 projections complete elapsed={time.monotonic()-start:.1f}s',flush=True)
                loss=np.concatenate(losses);pred=np.concatenate(predictions);atomic_npz(folder/'tokens.npz',nll=loss,predictions=pred);record={'condition':name,'seed':seed,'K':cfg['K'],'T':cfg['T'],'protocol_sha256':sha(OUT/'protocol.json'),'tokens_sha256':sha(folder/'tokens.npz'),'scores':scores(loss,pred,tokens),'bank_calls':records,'native_node_updates':sum(v['native_node_updates'] for v in records)};assert len(records)==96 and record['native_node_updates']==4*42467328*256;atomic_json(run,record);cases.append(record);guard(p);print(f'Complete student {name} seed{seed}: {record["scores"]}',flush=True)
        assert len(cases)==10;result={'completed':True,'protocol_sha256':sha(OUT/'protocol.json'),'baseline_tokens_sha256':sha(baseline),'FP32_scores':scores(base_loss,base_pred,tokens),'native_model_cases':10,'native_projection_bank_calls':960,'native_node_updates':sum(v['native_node_updates'] for v in cases),'cases':[{k:v for k,v in c.items() if k!='bank_calls'} for c in cases],'elapsed_seconds':time.monotonic()-start,'independently_audited':False,'new_test_scores_used':False,'scope':p['scope']};assert result['native_node_updates']==p['production_node_updates'];atomic_json(OUT/'manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    finally:
        for module,forward in zip(modules,original):module.forward=forward
        if worker.poll() is None:
            worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.stdin.close();worker.stdout.close();worker.wait()
        log.close()
if __name__=='__main__':main()
