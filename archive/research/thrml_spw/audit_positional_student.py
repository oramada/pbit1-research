"""Independent original-model causal inputs, packed outputs and all final metrics."""
import json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .positional_student_common import OUT,sha,guard,check_prerequisites
from .fine_range_common import atomic_npy,atomic_json
from .gpu_environment import cuda_environment

def decode(path):
    with np.load(path) as a:
        assert str(a['tag'])=='spw-f32-byteplanes-v1';shape=tuple(a['shape']);planes=a['byteplanes'];bits=np.empty(int(np.prod(shape))*4,np.uint8)
        for byte in range(4):bits[byte::4]=planes[byte]
        return bits.view('<f4').reshape(shape)

def main():
    p=json.loads((OUT/'protocol.json').read_text());check_prerequisites(p);m=json.loads((OUT/'manifest.json').read_text());assert m['protocol_sha256']==sha(OUT/'protocol.json') and m['native_model_cases']==10 and m['native_projection_bank_calls']==960;torch.set_num_threads(8);torch.use_deterministic_algorithms(True);model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(e['module']) for e in p['matrices']];tokens=np.load(p['tokens_path'])['tokens'];baseline=np.load(OUT/'baseline_tokens.npz');assert sha(OUT/'baseline_tokens.npz')==m['baseline_tokens_sha256'];work=OUT/'audit_work';work.mkdir(exist_ok=True);xp=work/'inputs.npy'
    def evaluate():
        losses=[];pred=[]
        with torch.inference_mode():
            for first in range(0,32,8):
                ids=torch.from_numpy(tokens[first:first+8]);z=model(input_ids=ids,use_cache=False).logits[:,:-1];losses.append(F.cross_entropy(z.reshape(-1,z.shape[-1]),ids[:,1:].reshape(-1),reduction='none').reshape(8,64).numpy());pred.append(z.argmax(-1).numpy())
        return np.concatenate(losses),np.concatenate(pred)
    loss,pred=evaluate();np.testing.assert_array_equal(loss,baseline['nll']);np.testing.assert_array_equal(pred,baseline['predictions']);environment=cuda_environment();environment['XLA_FLAGS']='--xla_gpu_autotune_level=0';log=(OUT/'audit_worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.positional_student_audit_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=environment)
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Independent native auditor exited {worker.poll()}')
        r=json.loads(line)
        if r['status']=='error':raise RuntimeError(str(r))
        return r
    cases=[]
    try:
        assert reply()['status']=='ready'
        for cfg in p['conditions']:
            for seed in p['sampling_seeds']:
                folder=OUT/'runs'/f'{cfg["name"]}_seed{seed}';run=json.loads((folder/'run.json').read_text());assert run['protocol_sha256']==sha(OUT/'protocol.json') and sha(folder/'tokens.npz')==run['tokens_sha256'];records={(v['bank'],v['stage']):v for v in run['bank_calls']};captured=[[] for _ in modules];hooks=[]
                for stage,module in enumerate(modules):
                    def before(module,args,index=stage):captured[index].append(args[0].detach().numpy().copy())
                    def after(module,args,output,index=stage):
                        bank=len(captured[index])-1;record=records[bank,index];assert sha(record['output'])==record['output_sha256'];return torch.from_numpy(decode(record['output'])).to(dtype=output.dtype,device=output.device)
                    hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
                try:loss,pred=evaluate()
                finally:
                    for hook in hooks:hook.remove()
                raw=np.load(folder/'tokens.npz');np.testing.assert_array_equal(loss,raw['nll']);np.testing.assert_array_equal(pred,raw['predictions']);ce=float(loss.astype(float).mean());derived={'cross_entropy':ce,'perplexity':float(np.exp(ce)),'token_accuracy':float(np.mean(pred==tokens[:,1:]))}
                for name,value in derived.items():assert value==run['scores'][name]
                for bank in range(4):
                    for stage in range(24):
                        record=records[bank,stage];atomic_npy(xp,captured[stage][bank]);assert sha(xp)==record['input_sha256'];worker.stdin.write(json.dumps({'input':str(xp),'input_sha256':sha(xp),'output':record['output']})+'\n');worker.stdin.flush();answer=reply();assert answer['bitwise_exact'] and answer['IEEE_envelope_passed'];xp.unlink()
                    print(f'Independent causal student {cfg["name"]} seed{seed} bank{bank}: all24 exact inputs/public outputs',flush=True)
                cases.append({'condition':cfg['name'],'seed':seed,'actual_input_hashes_exact':96,'all_token_losses_predictions_exact':True})
        worker.stdin.write('{"action":"finish"}\n');worker.stdin.flush();answer=reply();assert answer['passed'] and answer['bank_calls']==960;guard(p);result={'passed':True,'cases':cases,'model_cases':10,'all_actual_input_hashes_exact':960,'FP32_baseline_and_all_final_token_metrics_exact':True,'native_audit_sha256':sha(OUT/'native_audit.json'),'production_manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':sha(__file__),'scope':'Original-model pre/post hooks independently reconstruct every actual stochastic input and exact final metrics; separate public native sum and counts audits cover every bank. Independent byte-plane decoder. Consumed validation, no hardware proof.'};atomic_json(OUT/'model_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.stdin.close();worker.stdout.close();worker.wait()
        log.close()
if __name__=='__main__':main()
