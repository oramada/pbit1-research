"""Regenerate all native inputs by independent original-model hooks and replay."""
import json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .gpu_environment import cuda_environment
from .transformer_block_gpu import sha,save_json,save_array
OUT=Path('results/thrml_spw/all_blocks_clean')

def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['native_operator_runs']==120
    for name,h in m['execution']['source_sha256'].items():assert sha(name)==h
    tokens=np.load(OUT/'tokens.npz')['tokens'];assert sha(OUT/'tokens.npz')==p['tokens_sha256'];snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}";model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];scratch=OUT/'audit_work';scratch.mkdir(exist_ok=True);log=(OUT/'native_audit_worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.all_blocks_native_audit_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Audit worker exited {worker.poll()}')
        r=json.loads(line)
        if r['status']=='error':raise RuntimeError(str(r))
        return r
    results=[]
    try:
        assert reply()['status']=='ready'
        for seed in range(5):
            folder=OUT/'runs'/f'seed{seed}';record=json.loads((folder/'run.json').read_text());raw=np.load(folder/'prefix_tokens.npz');assert sha(folder/'prefix_tokens.npz')==record['token_archive_sha256'];outputs=[np.load(s['output'],mmap_mode='r') for s in record['stages']];inputs=[[] for _ in modules];case=0
            for prefix in [12,24]:
                hooks=[]
                for stage,module in enumerate(modules[:prefix]):
                    def before(module,args,_stage=stage):inputs[_stage].append(args[0].detach().numpy().copy())
                    def after(module,args,result,_stage=stage):return torch.from_numpy(np.asarray(outputs[_stage][case:case+1]).copy()).to(dtype=result.dtype,device=result.device)
                    if prefix==24:hooks.append(module.register_forward_pre_hook(before))
                    hooks.append(module.register_forward_hook(after))
                nll=[];pred=[]
                try:
                    with torch.inference_mode():
                        for case,row in enumerate(tokens):
                            ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];target=ids[:,1:];nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
                finally:
                    for hook in hooks:hook.remove()
                np.testing.assert_array_equal(np.stack(nll),raw[f'prefix{prefix}_nll']);np.testing.assert_array_equal(np.stack(pred),raw[f'prefix{prefix}_predictions'])
            for stage,s in enumerate(record['stages']):
                xp=scratch/'regenerated_inputs.npy';x=np.concatenate(inputs[stage]);save_array(xp,x);assert sha(xp)==s['inputs_sha256'] and list(x.shape)==s['input_shape'];worker.stdin.write(json.dumps({'stage':stage,'seed':seed,'regenerated_inputs':str(xp),'expected_input_sha256':s['inputs_sha256']})+'\n');worker.stdin.flush();response=reply();assert response['status']=='ok';xp.unlink();inputs[stage]=[];print(f'Independent all-block replay seed{seed} stage{stage}: {response["scaled_output_error"]:.3g}',flush=True)
            results.append({'seed':seed,'exact_regenerated_inputs':24,'first12_and_all24_token_metrics_exact':True});del inputs,outputs
        worker.stdin.write('{"action":"finish"}\n');worker.stdin.flush();assert reply()['operators']==120
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    r={'passed':True,'full_model_cases':5,'paired_prefix_model_cases':5,'exact_input_hashes':120,'first12_and_all24_token_metrics_exact':True,'temporary_inputs_removed_after_verified_replay':True,'source_sha256':sha(__file__),'runs':results,'scope':'Independent unchanged-model hooks reproduce first12 and all24 paths, every original input NPY SHA and per-token losses/predictions. All inputs supplied to separate native replay before removing regenerable copies.'};save_json(OUT/'independent_model_audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
