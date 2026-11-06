"""Regenerate runtime inputs by independent hooks, then audit all native paths."""
import json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .gpu_environment import cuda_environment
from .transformer_block_gpu import sha,save_array
ROOT=Path('results/thrml_spw/probe_placement')


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    p=json.loads((ROOT/'protocol.json').read_text());m=json.loads((ROOT/'manifest.json').read_text());assert m['native_forward_runs']==10
    for n,h in m['execution']['source_sha256'].items():assert sha(n)==h
    tokens=np.load(ROOT/'tokens.npz')['tokens'];assert sha(ROOT/'tokens.npz')==p['tokens_sha256']
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}";model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']]
    scratch=ROOT/'audit_work';scratch.mkdir(exist_ok=True);log=(ROOT/'native_audit_worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.probe_placement_native_audit_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=cuda_environment())
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Audit worker exited{worker.poll()}')
        r=json.loads(line)
        if r['status']=='error':raise RuntimeError(str(r))
        return r
    results=[]
    try:
        assert reply()['status']=='ready'
        for method in p['methods']:
            for seed in range(5):
                folder=ROOT/'runs'/f'{method}_seed{seed}';record=json.loads((folder/'run.json').read_text());raw=np.load(folder/'prefix_tokens.npz');assert sha(folder/'prefix_tokens.npz')==record['token_archive_sha256']
                outputs=[np.load(s['output'],mmap_mode='r') for s in record['stages']];inputs=[[] for _ in modules];hooks=[];case=0
                for stage,module in enumerate(modules):
                    def before(module,args,_stage=stage):inputs[_stage].append(args[0].detach().numpy().copy())
                    def after(module,args,result,_stage=stage):return torch.from_numpy(np.asarray(outputs[_stage][case:case+1]).copy()).to(dtype=result.dtype,device=result.device)
                    hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
                nll=[];pred=[]
                try:
                    with torch.inference_mode():
                        for case,row in enumerate(tokens):
                            ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];target=ids[:,1:]
                            nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
                finally:
                    for hook in hooks:hook.remove()
                np.testing.assert_array_equal(np.stack(nll),raw['prefix12_nll']);np.testing.assert_array_equal(np.stack(pred),raw['prefix12_predictions'])
                for stage,s in enumerate(record['stages']):
                    # Recreate one input artifact at a time, not another saved
                    # multi-case dataset. The exact original NPY SHA must match.
                    xp=scratch/'regenerated_inputs.npy';x=np.concatenate(inputs[stage]);save_array(xp,x);assert sha(xp)==s['inputs_sha256'];assert list(x.shape)==s['input_shape']
                    request={'method':method,'seed':seed,'stage':stage,'regenerated_inputs':str(xp),'expected_input_sha256':s['inputs_sha256']};worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=reply();assert response['status']=='ok'
                    xp.unlink();inputs[stage]=[]
                    print(f'Regenerated-input/native audit {method} seed{seed} stage{stage}, error={response["scaled_output_error"]:.3g}',flush=True)
                results.append({'method':method,'seed':seed,'exact_regenerated_input_hashes':12,'final_token_losses_exact':True,'final_token_predictions_exact':True})
                del inputs,outputs
        worker.stdin.write('{"action":"finish"}\n');worker.stdin.flush();assert reply()['operators']==120
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    assert len(results)==10
    audit={'passed':True,'native_model_runs':10,'regenerated_input_hashes_exact':120,'final_model_losses_predictions_exact':True,'temporary_inputs_removed_after_verified_replay':True,'source_sha256':sha(__file__),'runs':results,
        'scope':'Independent unchanged-model pre/post-hooks regenerate every input from pinned text/model and previous saved native outputs. All original NPY SHAs and final losses/predictions match; each regenerated input feeds separate native replay before its temporary file is removed. No unique stochastic evidence deleted.'}
    (ROOT/'independent_model_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
