"""First-case independent exact input regeneration, without new native sampling."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .transformer_block_gpu import sha,save_array
ROOT=Path('results/thrml_spw/probe_placement')


def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);p=json.loads((ROOT/'protocol.json').read_text());folder=ROOT/'runs/fixed_seed0';r=json.loads((folder/'run.json').read_text());assert r['status']=='complete'
    raw=np.load(folder/'prefix_tokens.npz');assert sha(folder/'prefix_tokens.npz')==r['token_archive_sha256'];tokens=np.load(ROOT/'tokens.npz')['tokens'];snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}"
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];outputs=[np.load(s['output'],mmap_mode='r') for s in r['stages']];inputs=[[] for _ in modules];hooks=[];case=0
    for stage,module in enumerate(modules):
        def before(module,args,_stage=stage):inputs[_stage].append(args[0].detach().numpy().copy())
        def after(module,args,result,_stage=stage):return torch.from_numpy(np.asarray(outputs[_stage][case:case+1]).copy()).to(dtype=result.dtype,device=result.device)
        hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    nll=[];pred=[]
    try:
        with torch.inference_mode():
            for case,row in enumerate(tokens):
                ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];target=ids[:,1:];nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
    finally:
        for hook in hooks:hook.remove()
    np.testing.assert_array_equal(np.stack(nll),raw['prefix12_nll']);np.testing.assert_array_equal(np.stack(pred),raw['prefix12_predictions'])
    work=ROOT/'preflight_work';work.mkdir(exist_ok=True);xp=work/'regenerated.npy'
    for stage,s in enumerate(r['stages']):
        x=np.concatenate(inputs[stage]);save_array(xp,x);assert sha(xp)==s['inputs_sha256'];xp.unlink()
    result={'passed':True,'native_case':'fixed_seed0','regenerated_input_hashes_exact':12,'final_losses_predictions_exact':True,'new_native_samples':0,'temporary_inputs_removed':True,'source_sha256':sha(__file__),'scope':'Only the first complete case: confirms input archive regeneration format and deterministic causal values. Full ten-case/native path audits still required.'}
    (ROOT/'input_regeneration_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
