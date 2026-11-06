"""Regenerate the witness input and replay the pinned numerical diagnostic."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .gpu_environment import cuda_environment
from .transformer_block_gpu import sha,save_array,save_json
OUT=Path('results/thrml_spw/all_blocks_clean')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args();ap=OUT/'rounding_witness/audit.json';a=json.loads(ap.read_text());assert a['passed'] and sha(OUT/'rounding_witness/arrays.npz')==a['arrays_sha256'];source=Path('research/thrml_spw/all_blocks_rounding_witness.py');assert sha(source)==a['source_sha256']
    if not args.replay:print(json.dumps({'cached_complete':True,'new_native_samples':False,'witness_audit_sha256':sha(ap)}));return
    original_audit_sha=sha(ap);torch.set_num_threads(8);torch.use_deterministic_algorithms(True);p=json.loads((OUT/'protocol.json').read_text());tokens=np.load(OUT/'tokens.npz')['tokens'];model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];folder=OUT/'runs/seed3';outputs=[np.load(folder/f'stage{s}_native_output.npy',mmap_mode='r') for s in range(23)];hooks=[];inputs=[];case=0
    for stage,module in enumerate(modules[:23]):
        def after(module,args,result,_stage=stage):return torch.from_numpy(np.asarray(outputs[_stage][case:case+1]).copy()).to(dtype=result.dtype,device=result.device)
        hooks.append(module.register_forward_hook(after))
    def before(module,args):inputs.append(args[0].detach().numpy().copy())
    hooks.append(modules[23].register_forward_pre_hook(before))
    try:
        with torch.inference_mode():
            for case,row in enumerate(tokens):model(input_ids=torch.from_numpy(row[None]),use_cache=False)
    finally:
        for h in hooks:h.remove()
    xp=OUT/'audit_work/regenerated_inputs.npy';save_array(xp,np.concatenate(inputs));assert sha(xp)==a['inputs_sha256'];log=(OUT/'rounding_witness/reproduction.log').open('w')
    try:subprocess.run([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.all_blocks_rounding_witness'],env=cuda_environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    finally:log.close()
    assert sha(ap)==original_audit_sha and sha(OUT/'rounding_witness/arrays.npz')==a['arrays_sha256'];xp.unlink();r={'passed':True,'exact_input_regenerated':True,'witness_source_pinned':True,'witness_audit_and_raw_array_bytes_unchanged':True,'native_observed_updates_executed':a['native_observed_updates_executed'],'source_sha256':sha(__file__),'scope':'An explicit same-key numerical diagnostic replay, not new quality evidence. Independent original-model hooks regenerate the temporary input; exact witness count/output/bound arrays and metadata remain unchanged. Additional verification work is included in resource ledger.'};save_json(OUT/'rounding_reproduction_audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
