"""Independent complete original-model and public native held-out path audit."""
import json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .gpu_environment import cuda_environment
from .transformer_block_gpu import sha,save_json,save_array
from .packed_float_outputs import load_output
OUT=Path('results/thrml_spw/fine_range_confirmation')

def main():
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert m['native_operators']==240 and m['native_model_cases']==10 and not m['test_scores_used_for_selection'];tokens=np.load(OUT/'tokens.npz')['test_tokens'];assert sha(OUT/'tokens.npz')==p['tokens_sha256']
    for name,h in m['execution']['source_sha256'].items():assert sha(name)==h
    audit_names=['audit_fine_range_paths.py','fine_range_audit_worker.py','range_fixed_replay_cohort.py','verify_range_localization_native.py','calibrated_mnist_path_audit.py','packed_float_outputs.py']
    audit_sources={str(Path('research/thrml_spw')/name):sha(Path('research/thrml_spw')/name) for name in audit_names};audit_execution={'protocol_sha256':sha(OUT/'protocol.json'),'production_manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':audit_sources};ap=OUT/'audit_execution.json'
    if ap.exists():assert json.loads(ap.read_text())==audit_execution
    else:save_json(ap,audit_execution)
    for name,h in audit_sources.items():
        archive=Path('results/thrml_spw/source_snapshots')/(h+'.py')
        if not archive.exists():archive.write_bytes(Path(name).read_bytes())
    model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']]
    baseline=np.load(OUT/'baseline_tokens.npz');bn=[];bp=[]
    with torch.inference_mode():
        for row in tokens:
            ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];bn.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').numpy());bp.append(logits.argmax(-1).numpy()[0])
    np.testing.assert_array_equal(np.stack(bn),baseline['FP32_nll']);np.testing.assert_array_equal(np.stack(bp),baseline['FP32_predictions'])
    environment=cuda_environment();environment['XLA_FLAGS']='--xla_gpu_autotune_level=0'
    scratch=OUT/'audit_work';scratch.mkdir(exist_ok=True);log=(OUT/'native_audit_worker.log').open('a');worker=subprocess.Popen([str(Path('.venv-thrml-gpu/bin/python').absolute()),'-u','-m','research.thrml_spw.fine_range_audit_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=environment)
    def reply():
        line=worker.stdout.readline()
        if not line:raise RuntimeError(f'Auditor exited {worker.poll()}')
        r=json.loads(line)
        if r['status']=='error':raise RuntimeError(str(r))
        return r
    results=[]
    try:
        assert reply()['status']=='ready'
        for condition in ['coarse','fine']:
            for seed in range(5):
                folder=OUT/'runs'/f'{condition}_seed{seed}';r=json.loads((folder/'run.json').read_text());assert r['execution']==m['execution'] and sha(folder/'prefix_tokens.npz')==r['token_archive_sha256'];raw=np.load(folder/'prefix_tokens.npz');outputs=[load_output(s['output']) for s in r['stages']];inputs=[[] for _ in modules];hooks=[];case=0
                for stage,module in enumerate(modules):
                    def before(module,args,_stage=stage):inputs[_stage].append(args[0].detach().numpy().copy())
                    def after(module,args,result,_stage=stage):return torch.from_numpy(np.asarray(outputs[_stage][case:case+1]).copy()).to(dtype=result.dtype,device=result.device)
                    hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
                nll=[];pred=[]
                try:
                    with torch.inference_mode():
                        for case,row in enumerate(tokens):
                            ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
                finally:
                    for hook in hooks:hook.remove()
                np.testing.assert_array_equal(np.stack(nll),raw['prefix24_nll']);np.testing.assert_array_equal(np.stack(pred),raw['prefix24_predictions'])
                for stage,s in enumerate(r['stages']):
                    xp=scratch/'regenerated_inputs.npy';save_array(xp,np.concatenate(inputs[stage]));assert sha(xp)==s['inputs_sha256'];worker.stdin.write(json.dumps({'inputs':str(xp),'input_sha256':sha(xp),'output':s['output']})+'\n');worker.stdin.flush();assert reply()['bitwise_exact'];xp.unlink();inputs[stage]=[];print(f'Fresh test audit {condition} seed{seed} stage{stage}: exact model/publicF32 output',flush=True)
                results.append({'condition':condition,'seed':seed,'input_hashes_exact':24,'raw_token_losses_predictions_exact':True});del inputs,outputs
        worker.stdin.write('{"action":"finish"}\n');worker.stdin.flush();assert reply()['operators']==240
    finally:
        if worker.poll() is None:worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();log.close()
    r={'passed':True,'model_cases_exact':10,'input_hashes_exact':240,'all_raw_token_losses_predictions_exact':True,'packed_outputs_lossless_and_used_as_values':True,'native_audit_sha256':sha(OUT/'native_audit.json'),'source_sha256':sha(__file__),'rows':results,'FP32_baseline_raw_losses_predictions_exact':True,'scope':'Independent original-model hooks regenerate every current stochastic input, exact NPY SHAs and final token metrics. Separate public defaultF32/dynamicfield execution and frequency/IEEE bounds cover all240 native operators; descriptive numerical flags preserved. One checkpoint, ideal devices, no physical proof.'};save_json(OUT/'model_audit.json',r);print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
