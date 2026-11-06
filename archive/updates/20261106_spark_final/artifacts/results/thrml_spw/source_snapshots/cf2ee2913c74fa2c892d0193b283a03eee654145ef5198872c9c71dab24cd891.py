"""Early read-only integration check; final complete native/model audits still required."""
import argparse,json,hashlib,io
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .fine_range_common import OUT,sha,atomic_json
from .packed_float_outputs import load_output
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=[f'{c}_seed{s}' for c in ['coarse','fine'] for s in range(5)],required=True);args=parser.parse_args();p=json.loads((OUT/'protocol.json').read_text());execution=json.loads((OUT/'execution.json').read_text());folder=OUT/'runs'/args.case;record=json.loads((folder/'run.json').read_text());assert record['status']=='complete' and record['execution']==execution and sha(folder/'prefix_tokens.npz')==record['token_archive_sha256'] and sha(OUT/'tokens.npz')==p['tokens_sha256']
    for name,h in execution['source_sha256'].items():assert sha(name)==h
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(n) for n in p['targets_in_forward_order']];outputs=[]
    for r in record['stages']:
        meta=json.loads(Path(r['metadata']).read_text());assert sha(r['output'])==meta['output_sha256'];outputs.append(load_output(r['output']))
    inputs=[[] for _ in modules];hooks=[];case=0
    for stage,module in enumerate(modules):
        def before(module,args,_stage=stage):inputs[_stage].append(args[0].detach().numpy().copy())
        def after(module,args,result,_stage=stage):return torch.from_numpy(np.asarray(outputs[_stage][case:case+1]).copy()).to(dtype=result.dtype,device=result.device)
        hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    tokens=np.load(OUT/'tokens.npz')['test_tokens'];nll=[];pred=[]
    try:
        with torch.inference_mode():
            for case,row in enumerate(tokens):
                ids=torch.from_numpy(row[None]);logits=model(input_ids=ids,use_cache=False).logits[:,:-1];nll.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').numpy());pred.append(logits.argmax(-1).numpy()[0])
    finally:
        for hook in hooks:hook.remove()
    raw=np.load(folder/'prefix_tokens.npz');np.testing.assert_array_equal(np.stack(nll),raw['prefix24_nll']);np.testing.assert_array_equal(np.stack(pred),raw['prefix24_predictions']);rows=[]
    for stage,r in enumerate(record['stages']):
        x=np.concatenate(inputs[stage]);stream=io.BytesIO();np.save(stream,x);h=hashlib.sha256(stream.getvalue()).hexdigest();assert h==r['inputs_sha256'];rows.append({'stage':stage,'actual_student_input_NPY_sha256':h,'shape':list(x.shape)})
    out=OUT/'incremental_model_checks';out.mkdir(exist_ok=True);result={'passed':True,'case':args.case,'completed_model_cases_checked':1,'exact_actual_student_inputs':24,'all_final_token_losses_predictions_exact':True,'production_execution_sha256':sha(OUT/'execution.json'),'run_record_sha256':sha(folder/'run.json'),'source_sha256':sha(__file__),'rows':rows,'scope':'Read-only original-model check of one completed case while production continues. No native replay, full-cohort gate, method selection, threshold change or task-quality conclusion.'};atomic_json(out/f'{args.case}.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
