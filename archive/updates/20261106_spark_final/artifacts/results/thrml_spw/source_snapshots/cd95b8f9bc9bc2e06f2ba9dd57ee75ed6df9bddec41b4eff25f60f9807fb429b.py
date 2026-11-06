"""CPU/meta-only inventory for a bounded-memory LFM3 diagnostic path."""
from pathlib import Path
import json
from .prepare_v1 import OUT,sha

def main():
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    path=Path('data/thrml_spw/lfm3_1p7b/checkpoint');cfg=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=True)
    with torch.device('meta'):model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=True,attn_implementation='eager')
    assert all(p.is_meta for p in model.parameters())and not torch.cuda.is_initialized()
    canonical=[];seen={};aliases={};modules=[]
    for name,p in model.named_parameters(remove_duplicate=False):
        aliases[name]=seen.setdefault(id(p),name)
        if aliases[name]==name:canonical.append({'name':name,'shape':list(p.shape),'elements':p.numel(),'BF16_bytes':2*p.numel()})
    prior=json.loads(Path('results/thrml_spw/lfm3_1p7b/meta_inventory.json').read_text());assert sum(r['elements']for r in canonical)==prior['unique_parameters']==1728801536
    for name,m in model.named_modules():
        direct=list(m.parameters(recurse=False))
        if direct:modules.append({'name':name,'class':type(m).__name__,'direct_parameter_BF16_bytes':sum(2*p.numel()for p in direct)})
    result={'status':'meta_inventory_only_no_weights_loaded_or_model_forward','model':prior['checkpoint'],'revision':prior['revision'],'unique_parameters':1728801536,
        'BF16_weight_bytes':2*1728801536,'largest_leaf_BF16_bytes':max(r['direct_parameter_BF16_bytes']for r in modules),
        'canonical':canonical,'aliases':aliases,'parameter_modules':modules,
        'source_sha256':sha(__file__),'checkpoint_sources':{n:sha(path/n)for n in ['config.json','configuration_lfm3.py','modeling_lfm3.py','lfm3_conv.py']},
        'proposed_execution':'KeepBF16weightsonCPU; pre/posthookparameterizedleafmodulestotransferweightsforoneforwardonly; disablewideconvfastpathtokeepnormalleafcalls;use_cache=False,seq_idx=None. Mustvalidateagainstordinaryforwardbeforescientificuse.',
        'scope':'Memoryplanningonly. Metaallocatedarchitecture hasnoactualweightdata. Thisdoesnotproveoffloadcorrectness,2GiBpeak,neuralqualityoronepbitinference.'}
    root=OUT/'lfm3_memory_plan_v1';root.mkdir(exist_ok=False);(root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['model','unique_parameters','BF16_weight_bytes','largest_leaf_BF16_bytes','status']}))

if __name__=='__main__':main()
