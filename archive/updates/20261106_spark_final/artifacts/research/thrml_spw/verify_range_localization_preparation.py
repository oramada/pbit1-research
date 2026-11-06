"""Independent full-weight finite-control reconstruction and validation source audit."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .transformer_block_gpu import sha,save_json
OUT=Path('results/thrml_spw/range_localization')
def main():
    p=json.loads((OUT/'protocol.json').read_text());source=np.load(p['source_validation_archive']);ar=np.load(OUT/'validation_tokens.npz');assert sha(p['source_validation_archive'])==p['source_validation_sha256'] and sha(OUT/'validation_tokens.npz')==p['tokens_sha256'];np.testing.assert_array_equal(ar['tokens'],source['validation_tokens']);np.testing.assert_array_equal(ar['token_offsets'],source['validation_token_offsets']);assert ar['tokens'].shape==(32,65);torch.set_num_threads(8);model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();total=0
    for stage,name in enumerate(p['targets_in_forward_order']):
        module=model.get_submodule(name);w=module.weight.detach().numpy();previous=None
        for g in [128,64,32,16]:
            c=p['control_artifacts'][str(g)][stage];assert sha(c['path'])==c['sha256'];a=np.load(c['path']);lower=np.stack([w[i:i+g].min(0) for i in range(0,len(w),g)]);upper=np.stack([w[i:i+g].max(0) for i in range(0,len(w),g)]);span=upper-lower;np.testing.assert_array_equal(a['lower'],lower);np.testing.assert_array_equal(a['span'],span);np.testing.assert_array_equal(a['bias'],module.bias.detach().numpy());lo=np.repeat(lower,g,0);sp=np.repeat(span,g,0);prob=np.zeros_like(w);np.divide(w-lo,sp,out=prob,where=sp!=0);eps=1/(1+np.exp(8));prob=np.clip(prob.astype(float),eps,1-eps);field=((np.log(prob)-np.log1p(-prob))/2).astype(np.float32);codes=np.rint((np.clip(field.astype(float),-4,4)+4)*255/8).astype(np.uint8);np.testing.assert_array_equal(codes,a['codes']);ideal=(w.astype(float)-lo.astype(float))*(np.repeat(upper,g,0).astype(float)-w.astype(float));assert np.all(ideal>=0)
            if previous is not None:assert np.all(ideal<=previous)
            previous=ideal;assert c['parameter_payload_bytes']==sum(a[k].nbytes for k in ['codes','lower','span','bias']);total+=w.size
    assert total==4*42467328 and len(p['native_conditions'])*len(p['sampling_seeds'])*24==360
    for name,h in p['source_sha256'].items():assert sha(name)==h
    r={'passed':True,'weight_group_coordinates_checked':total,'native_probe_count_predeclared':360,'all_ranges_codes_biases_exact':True,'all_nested_ideal_variance_coordinates_monotone':True,'validation_only_consumed_for_development':True,'source_sha256':sha(__file__),'new_model_scores_or_native_samples':False};save_json(OUT/'preparation_audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
