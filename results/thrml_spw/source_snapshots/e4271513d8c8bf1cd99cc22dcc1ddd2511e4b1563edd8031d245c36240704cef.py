"""Independent pinned checkpoint, positional codes and original-model anchor audit."""
import json,io
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .positional_operator_common import OUT,ROOT,sha,digest,guard

def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);torch.set_num_threads(8);torch.use_deterministic_algorithms(True);model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}",local_files_only=True,attn_implementation='eager').float().eval();modules=[model.get_submodule(c['module']) for c in p['matrices']];capture=[[] for _ in modules];hooks=[]
    for i,m in enumerate(modules):
        def before(module,args,index=i):capture[index].append(args[0].detach().numpy()[0,[16,48]].copy())
        hooks.append(m.register_forward_pre_hook(before))
    tokens=np.load(ROOT/'validation_tokens.npz')['tokens']
    try:
        with torch.inference_mode():
            for row in tokens:model(input_ids=torch.from_numpy(row[None]),use_cache=False)
    finally:
        for h in hooks:h.remove()
    checked=0
    for c,mod,values in zip(p['matrices'],modules,capture):
        w=mod.weight.detach().numpy();assert digest(w)==c['original_weight_sha256'];v=w.reshape(-1,16,w.shape[1]);lower=v.min(1);span=v.max(1)-lower;lo=np.repeat(lower,16,0);sp=np.repeat(span,16,0);x=np.concatenate(values);anchor=c['anchor'];np.testing.assert_array_equal(x,np.load(anchor['inputs']));buf=io.BytesIO();np.save(buf,x);import hashlib
        assert hashlib.sha256(buf.getvalue()).hexdigest()==anchor['inputs_sha256']==sha(anchor['inputs']);np.testing.assert_array_equal(x.astype(float)@w.astype(float)+mod.bias.detach().numpy(),np.load(anchor['target']));assert sha(anchor['target'])==anchor['target_sha256']
        normalized=np.zeros_like(w);np.divide(w-lo,sp,out=normalized,where=sp!=0);word=np.clip(np.rint(normalized*15),0,15).astype(np.uint8)
        for which in ['single_controls','positional_controls']:
            e=c[which];assert sha(e['path'])==e['sha256'];a=np.load(e['path']);np.testing.assert_array_equal(a['lower'],lower);np.testing.assert_array_equal(a['span'],span);np.testing.assert_array_equal(a['bias'],mod.bias.detach().numpy())
            if which=='positional_controls':np.testing.assert_array_equal(a['codes'],word)
            else:
                eps=1/(1+np.exp(8));q=np.clip(normalized.astype(float),eps,1-eps);f=(np.log(q/(1-q))/2).astype(np.float32);codes=np.rint((np.clip(f.astype(float),-4,4)+4)*255/8).astype(np.uint8);np.testing.assert_array_equal(codes,a['codes'])
        checked+=w.size;print(f'Independent original-model anchors/ranges/positional controls stage{c["stage"]}: exact',flush=True)
    assert checked==42467328;guard(p);r={'passed':True,'original_model_anchor_inputs_exact':24,'original_F64_targets_exact':24,'source_weight_coordinates_and_both_control_formats_checked':checked,'producer_protocol_sha256':sha(OUT/'protocol.json'),'source_sha256':sha(__file__),'prior_anchor_audit_sha256':p['prior_anchor_audit_sha256'],'scope':'Independent checkpoint and original-model hooks verify every input, target, range, code and bias. Consumed validation only; no stochastic neural quality or hardware result.'};(OUT/'control_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
