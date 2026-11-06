"""Prepare weight-only positional controls and seal validation operator comparison."""
from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .positional_operator_common import OUT,ROOT,SOURCES,sha

def main():
    assert not (OUT/'protocol.json').exists();OUT.mkdir(parents=True,exist_ok=True);old=json.loads((ROOT/'protocol.json').read_text());anchors=json.loads((ROOT/'analytic_manifest.json').read_text())['anchors'];assert json.loads((ROOT/'analytic_evidence_audit.json').read_text())['passed'];torch.set_num_threads(8);model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{old['revision']}",local_files_only=True,attn_implementation='eager').float().eval();entries=[];folder=OUT/'controls_K4';folder.mkdir(exist_ok=True)
    for stage,name in enumerate(old['targets_in_forward_order']):
        entry=old['control_artifacts']['16'][stage];assert sha(entry['path'])==entry['sha256'];a=np.load(entry['path']);mod=model.get_submodule(name);w=mod.weight.detach().numpy();lo=np.repeat(a['lower'],16,0);span=np.repeat(a['span'],16,0);np.testing.assert_array_equal(a['lower'],w.reshape(-1,16,w.shape[1]).min(1));np.testing.assert_array_equal(a['bias'],mod.bias.detach().numpy());normalized=np.divide(w-lo,span,out=np.zeros_like(w),where=span!=0);codes=np.rint(np.clip(normalized,0,1)*15).astype(np.uint8);path=folder/f'stage{stage}.npz';np.savez(path,codes=codes,lower=a['lower'],span=a['span'],bias=a['bias']);anchor=anchors[stage]
        for key in ['inputs','target']:assert sha(anchor[key])==anchor[key+'_sha256']
        entries.append({'stage':stage,'module':name,'weights':int(w.size),'shape':list(w.shape),'original_weight_sha256':sha_bytes(w),'single_controls':entry,'positional_controls':{'path':str(path),'sha256':sha(path)},'anchor':anchor})
    p={'schema':'finite-positional-neural-operator-validation-v1','revision':old['revision'],'group':16,'field_bound':4,'one_node_control_bits':8,'positional_nodes':4,'positional_control_levels':[-4,4],'seeds':list(range(5)),'single_samples':[4,16,64,256],'positional_samples':[1,4,16,64],'key_root':1237520,'matrices':entries,'converted_weights':42467328,'anchor_rows':64,'native_operators':960,'conditions_fixed_before_sampling':True,'validation_only':True,'prior_anchor_audit_sha256':sha(ROOT/'analytic_evidence_audit.json'),'prior_numerical_gate_failed_and_unchanged':True,'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in SOURCES},'gates':'All960 outputs must replay byte-exact with independent public factors/sum-only observers under autotune0/defaultF32. Separate integer-count replay must satisfy derived IEEE forward error envelope. Analytical targetMSE=conditionalvariance/T+programmedmean bias^2 independently checked. Five-seed comparisons descriptive; no new heldout neural quality or physical systems gate.','scope':'All24 full matrices, actual FP32-teacher validation activations reused from independently audited anchors. Same local ranges for K1/K4. Each operator samples full weight realization then sums preactivations before any nonlinearity. This internal positional baseline is not an asserted Extropic architecture; not end-to-end stochastic student quality or hardware area/latency/energy.'}
    (OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'prepared':True,'protocol_sha256':sha(OUT/'protocol.json'),'native_operators':960,'new_neural_scores':0,'new_native_samples':0}))
def sha_bytes(w):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest()
if __name__=='__main__':main()
