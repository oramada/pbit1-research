"""Independently verify source weights, finite controls, ranges and unused tokens."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub.constants import HF_HUB_CACHE
from .transformer_block_gpu import sha,save_json
OUT=Path('results/thrml_spw/all_blocks_clean')

def main():
    p=json.loads((OUT/'protocol.json').read_text());torch.set_num_threads(8);snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}";model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval();total=0
    for stage,c in enumerate(p['control_artifacts']):
        assert c['stage']==stage and sha(c['path'])==c['sha256'];module=model.get_submodule(c['module']);w=module.weight.detach().numpy();a=np.load(c['path']);assert c['shape']==list(w.shape) and c['weights']==w.size
        lo=w.reshape(-1,128,w.shape[1]).min(1);hi=w.reshape(-1,128,w.shape[1]).max(1);span=hi-lo;np.testing.assert_array_equal(a['lower'],lo);np.testing.assert_array_equal(a['span'],span);np.testing.assert_array_equal(a['bias'],module.bias.detach().numpy())
        lower=np.repeat(lo,128,0);radius=np.repeat(span,128,0);prob=np.zeros_like(w);np.divide(w-lower,radius,out=prob,where=radius!=0);eps=1/(1+np.exp(8));prob=np.clip(prob.astype(float),eps,1-eps);field=((np.log(prob)-np.log1p(-prob))/2).astype(np.float32);codes=np.rint((np.clip(field.astype(float),-4,4)+4)*255/8).astype(np.uint8);np.testing.assert_array_equal(a['codes'],codes);total+=w.size
    assert total==42467328==p['converted_weights'];assert sum(v.numel() for v in model.parameters())==p['model_unique_parameters'];assert len(p['targets_in_forward_order'])==24
    ar=np.load(OUT/'tokens.npz');starts=ar['token_offsets'];assert len(starts)==32 and sha(OUT/'tokens.npz')==p['tokens_sha256'];assert np.min(np.diff(starts))>=65
    for r in p['excluded_archives']:
        assert sha(r['path'])==r['sha256'];old=np.load(r['path'])[r['key']];assert not np.any((starts[:,None]<old[None]+65)&(old[None]<starts[:,None]+65))
    old=np.array(p['legacy_excluded_offsets']);assert not np.any((starts[:,None]<old[None]+65)&(old[None]<starts[:,None]+65))
    texts=pq.read_table(p['source_parquet_path'],columns=['text']).column('text').to_pylist();tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True);ids=np.asarray(tok('\n'.join(t for t in texts if t and t.strip()),add_special_tokens=False)['input_ids'],np.int64);np.testing.assert_array_equal(ar['tokens'],np.stack([ids[i:i+65] for i in starts]))
    for name,h in p['source_sha256'].items():assert sha(name)==h
    r={'passed':True,'matrices_checked':24,'weight_controls_checked':total,'fresh_nonoverlapping_windows':32,'all_source_weights_ranges_codes_biases_exact':True,'source_sha256':sha(__file__),'no_model_scores_or_native_observations':True};save_json(OUT/'preparation_audit.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
