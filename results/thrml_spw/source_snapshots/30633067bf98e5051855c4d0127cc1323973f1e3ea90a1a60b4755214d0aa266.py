"""CPU-only semantic check for scoring only required LFM3 prediction positions."""
import json,sys
from .prepare_v1 import OUT,sha

def main():
 import torch
 from transformers import AutoConfig,AutoModelForCausalLM
 torch.set_num_threads(2);torch.manual_seed(2026112101)
 cfg=AutoConfig.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',local_files_only=True,trust_remote_code=True)
 cfg.hidden_size=64;cfg.intermediate_size=128;cfg.num_hidden_layers=3;cfg.num_attention_heads=4;cfg.num_key_value_heads=2;cfg.head_dim=16
 cfg.layer_types=['conv','full_attention','sliding_attention'];cfg.vocab_size=256;cfg.max_position_embeddings=1024;cfg.sliding_window=32;cfg.bos_token_id=0;cfg.eos_token_id=1;cfg.pad_token_id=2
 model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=True,attn_implementation='eager').to(torch.bfloat16).eval()
 sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv=lambda *a:False
 checks=[]
 with torch.inference_mode():
  for length,first in [(17,1),(65,33),(129,65),(257,129)]:
   ids=(torch.arange(length)[None]*37)%256;last=length;positions=torch.arange(first-1,last-1)
   full=model(input_ids=ids,use_cache=False).logits
   subset=model(input_ids=ids,use_cache=False,logits_to_keep=positions).logits
   assert torch.equal(subset,full[:,positions])
   ref=torch.nn.functional.cross_entropy(full[0,positions].float(),ids[0,first:last],reduction='none')
   got=torch.cat([torch.nn.functional.cross_entropy(subset[0,start:start+64].float(),ids[0,first+start:min(first+start+64,last)],reduction='none')for start in range(0,last-first,64)])
   assert torch.equal(ref,got)
   checks.append({'length':length,'first_target':first,'targets':last-first,'logits_bitwise':True,'chunked_CE_bitwise':True})
 assert not torch.cuda.is_initialized()
 output={'status':'all4_selected_logit_and_chunked_score_fixtures_passed','source_sha256':sha(__file__),'checks':checks,'scope':'Tiny synthetic3layerCPU BF16 semantic check. FullcheckpointCUDA replay still required before accepting finalbenchmark.'}
 (OUT/'lfm3_selected_logits_verification_v1.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))
if __name__=='__main__':main()
