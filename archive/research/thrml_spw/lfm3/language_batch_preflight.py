from pathlib import Path
import hashlib,json,sys
import numpy as np,torch
from transformers import AutoModelForCausalLM
root=Path('results/thrml_spw/lfm3_1p7b');out=root/'language_batch_preflight';out.mkdir(exist_ok=True)
torch.set_num_threads(4)
model=AutoModelForCausalLM.from_pretrained('data/thrml_spw/lfm3_1p7b/checkpoint',trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
sys.modules[type(model.model.layers[0].conv).__module__].can_use_lfm3_conv=lambda *args:False
ids=torch.from_numpy(np.load(root/'language_subset/tokens.npz')['tokens']).to('cuda')
with torch.inference_mode():y=model(input_ids=ids,use_cache=False).logits.float().cpu().numpy()
np.save(out/'logits.npy',y)
b=np.load(root/'language_controls/BF16_logits.npy');diff=y[:,:-1].astype(np.float64)-b
r={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'exact_batch_vs_individual':bool(np.array_equal(y[:,:-1],b)),'logit_MSE':float(np.mean(diff**2)),'maximum_absolute_difference':float(np.max(np.abs(diff))),'batch_shape':list(y.shape)}
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
