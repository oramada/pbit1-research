"""Freeze all-six-block ideal finite-controller scale test before task scoring."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub.constants import HF_HUB_CACHE
from .sequential_transformer import projection_targets
from .calibrated_transformer_controls import export_controls
from .transformer_block_gpu import sha,save_json
OUT=Path('results/thrml_spw/all_blocks')

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Immutable preparation already exists')
    old=json.loads(Path('results/thrml_spw/probe_placement/protocol.json').read_text())
    assert json.loads(Path('results/thrml_spw/probe_placement/evidence_audit.json').read_text())['passed']
    exclusions=old['excluded_archives']+[{'path':'results/thrml_spw/probe_placement/tokens.npz','key':'token_offsets','sha256':sha('results/thrml_spw/probe_placement/tokens.npz')}]
    used=np.concatenate([np.load(r['path'])[r['key']] for r in exclusions]+[np.array(old['legacy_excluded_offsets'])])
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{old['revision']}"
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    parquet=Path(old['source_parquet_path']);assert sha(parquet)==old['source_parquet_sha256']
    text=pq.read_table(parquet,columns=['text']).column('text').to_pylist();ids=np.asarray(tok('\n'.join(t for t in text if t and t.strip()),add_special_tokens=False)['input_ids'],np.int64)
    eligible=np.arange(len(ids)//65)*65;eligible=eligible[~np.any((eligible[:,None]<used[None]+65)&(used[None]<eligible[:,None]+65),axis=1)]
    starts=np.sort(np.random.default_rng(2026116800).choice(eligible,32,replace=False));tokens=np.stack([ids[i:i+65] for i in starts]);np.savez_compressed(OUT/'tokens.npz',tokens=tokens,token_offsets=starts)
    torch.set_num_threads(8);model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval();targets=projection_targets(range(6));controls=[];(OUT/'controls').mkdir(exist_ok=True)
    for stage,name in enumerate(targets):
        module=model.get_submodule(name);w=module.weight.detach().numpy();path=OUT/'controls'/f'stage{stage}.npz';export_controls(path,w,module.bias.detach().numpy())
        a=np.load(path);assert w.shape[0]%128==0
        controls.append({'stage':stage,'module':name,'path':str(path),'sha256':sha(path),'shape':list(w.shape),'weights':int(w.size),'payload_bytes':sum(a[k].nbytes for k in a.files)})
        if stage<12:assert sha(path)==old['control_artifacts'][stage]['sha256']
    n=sum(r['weights'] for r in controls);total=sum(p.numel() for p in model.parameters());assert n==42467328 and total==81912576
    p={'schema':'spw-all-six-blocks-ideal-fixed-regime-v1','revision':old['revision'],'dataset_revision':old['dataset_revision'],'source_parquet_path':str(parquet),'source_parquet_sha256':sha(parquet),'tokens_sha256':sha(OUT/'tokens.npz'),'token_selection_seed':2026116800,'excluded_archives':exclusions,'legacy_excluded_offsets':old['legacy_excluded_offsets'],
       'windows':32,'sequence_length':65,'targets':2048,'targets_in_forward_order':targets,'control_artifacts':controls,'converted_weights':n,'model_unique_parameters':total,'converted_fraction':n/total,'sampling_seeds':list(range(5)),'native_samples':1024,'contexts_per_bank':8,'key_root':1170730,'bootstrap_seed':2026116801,'bootstrap_draws':10000,
       'selection':'No new task-driven selection. Fixed established T1024, 8bit[-4,4], group128 and all24 projections. Retain every seed and all prefixes regardless of outcome. Primary contrast all24 minus first12 on same text and shared first12 native outputs. No rescue budget.',
       'execution':'One free native THRML node per normalized signed weight. Ideal unary fields decoded from finite8bit codes, no gain/offset error, no calibration. Actual student inputs at all24 projections, accumulation before original nonlinearities. Embeddings, final tied head, LayerNorm and biases deterministic.',
       'decision':'Descriptive frozen point PPLratio<=1.05 and accuracydelta>=-.01 vsFP32 for prefixes12/24. Strict crossed seed/bank97.5%CE upper<=log1.05 and2.5%accuracy lower>=-.01. Five seeds crossed with four8contextbanks; sensitivity intervals, not guaranteedcoverage. This is a fixed-regime scaling test, not a validation-selected full-model confirmation.',
       'evidence':'Preserve all native outputs, every prefix per-token loss/prediction, exactNPY SHA of every runtime input. Regenerate input arrays independently from pinned model/text and prior saved outputs for all-node public THRML replay; temporary input copies removed only after hash verification.',
       'scope':'51.84% unique parameter conversion in one pinned pretrained checkpoint, all six Transformer blocks. Not full model: embeddings/head remain deterministic. Ideal finite controllers and short65token contexts,32 fresh windows; no nonideal-device or physical advantage claim.',
       'source_sha256':{str(Path('research/thrml_spw')/name):sha(Path('research/thrml_spw')/name) for name in ['prepare_all_blocks.py','calibrated_transformer_controls.py','sequential_transformer.py']},'model_scores_computed':False,'native_samples_collected':False}
    save_json(OUT/'protocol.json',p);print(json.dumps({'prepared':True,'converted_weights':n,'converted_fraction':n/total,'new_windows':32,'controls_payload_bytes':sum(c['payload_bytes'] for c in controls),'no_model_scores':True},indent=2))

if __name__=='__main__':main()
