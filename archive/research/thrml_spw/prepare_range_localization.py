"""Freeze validation-only ranges, variance diagnostics and native full-matrix probes."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub.constants import HF_HUB_CACHE
from .localized_controls import build
from .transformer_block_gpu import sha,save_json
OUT=Path('results/thrml_spw/range_localization')
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Prepared immutable study exists')
    assert shutil.disk_usage(OUT).free>600000000
    old=json.loads(Path('results/thrml_spw/sample_allocation/protocol.json').read_text());assert json.loads(Path('results/thrml_spw/sample_allocation/evidence_audit.json').read_text())['passed'];tokens_path=Path('results/thrml_spw/sample_allocation/tokens.npz');assert sha(tokens_path)==old['tokens_sha256'];data=np.load(tokens_path);np.savez_compressed(OUT/'validation_tokens.npz',tokens=data['validation_tokens'],token_offsets=data['validation_token_offsets']);torch.set_num_threads(8);model=AutoModelForCausalLM.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{old['revision']}",local_files_only=True,attn_implementation='eager').float().eval();artifacts={}
    for group in [16,32,64,128]:
        entries=[];folder=OUT/f'controls_g{group}'
        if group!=128:folder.mkdir(exist_ok=True)
        for stage,name in enumerate(old['targets_in_forward_order']):
            module=model.get_submodule(name);w=module.weight.detach().numpy();a=build(w,module.bias.detach().numpy(),group)
            if group==128:
                entry=old['control_artifacts'][stage];assert sha(entry['path'])==entry['sha256'];existing=np.load(entry['path'])
                for key in existing.files:np.testing.assert_array_equal(a[key],existing[key])
                path=Path(entry['path'])
            else:path=folder/f'stage{stage}.npz';np.savez(path,**a)
            entries.append({'stage':stage,'module':name,'group_size':group,'path':str(path),'sha256':sha(path),'shape':list(w.shape),'weights':int(w.size),'parameter_payload_bytes':sum(a[k].nbytes for k in ['codes','lower','span','bias']),'group_flag_bytes':2 if group!=128 else 0})
        artifacts[str(group)]=entries
    p={'schema':'spw-nested-local-ranges-validation-only-v1','revision':old['revision'],'dataset_revision':old['dataset_revision'],'source_validation_archive':str(tokens_path),'source_validation_sha256':sha(tokens_path),'tokens_sha256':sha(OUT/'validation_tokens.npz'),'windows':32,'sequence_length':65,'targets':2048,'validation_reused_for_development':True,'targets_in_forward_order':old['targets_in_forward_order'],'control_artifacts':artifacts,'groups':[16,32,64,128],'converted_weights':42467328,'model_unique_parameters':81912576,'fields':'8bit [-4,4], one free node per normalized signed weight, minmax lower/span grouped along input rows; no exact endpoint clamps',
        'anchor_positions':[16,48],'anchor_rows':64,'native_conditions':[{'name':'g128_T1024','group':128,'samples':1024},{'name':'g128_T512','group':128,'samples':512},{'name':'g16_T512','group':16,'samples':512}],'sampling_seeds':list(range(5)),'key_root':1200730,
        'theory':'For exact ideal p=(w-l)/(u-l), one-draw weight variance=(w-l)*(u-w). Nested minmax intervals imply coordinatewise variance16<=variance32<=variance64<=variance128. Finite controller clipping/rounding is analyzed separately and is not covered by ideal monotonicity. Operator conditionalvariance mean(X^2@v)/T; total targetMSE adds squared programmedmean bias.',
        'experiment':'Only existing validation text is used, never a fresh test quality claim. Four full analytic meanweight neural models and all42m weight-variance coordinates. Native probes use every24 full matrices on64 actual FP32-teacher activation rows (positions16,48 from every32 validation contexts), one weight bank shared acrossrows, five seeds. Primary operatorMSE contrast g16T512 versusg128T1024; secondary matchedT g16T512 versusg128T512. Conditions fixed before any new scoring/sampling. This does not execute the full stochastic neural network.',
        'gates':'Require exact ideal monotonicity at every coordinate; verify original source weights/codes/ranges; independent actual native output-sum and integerfrequency replay, dynamic fields/defaultF32, exact F32 outputs plus derived IEEE envelope. Empirical perlayer/condition mean targetMSE must be within20% of independent var/T+bias^2 prediction (practical simulator check, not formal coverage). Five paired MC seeds, all360=24*3*5 probes retained. No rescue or selection fromnewtest.',
        'resources':'One physical logical node per converted weight in every condition. Finer groups add explicitly counted F32metadata; compare samples/readouts and parameterpayload separately. Native probe Y arrays retained; inputanchors deterministic, independently regenerated and SHA matched. Full native endtoend heldout confirmation remains required and is deferred until persistent storage is available. No smaller probe is presented as full neural quality proof.',
        'scope':'Development-stage variance/range/operator evidence on one pretrained checkpoint and consumed validation32. Ideal static independent THRML Gibbs pbits, allsixblock learned matrices51.84% params; embeddings/head/norm/bias deterministic. No full-model stochastic task-quality, physical energy/latency or Extropic architecture-superiority claim.',
        'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['prepare_range_localization.py','localized_controls.py']},'new_model_scores_computed':False,'new_native_samples_collected':False}
    save_json(OUT/'protocol.json',p);print(json.dumps({'prepared':True,'groups':[16,32,64,128],'native_probes':360,'new_controller_bytes':sum(c['parameter_payload_bytes']+c['group_flag_bytes'] for g in ['16','32','64'] for c in artifacts[g]),'storage_free_bytes':shutil.disk_usage(OUT).free,'full_native_neural_confirmation_pending_storage':True},indent=2))
if __name__=='__main__':main()
