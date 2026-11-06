"""Reserve domain-qualified unused validation/test text and freeze allocation rule."""
import json,shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from .transformer_block_gpu import sha,save_json
OUT=Path('results/thrml_spw/sample_allocation')
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Immutable study already prepared')
    assert shutil.disk_usage(OUT).free>4500000000
    old=json.loads(Path('results/thrml_spw/all_blocks_clean/protocol.json').read_text());assert json.loads(Path('results/thrml_spw/all_blocks_clean/evidence_audit.json').read_text())['passed'];history=[]
    for directory in ['data/thrml_spw','results/thrml_spw']:
        for path in sorted(Path(directory).rglob('*.npz')):
            if path.name not in ['tokens.npz','export.npz'] or path.parent==OUT:continue
            with np.load(path) as a:
                for key in a.files:
                    if 'offset' in key and a[key].ndim==1 and a[key].size and a[key].dtype.kind in 'iu':history.append({'path':str(path),'key':key,'domain':'validation' if 'validation' in key else 'test','sha256':sha(path),'windows':len(a[key])})
    tok=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{old['revision']}",local_files_only=True);sources={};saved={}
    for domain,seed in [('validation',2026117000),('test',2026117001)]:
        path=Path(hf_hub_download('Salesforce/wikitext',f'wikitext-2-raw-v1/{domain}-00000-of-00001.parquet',repo_type='dataset',revision=old['dataset_revision'],local_files_only=True));sources[domain]={'path':str(path),'sha256':sha(path)};texts=pq.read_table(path,columns=['text']).column('text').to_pylist();ids=np.asarray(tok('\n'.join(s for s in texts if s and s.strip()),add_special_tokens=False)['input_ids'],np.int64);used=[np.load(r['path'])[r['key']] for r in history if r['domain']==domain]
        if domain=='test':used.append(np.array(old['legacy_excluded_offsets']))
        used=np.concatenate(used);eligible=np.arange(len(ids)//65)*65;eligible=eligible[~np.any((eligible[:,None]<used[None]+65)&(used[None]<eligible[:,None]+65),axis=1)];starts=np.sort(np.random.default_rng(seed).choice(eligible,32,replace=False));saved[domain+'_tokens']=np.stack([ids[i:i+65] for i in starts]);saved[domain+'_token_offsets']=starts
    np.savez_compressed(OUT/'tokens.npz',**saved)
    for c in old['control_artifacts']:assert sha(c['path'])==c['sha256']
    p={'schema':'spw-validation-selected-equal-update-sample-allocation-v1','revision':old['revision'],'dataset_revision':old['dataset_revision'],'source_parquets':sources,'token_history':history,'legacy_excluded_test_offsets':old['legacy_excluded_offsets'],'tokens_sha256':sha(OUT/'tokens.npz'),'windows_each_split':32,'targets_each_split':2048,'contexts_per_bank':8,'sampling_seeds':list(range(5)),'targets_in_forward_order':old['targets_in_forward_order'],'control_artifacts':old['control_artifacts'],'converted_weights':42467328,'model_unique_parameters':81912576,
       'validation_samples':256,'validation_key_root':1180730,'test_key_root':1190730,'sample_levels':[256,512,1024,2048,4096],'weight_unit':589824,'sample_unit':256,'exact_budget_units':288,'uniform_samples':1024,'loss_floor':1e-4,'bootstrap_seed':2026117002,'bootstrap_draws':10000,
       'selection':'Compute single-stochastic-projection CE atT256, five seeds, on validationonly; all other converted matrices set to their exact F64-analytic programmed means castF32. Coefficient S_l=max(meanCE_l-meanCE_programmed_mean_all24,1e-4). Minimize sum_l S_l/(T_l/256) via exact DP with sum_l(N_l/589824)*(T_l/256)=288. Allowed sample levels fixed; exact cost tie breaks lexicographically smallest schedule. No hyperparameter search or test scoring before frozen allocation audit. Surrogate assumes separable inverseT loss; interactions/noisy estimates can defeat it.',
       'test':'Evaluate bothuniformT1024 and frozenallocation all24 projections on32 unused test windows, five seeds. Whole current stochastic student inputs at every projection. Key root1190730 foldedseed,stage,bank commonacrossmethods; differentT has different internal PRNGsplit schedule and is not a shared sampleprefix. Retain all10 cases; no rescue or choosing a winner fromtest.',
       'quality':'Primary paired CE allocated-minus-uniform. Separate fixedpoint PPLratio<=1.05 andaccuracydelta>=-.01 vsFP32; stricter crossed5seed/four8contextbank10000resamples CE97.5%upper<=log1.05 andaccuracy2.5%lower>=-.01. Descriptive sensitivity, not guaranteedcoverage. No tolerance relaxed.',
       'audit':'Every native operator independently public-factor replayed with dynamic fields, defaultF32 and exactF32 output/count observer; require byte-identical production outputs. Compute F64 frequency reconstruction and an IEEE F32 forward-error envelope for every output. Record2e-5 discrepancy flag descriptively, do not treat it as a sampled-state mismatch if exact publicF32 replay and derived envelope pass. Audit all NNinputs/outputs and scores independently. Bounds/requirements fixed before new data.',
       'resources':'All42,467,328 normalized signed weights useone free node; finite8bit[-4,4] fields/group128 ranges unchanged. Test methods matchsumN_lT_l exactly, not latency or energy. Report sumT_l/maxT_l, metadata48byte schedule, validation selection and verification costs separately. Preserve all nativeY, rawtokenmetrics and inputNPY SHA; temporaryinputs regenerated independently and removed only after verified.',
       'scope':'One DistilGPT2 checkpoint, allsixblock matrices51.84% params, embeddings/head/norm/bias deterministic. Ideal static independent devices. Short65tokencontexts. Allocation mayfail; no full-model/nonideal/physical superiority claim.',
       'source_sha256':{str(Path('research/thrml_spw')/n):sha(Path('research/thrml_spw')/n) for n in ['prepare_sample_allocation.py','sample_allocation_math.py']},'model_scores_computed':False,'native_samples_collected':False}
    save_json(OUT/'protocol.json',p);print(json.dumps({'prepared':True,'fresh_validation_windows':32,'fresh_test_windows':32,'offset_namespaces_qualified':True,'equal_updates_per_bank':42467328*1024,'model_scores_computed':False},indent=2))
if __name__=='__main__':main()
