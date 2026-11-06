"""Prospective higher-budget study after the frozen T512 confirmation miss.

Only reserve tokens and a decision rule here; compute no model scores. The failed
confirmation remains immutable and is explicitly treated as consumed evidence.
"""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoTokenizer
from .transformer_export import MODEL_REVISION,DATASET_REVISION
from .verify_transformer_control import sha


def main():
    root=Path('results/thrml_spw');out=Path('data/thrml_spw/quarter_conservative');out.mkdir(exist_ok=True)
    evidence=root/'quarter_conservative_preparation';evidence.mkdir(exist_ok=True)
    if (out/'tokens.npz').exists():raise RuntimeError('New study tokens already reserved; reuse immutable artifacts')
    prior=root/'quarter_confirmation/manifest.json';outcome=json.loads(prior.read_text());assert not outcome['summary']['mean_quality_tolerance_passed']
    assert json.loads((root/'quarter_confirmation/evidence_audit.json').read_text())['passed']
    old=np.load('data/thrml_spw/transformer_projection/export.npz')
    prior_first=np.load('data/thrml_spw/transformer_confirmation/tokens.npz');prior_quarter=np.load('data/thrml_spw/quarter_confirmation/tokens.npz')
    tokenizer=AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE)/f'models--distilbert--distilgpt2/snapshots/{MODEL_REVISION}',local_files_only=True)
    saved={};sources={};exclusions={};legacy=None
    for split,seed in [('validation',20261109),('test',20261110)]:
        path=Path(hf_hub_download('Salesforce/wikitext',f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet',repo_type='dataset',revision=DATASET_REVISION,local_files_only=True));sources[split]=sha(path)
        texts=pq.read_table(path,columns=['text']).column('text').to_pylist();ids=np.asarray(tokenizer('\n'.join(s for s in texts if s and s.strip()),add_special_tokens=False)['input_ids'],dtype=np.int64)
        used=old['validation_token_offsets'] if split=='validation' else np.concatenate([old['test_token_offsets'],prior_first['token_offsets'],prior_quarter['token_offsets']])
        offsets=np.setdiff1d(np.arange(len(ids)//65)*65,used)
        if split=='test':
            legacy=np.sort(np.random.default_rng(17).choice(len(ids)-64,size=4,replace=False))
            offsets=offsets[np.array([not any(start<prior+65 and prior<start+65 for prior in legacy) for start in offsets])]
        chosen=np.sort(np.random.default_rng(seed).choice(offsets,128,replace=False));tokens=np.stack([ids[start:start+65] for start in chosen])
        assert not set(chosen)&set(used)
        if split=='test':assert not any(start<prior+65 and prior<start+65 for start in chosen for prior in legacy)
        saved[split+'_tokens']=tokens;saved[split+'_token_offsets']=chosen;exclusions[split]=len(used)
    np.savez_compressed(out/'tokens.npz',**saved)
    protocol={'model_revision':MODEL_REVISION,'dataset_revision':DATASET_REVISION,'source_parquet_sha256':sources,'tokens_sha256':sha(out/'tokens.npz'),
        'motivating_failed_confirmation_sha256':sha(prior),'motivating_failed_confirmation_relative_perplexity':outcome['summary']['relative_perplexity'],
        'source_development_protocol_sha256':sha(root/'transformer_quarter/protocol.json'),
        'validation_windows':128,'test_windows':128,'targets_per_split':8192,'exclusions':exclusions,'legacy_test_starts_excluded':legacy.tolist(),
        'selection_seeds':{'validation':20261109,'test':20261110},'targets_in_forward_order':outcome['config']['targets_in_forward_order'],
        'converted_weights':21233664,'field_bits':8,'field_bound':4.,'input_block_size':128,'sampling_seeds':list(range(5)),
        'sample_search':[1024,2048],'contexts_per_bank':8,'validation_domain':1000730,'test_domain':1010730,
        'selection':'On new validation only, ascending T; require mean CE increase<=log(1.05), mean accuracy loss<=0.01, paired crossed seed/bank bootstrap97.5%-CE increase<=log(1.05), and bootstrap2.5%-accuracy difference>=-0.01. If none pass record failure; do not relax thresholds.',
        'selection_bootstrap':{'resamples':10000,'seed':20261111,'bank_contexts':8,'assumption':'Exchangeable bank sensitivity heuristic, not formal distribution-free coverage'},
        'confirmation':'Freeze first passing T; if none pass, largest T is explicitly failed diagnostic. Evaluate original deterministic controls and five native seeds on reserved test, without tuning. Report mean tolerance and paired sensitivity with bootstrap seed20261112.',
        'study_motivation':'The original point-mean calibration selected T512 narrowly but independent confirmation missed5%. This is a new, prospectively specified development study with larger calibration data and a conservative resampling rule, not a rescue of the consumed confirmation.',
        'no_model_scores_computed':True,'scope':'Same WikiText-2 corpus, new disjoint positions; first three blocks only, not full-model or physical-chip proof. Higher T doubles or quadruples prior stochastic cost and must be counted.',
        'source_sha256':sha(__file__)}
    (evidence/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({k:protocol[k] for k in ['validation_windows','test_windows','exclusions','sample_search','no_model_scores_computed','tokens_sha256']}))

if __name__=='__main__':main()
