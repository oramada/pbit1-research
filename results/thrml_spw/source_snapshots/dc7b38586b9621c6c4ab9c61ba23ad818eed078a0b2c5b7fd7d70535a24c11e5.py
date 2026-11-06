"""Independent frozen quarter-model confirmation audit, including every native stage."""
import json,csv
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha,quantize as finite_properties
from .verify_transformer_quarter import affine

OUT=Path('results/thrml_spw/quarter_confirmation')

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def check(nll,pred,tokens,row):
    ce=float(np.mean(nll.astype(np.float64)));acc=float(np.mean(pred==tokens[:,1:]))
    np.testing.assert_allclose(ce,float(row['cross_entropy']),atol=1e-12);np.testing.assert_allclose(acc,float(row['token_accuracy']),atol=1e-12)
    np.testing.assert_allclose(np.exp(ce),float(row['perplexity']),rtol=1e-12)

def main():
    out=OUT;p=json.loads((out/'protocol.json').read_text());final=json.loads((out/'manifest.json').read_text())
    assert p==final['config'] and final['no_confirmation_tuning'] and not p['confirmation_used_for_tuning']
    frozen=Path('results/thrml_spw/quarter_confirmation_preparation/frozen_evaluation.json');assert sha(frozen)==p['frozen_evaluation_sha256']
    frozen_p=json.loads(frozen.read_text())
    for k,v in frozen_p.items():assert p[k]==v
    assert sha('results/thrml_spw/transformer_quarter/selection.json')==p['selection_sha256']
    assert sha('results/thrml_spw/transformer_quarter/protocol.json')==p['development_protocol_sha256']
    assert sha('results/thrml_spw/quarter_confirmation_preparation/protocol.json')==p['reservation_protocol_sha256']
    assert sha('results/thrml_spw/quarter_confirmation_preparation/reservation_audit.json')==p['reservation_audit_sha256']
    assert sha('research/thrml_spw/quarter_confirmation.py')==p['driver_source_sha256']
    assert sha('research/thrml_spw/sequential_transformer.py')==p['runner_source_sha256']
    assert p['selected_samples']==512 and p['field_bits']==8 and p['field_bound']==4. and p['confirmation_domain']==990730
    expected=[f'transformer.h.{b}.{s}' for b in range(3) for s in ['attn.c_attn','attn.c_proj','mlp.c_fc','mlp.c_proj']]
    assert p['targets_in_forward_order']==expected;assert sha(p['input_data_path'])==p['input_export_sha256']
    tokens=np.load(p['input_data_path'])['tokens'];assert tokens.shape==(128,65)
    weights=np.load(out/'weights.npz');assert sha(out/'weights.npz')==p['weights_sha256']
    assert sum(weights[f'weight_{i}'].size for i in range(12))==p['converted_matrix_weights']==21233664
    props=[finite_properties(weights[f'weight_{i}'],8,4.) for i in range(12)]
    base=read(out/'baseline_metrics.csv');base_raw=np.load(out/'baseline_tokens.npz')
    assert {r['condition'] for r in base}=={'FP32','affine4_block128','affine8_block128','programmed_mean8'}
    for r in base:check(base_raw[r['condition']+'_nll'],base_raw[r['condition']+'_predictions'],tokens,r)
    rows=read(out/'native_metrics.csv');prefixes=read(out/'prefix_metrics.csv');assert len(rows)==final['run_count']==5
    assert {int(r['sampling_seed']) for r in rows}==set(range(5));assert len(prefixes)==65
    stages_count=0
    for row in rows:
        seed=int(row['sampling_seed']);folder=out/'runs'/f'confirmation_seed{seed}';rec=json.loads((folder/'run.json').read_text());raw=np.load(folder/'prefix_tokens.npz')
        assert rec['status']=='complete' and rec['split']=='confirmation' and rec['samples']==512 and len(rec['stages'])==12
        assert rec['exact_prefix_input_checks']==int(row['exact_prefix_input_checks'])==128*78
        check(raw['prefix12_nll'],raw['prefix12_predictions'],tokens,row);total=0
        for i,s in enumerate(rec['stages']):
            meta=json.loads(Path(s['metadata']).read_text());sig=meta['signature'];ctrl=meta['controls'];qw,var,free,clipped=props[i]
            assert meta['status']=='complete' and sig['stage']==i and sig['seed']==seed and sig['samples']==512 and sig['domain']==990730
            assert sig['field_bits']==8 and sig['field_bound']==4. and sig['contexts_per_batch']==8
            assert sig['weights_sha256']==p['weights_sha256'] and sig['jax_backend']=='gpu' and sig['matmul_precision']=='highest'
            for path,digest in sig['source_sha256'].items():assert sha(path)==digest
            for key,checksum in [('inputs','inputs_sha256'),('reference','reference_sha256')]:assert sha(s[key])==sig[checksum]
            assert sha(s['output'])==meta['output_sha256'];assert s['prior_native_output_sha256']==[sha(folder/f'stage{k}_native_output.npy') for k in range(i)]
            x,ref,y=[np.load(s[k],mmap_mode='r') for k in ['inputs','reference','output']]
            conventional=x@weights[f'weight_{i}']+weights[f'bias_{i}'];assert np.max(np.abs(conventional-ref))/(1+np.max(np.abs(ref)))<2e-5
            mean=x.astype(np.float64)@qw+weights[f'bias_{i}'];bias=float(np.mean((mean-ref)**2));pred=float(np.mean(np.mean(x.astype(np.float64)**2,axis=(0,1))@var)/512)
            mse=float(np.mean((y.astype(np.float64)-ref)**2));centered=float(np.mean((y.astype(np.float64)-mean)**2))
            for value,key in [(bias,'conditional_control_bias_mse'),(pred,'predicted_sampling_mse'),(bias+pred,'predicted_total_mse'),(mse,'conditional_linear_mse'),(centered,'sampling_mse_vs_programmed_expectation')]:np.testing.assert_allclose(value,meta[key],rtol=1e-9,atol=1e-14)
            n=weights[f'weight_{i}'].size;assert ctrl['free_pbits']==free and ctrl['range_clipped_fields']==clipped
            assert ctrl['metadata_bits']==n//128*64 and ctrl['hypothetical_packed_controller_bits']==free*8
            assert meta['bank_batches']==16 and meta['free_node_updates']==16*512*free and meta['logical_weight_readouts']==16*512*n
            assert meta['binary_product_terms']==128*65*512*n;total+=meta['free_node_updates'];stages_count+=1
        assert total==int(row['free_node_updates'])==rec['free_node_updates']
        for prefix in range(13):
            reported=next(r for r in prefixes if int(r['sampling_seed'])==seed and int(r['converted_prefix'])==prefix)
            check(raw[f'prefix{prefix}_nll'],raw[f'prefix{prefix}_predictions'],tokens,reported)
    summary=final['summary'];ce=np.array([float(r['cross_entropy']) for r in rows]);margin=1.96*ce.std(ddof=1)/np.sqrt(5);fp=next(r for r in base if r['condition']=='FP32');delta=ce.mean()-float(fp['cross_entropy']);acc=np.mean([float(r['token_accuracy']) for r in rows])
    for value,key in [(ce.mean(),'cross_entropy_mean'),(margin,'cross_entropy_ci95_margin'),(np.exp(delta),'relative_perplexity'),(np.exp(delta-margin),'relative_perplexity_lower'),(np.exp(delta+margin),'relative_perplexity_upper'),(acc,'token_accuracy_mean')]:np.testing.assert_allclose(value,summary[key],rtol=1e-12)
    assert summary['mean_quality_tolerance_passed']==bool(delta<=np.log(1.05) and float(fp['token_accuracy'])-acc<=.01)
    hook=json.loads((out/'independent_hook_audit.json').read_text());assert hook['passed'] and hook['completed_runs_audited']==5
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoModelForCausalLM
    from .sequential_transformer import SequentialRunner
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}"
    runner=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),expected)
    for condition in ['FP32','affine4_block128','affine8_block128','programmed_mean8']:
        try:
            if condition!='FP32':
                with torch.no_grad():
                    for i,(module,w) in enumerate(zip(runner.modules,runner.weights)):
                        q=props[i][0].astype(np.float32) if condition=='programmed_mean8' else affine(w,4 if condition=='affine4_block128' else 8)
                        module.weight.copy_(torch.from_numpy(q))
            result=runner.evaluate(tokens);np.testing.assert_array_equal(result['nll'],base_raw[condition+'_nll']);np.testing.assert_array_equal(result['predictions'],base_raw[condition+'_predictions'])
        finally:runner.restore()
    report={'passed':True,'native_runs':5,'operator_checkpoints':stages_count,'independent_native_hook_replays':5,'independent_deterministic_replays':4,
        'target_tokens':8192,'converted_weights':21233664,'source_sha256':sha(__file__),
        'scope':'Frozen quarter-model confirmation, raw causal twelve-stage inputs/outputs, finite fields, error decomposition, counts and token losses; separate actual model replay. Not full-model or hardware proof.'}
    (out/'evidence_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
