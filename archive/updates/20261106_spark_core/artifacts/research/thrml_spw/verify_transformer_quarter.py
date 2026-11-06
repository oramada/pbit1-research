"""Independent raw evidence audit for the twelve-projection development study."""
import json,csv
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha,quantize as finite_properties

OUT=Path('results/thrml_spw/transformer_quarter')

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def check(nll,pred,tokens,row):
    ce=float(nll.astype(np.float64).mean());acc=float(np.mean(pred==tokens[:,1:]))
    np.testing.assert_allclose(ce,float(row['cross_entropy']),atol=1e-12);np.testing.assert_allclose(acc,float(row['token_accuracy']),atol=1e-12)
    np.testing.assert_allclose(np.exp(ce),float(row['perplexity']),rtol=1e-12)

def affine(w,bits):
    grouped=w.reshape(-1,128,w.shape[1]);lo=np.minimum(grouped.min(1,keepdims=True),0);hi=np.maximum(grouped.max(1,keepdims=True),0)
    levels=2**bits-1;scale=(hi-lo)/levels;zero=np.clip(np.rint(np.divide(-lo,scale,out=np.zeros_like(lo),where=scale!=0)),0,levels)
    code=np.clip(np.rint(np.divide(grouped,scale,out=np.zeros_like(grouped),where=scale!=0))+zero,0,levels)
    return ((code-zero)*scale).reshape(w.shape)

def main():
    out=OUT;p=json.loads((out/'protocol.json').read_text());final=json.loads((out/'manifest.json').read_text());selection=json.loads((out/'selection.json').read_text())
    assert final['config']==p and p['development_only'] and not p['completed_confirmation_data_used']
    assert p['driver_source_sha256']==sha('research/thrml_spw/transformer_quarter.py')
    assert p['runner_source_sha256']==sha('research/thrml_spw/sequential_transformer.py')
    expected_targets=[f'transformer.h.{b}.{s}' for b in range(3) for s in ['attn.c_attn','attn.c_proj','mlp.c_fc','mlp.c_proj']]
    assert p['targets_in_forward_order']==expected_targets;assert len(expected_targets)==12
    assert sha(p['input_data_path'])==p['input_export_sha256'];data=np.load(p['input_data_path'])
    w=np.load(out/'weights.npz');assert sha(out/'weights.npz')==final['weights_sha256'];assert sum(w[f'weight_{i}'].size for i in range(12))==p['converted_matrix_weights']==21233664
    props=[finite_properties(w[f'weight_{i}'],8,4.) for i in range(12)]
    rows=read(out/'native_metrics.csv');prefixes=read(out/'prefix_metrics.csv');bases=read(out/'baseline_metrics.csv');base_raw=np.load(out/'baseline_tokens.npz')
    for r in bases:
        split,c=r['split'],r['condition'];check(base_raw[f'{split}_{c}_nll'],base_raw[f'{split}_{c}_predictions'],data[split+'_tokens'],r)
        if c=='FP32':np.testing.assert_array_equal(base_raw[f'{split}_{c}_nll'],data[split+'_reference_token_nll'])
    times=[r['samples'] for r in selection['profiles']];assert times==p['calibration_sample_search'][:len(times)]
    assert {(r['split'],int(r['samples']),int(r['sampling_seed'])) for r in rows}=={('validation',t,s) for t in times for s in range(5)}|{('test',selection['selected_samples'],s) for s in range(5)}
    assert len(rows)==final['run_count'] and len(prefixes)==len(rows)*13
    operators=0
    for r in rows:
        split,t,seed=r['split'],int(r['samples']),int(r['sampling_seed']);tokens=data[split+'_tokens'];folder=out/'runs'/f'{split}_T{t}_seed{seed}'
        rec=json.loads((folder/'run.json').read_text());assert rec['status']=='complete' and len(rec['stages'])==12
        assert rec['exact_prefix_input_checks']==int(r['exact_prefix_input_checks'])==len(tokens)*78
        raw=np.load(folder/'prefix_tokens.npz');check(raw['prefix12_nll'],raw['prefix12_predictions'],tokens,r);total=0
        for stage,s in enumerate(rec['stages']):
            qw,var,free,clipped=props[stage];meta=json.loads(Path(s['metadata']).read_text());sig=meta['signature'];ctrl=meta['controls']
            assert sig['seed']==seed and sig['stage']==stage and sig['samples']==t and sig['field_bits']==8 and sig['field_bound']==4.
            assert sig['domain']==p['calibration_domain' if split=='validation' else 'test_domain']
            assert sig['weights_sha256']==final['weights_sha256'] and sig['jax_backend']=='gpu' and sig['matmul_precision']=='highest'
            for path,digest in sig['source_sha256'].items():assert sha(path)==digest
            assert sha(s['inputs'])==sig['inputs_sha256'] and sha(s['reference'])==sig['reference_sha256'] and sha(s['output'])==meta['output_sha256']
            x,ref,y=[np.load(s[k],mmap_mode='r') for k in ('inputs','reference','output')]
            expected=x@w[f'weight_{stage}']+w[f'bias_{stage}'];assert np.max(np.abs(expected-ref))/(1+np.max(np.abs(ref)))<2e-5
            mu=x.astype(np.float64)@qw+w[f'bias_{stage}'];bias=float(np.mean((mu-ref)**2));pred=float(np.mean(np.mean(x.astype(np.float64)**2,axis=(0,1))@var)/t)
            mse=float(np.mean((y.astype(np.float64)-ref)**2));centered=float(np.mean((y.astype(np.float64)-mu)**2))
            for value,k in [(bias,'conditional_control_bias_mse'),(pred,'predicted_sampling_mse'),(pred+bias,'predicted_total_mse'),(mse,'conditional_linear_mse'),(centered,'sampling_mse_vs_programmed_expectation')]:np.testing.assert_allclose(value,meta[k],rtol=1e-9,atol=1e-14)
            assert ctrl['free_pbits']==free and ctrl['range_clipped_fields']==clipped and ctrl['metadata_bits']==w[f'weight_{stage}'].size//128*64
            assert ctrl['hypothetical_packed_controller_bits']==free*8
            batches=(len(tokens)+7)//8;assert meta['bank_batches']==batches
            assert meta['free_node_updates']==batches*t*free
            assert meta['logical_weight_readouts']==batches*t*w[f'weight_{stage}'].size
            assert meta['binary_product_terms']==len(tokens)*65*t*w[f'weight_{stage}'].size
            assert s['prior_native_output_sha256']==[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)]
            total+=meta['free_node_updates'];operators+=1
        assert total==rec['free_node_updates']==int(r['free_node_updates'])
        for prefix in range(13):
            row=next(a for a in prefixes if a['split']==split and int(a['samples'])==t and int(a['sampling_seed'])==seed and int(a['converted_prefix'])==prefix)
            check(raw[f'prefix{prefix}_nll'],raw[f'prefix{prefix}_predictions'],tokens,row)
    refs={split:next(r for r in bases if r['split']==split and r['condition']=='FP32') for split in ('validation','test')}
    for profile in selection['profiles']:
        group=[r for r in rows if r['split']=='validation' and int(r['samples'])==profile['samples']];ce=np.mean([float(r['cross_entropy']) for r in group]);acc=np.mean([float(r['token_accuracy']) for r in group])
        passed=bool(ce-float(refs['validation']['cross_entropy'])<=np.log(1.05) and float(refs['validation']['token_accuracy'])-acc<=.01)
        assert passed==profile['passes'];np.testing.assert_allclose(ce,profile['cross_entropy_mean'],atol=1e-12)
    passing=[r for r in selection['profiles'] if r['passes']];assert selection['gate_passed']==bool(passing)
    assert selection['selected_samples']==(passing[0]['samples'] if passing else times[-1]);assert not selection['test_used_in_selection']
    for row in read(out/'summary.csv'):
        group=[r for r in rows if r['split']==row['split'] and r['samples']==row['samples']];ce=np.array([float(r['cross_entropy']) for r in group]);margin=1.96*ce.std(ddof=1)/np.sqrt(5)
        np.testing.assert_allclose(ce.mean(),float(row['cross_entropy_mean']),atol=1e-12);np.testing.assert_allclose(margin,float(row['cross_entropy_ci95_margin']),atol=1e-12)
        assert int(row['sampling_rounds_per_bank_batch'])==12*int(row['samples'])
    hook=json.loads((out/'independent_hook_audit.json').read_text());assert hook['passed'] and hook['completed_runs_audited']==len(rows)
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoModelForCausalLM
    from .sequential_transformer import SequentialRunner
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}"
    runner=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),p['targets_in_forward_order'])
    for condition in ('FP32','affine4_block128','affine8_block128','programmed_mean8'):
        try:
            if condition!='FP32':
                with torch.no_grad():
                    for i,(module,original) in enumerate(zip(runner.modules,runner.weights)):
                        quantized=props[i][0].astype(np.float32) if condition=='programmed_mean8' else affine(original,4 if condition=='affine4_block128' else 8)
                        module.weight.copy_(torch.from_numpy(quantized))
            for split in ('validation','test'):
                result=runner.evaluate(data[split+'_tokens']);np.testing.assert_array_equal(result['nll'],base_raw[f'{split}_{condition}_nll']);np.testing.assert_array_equal(result['predictions'],base_raw[f'{split}_{condition}_predictions'])
        finally:runner.restore()
    result={'passed':True,'native_runs':len(rows),'operator_checkpoints':operators,'independent_native_hook_runs':len(rows),'independent_deterministic_replays':8,'converted_weights':21233664,'converted_fraction':p['converted_fraction'],
        'source_sha256':sha(__file__),'scope':'Complete declared development schedules and raw twelve-stage lineage, fields, bias/variance MSE, counts, gates and token losses; independent whole-model downstream replay. No new confirmation or physical proof.'}
    (out/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
