"""Independent confirmation audit of raw arrays, frozen choices and model replay."""
import csv,json
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha,quantize as finite_properties

OUT=Path('results/thrml_spw/transformer_confirmation')

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def float_properties(w):
    block=w.reshape(-1,128,w.shape[1]);lower=np.repeat(block.min(1),128,0);span=np.repeat(block.max(1),128,0)-lower
    p=np.zeros_like(w);np.divide(w-lower,span,out=p,where=span!=0);p=np.clip(p,0,1).astype(np.float64)
    free=(p>0)&(p<1);b=((np.log(p[free])-np.log1p(-p[free]))/2).astype(np.float32).astype(np.float64)
    q=p.copy();q[free]=np.exp(-np.logaddexp(0,-2*b))
    return lower+span*q,span**2*q*(1-q),int(free.sum())

def scores(nll,pred,tokens,row):
    ce=float(nll.astype(np.float64).mean());acc=float(np.mean(pred==tokens[:,1:]))
    np.testing.assert_allclose(ce,float(row['cross_entropy']),atol=1e-12);np.testing.assert_allclose(acc,float(row['token_accuracy']),atol=1e-12)
    np.testing.assert_allclose(np.exp(ce),float(row['perplexity']),rtol=1e-12)

def main():
    out=OUT;p=json.loads((out/'protocol.json').read_text());final=json.loads((out/'manifest.json').read_text())
    assert p==final['config'] and final['no_confirmation_tuning'] and not p['confirmation_used_for_tuning']
    assert p['driver_source_sha256']==sha('research/thrml_spw/transformer_confirmation.py')
    assert sha('results/thrml_spw/transformer_control/selection.json')==p['selected_calibration_sha256']
    assert sha('results/thrml_spw/transformer_confirmation_preparation/protocol.json')==p['reservation_protocol_sha256']
    assert sha('results/thrml_spw/transformer_confirmation_preparation/reservation_audit.json')==p['reservation_audit_sha256']
    assert p['selected_field_bits']==8 and p['samples']==256 and p['confirmation_domain']==960730
    assert p['sampling_seeds']==list(range(5))
    assert sha(p['input_data_path'])==p['input_export_sha256'];data=np.load(p['input_data_path']);tokens=data['tokens'];assert tokens.shape==(128,65)
    assert sha(out/'weights.npz')==p['weights_sha256'];weights=np.load(out/'weights.npz')
    baseline=np.load(out/'baseline_tokens.npz');base_rows=read(out/'baseline_metrics.csv')
    for row in base_rows:scores(baseline[row['condition']+'_nll'],baseline[row['condition']+'_predictions'],tokens,row)
    rows=read(out/'native_metrics.csv');prefixes=read(out/'prefix_metrics.csv')
    assert {(r['condition'],int(r['sampling_seed'])) for r in rows}=={(c,s) for c in ('native_float','native_finite8') for s in range(5)}
    assert len(rows)==final['run_count']==10;assert len(prefixes)==50
    properties={};operators=0
    for r in rows:
        cond,seed=r['condition'],int(r['sampling_seed']);folder=out/'runs'/f'{cond}_seed{seed}'
        rec=json.loads((folder/'run.json').read_text());assert rec['status']=='complete';assert rec['exact_prefix_input_checks']==1280
        raw=np.load(folder/'prefix_tokens.npz');scores(raw['prefix4_nll'],raw['prefix4_predictions'],tokens,r);total=0
        for stage,s in enumerate(rec['stages']):
            key=(cond,stage)
            if key not in properties:
                w=weights[f'weight_{stage}'];properties[key]=finite_properties(w,8,4.)[:3] if cond=='native_finite8' else float_properties(w)
            meanw,var,free=properties[key]
            meta=json.loads(Path(s['metadata']).read_text());sig=meta['signature'];controls=meta['controls'];assert meta['status']=='complete'
            assert sig['seed']==seed and sig['stage']==stage and sig['samples']==256 and sig['domain']==960730 and sig['contexts_per_batch']==8
            assert sig['weights_sha256']==p['weights_sha256'] and sig['jax_backend']=='gpu' and sig['matmul_precision']=='highest'
            for path,digest in sig['source_sha256'].items():assert sha(path)==digest
            assert sha(s['inputs'])==sig['inputs_sha256'] and sha(s['reference'])==sig['reference_sha256'] and sha(s['output'])==meta['output_sha256']
            x,ref,y=[np.load(s[k],mmap_mode='r') for k in ('inputs','reference','output')]
            expected=x@weights[f'weight_{stage}']+weights[f'bias_{stage}']
            assert np.max(np.abs(expected-ref))/(1+np.max(np.abs(ref)))<2e-5
            mse=float(np.mean((y.astype(np.float64)-ref)**2));pred=float(np.mean(np.mean(x.astype(np.float64)**2,axis=(0,1))@var)/256)
            np.testing.assert_allclose(mse,meta['conditional_linear_mse'],rtol=1e-10);np.testing.assert_allclose(pred,meta['predicted_sampling_mse'],rtol=1e-9)
            if cond=='native_finite8':
                assert sig['field_bits']==8 and sig['field_bound']==4.
                expected=x.astype(np.float64)@meanw+weights[f'bias_{stage}'];bias=float(np.mean((expected-ref)**2));centered=float(np.mean((y.astype(np.float64)-expected)**2))
                np.testing.assert_allclose(bias,meta['conditional_control_bias_mse'],rtol=1e-9,atol=1e-14)
                np.testing.assert_allclose(centered,meta['sampling_mse_vs_programmed_expectation'],rtol=1e-9)
                np.testing.assert_allclose(pred+bias,meta['predicted_total_mse'],rtol=1e-9)
                assert controls['hypothetical_packed_controller_bits']==free*8
            else:assert 'field_bits' not in sig
            assert meta['bank_batches']==16 and controls['metadata_bits']==weights[f'weight_{stage}'].size//128*64
            assert meta['free_node_updates']==16*256*free
            assert meta['logical_weight_readouts']==16*256*weights[f'weight_{stage}'].size
            assert meta['binary_product_terms']==128*65*256*weights[f'weight_{stage}'].size
            assert s['prior_native_output_sha256']==[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)]
            total+=meta['free_node_updates'];operators+=1
        assert total==int(r['free_node_updates'])==rec['free_node_updates']
        for prefix in range(5):
            row=next(a for a in prefixes if a['condition']==cond and int(a['sampling_seed'])==seed and int(a['converted_prefix'])==prefix)
            scores(raw[f'prefix{prefix}_nll'],raw[f'prefix{prefix}_predictions'],tokens,row)
    fp=next(r for r in base_rows if r['condition']=='FP32')
    for row in read(out/'summary.csv'):
        group=[r for r in rows if r['condition']==row['condition']];ce=np.array([float(r['cross_entropy']) for r in group]);margin=1.96*ce.std(ddof=1)/np.sqrt(5)
        delta=ce.mean()-float(fp['cross_entropy']);acc=np.mean([float(r['token_accuracy']) for r in group])
        for value,k in [(ce.mean(),'cross_entropy_mean'),(margin,'cross_entropy_ci95_margin'),(np.exp(delta),'relative_perplexity'),(np.exp(delta-margin),'relative_perplexity_lower'),(np.exp(delta+margin),'relative_perplexity_upper')]:np.testing.assert_allclose(value,float(row[k]),rtol=1e-12)
        assert (row['mean_quality_tolerance_passed']=='True')==bool(delta<=np.log(1.05) and float(fp['token_accuracy'])-acc<=.01)
    hook=json.loads((out/'independent_hook_audit.json').read_text());assert hook['passed'] and hook['completed_runs_audited']==10
    # Independently reproduce both deterministic baselines with output hooks intact.
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoModelForCausalLM
    from .transformer_block_gpu import BlockRunner
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}"
    runner=BlockRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval())
    for condition in ('FP32','affine_zero_block128_4'):
        if condition!='FP32':
            with torch.no_grad():
                for module,w in zip(runner.modules,runner.weights):
                    blocks=w.reshape(-1,128,w.shape[1]);lo=np.minimum(blocks.min(1,keepdims=True),0);hi=np.maximum(blocks.max(1,keepdims=True),0);scale=(hi-lo)/15
                    zero=np.clip(np.rint(np.divide(-lo,scale,out=np.zeros_like(lo),where=scale!=0)),0,15)
                    q=(np.clip(np.rint(np.divide(blocks,scale,out=np.zeros_like(blocks),where=scale!=0))+zero,0,15)-zero)*scale
                    module.weight.copy_(torch.from_numpy(q.reshape(w.shape)))
        result=runner.evaluate(tokens);np.testing.assert_array_equal(result['nll'],baseline[condition+'_nll']);np.testing.assert_array_equal(result['predictions'],baseline[condition+'_predictions'])
    result={'passed':True,'confirmation_native_runs':10,'native_operator_checkpoints':operators,'independent_native_hook_replays':10,'independent_deterministic_model_replays':2,
        'reserved_target_tokens':8192,'frozen_selected_bits':8,'source_sha256':sha(__file__),
        'scope':'Frozen choices, reserved provenance, all current student operator arrays and lineage, counts and conditional MSE, token losses and independent actual model replay; not full-model or hardware proof'}
    (out/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
