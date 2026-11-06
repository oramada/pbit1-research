"""Raw audit of the prospective conservative calibration and new confirmation."""
import json,csv
from pathlib import Path
import numpy as np
from .verify_transformer_control import sha,quantize as finite_properties
from .verify_transformer_quarter import affine

OUT=Path('results/thrml_spw/quarter_conservative')

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def check(nll,pred,tokens,row):
    ce=float(nll.astype(np.float64).mean());acc=float(np.mean(pred==tokens[:,1:]))
    np.testing.assert_allclose(ce,float(row['cross_entropy']),atol=1e-12);np.testing.assert_allclose(acc,float(row['token_accuracy']),atol=1e-12)
    np.testing.assert_allclose(np.exp(ce),float(row['perplexity']),rtol=1e-12)

def independent_profile(nll,pred,ref_nll,ref_pred,tokens,t,seed):
    loss=nll.mean(-1,dtype=np.float64);ref=np.repeat(ref_nll.mean(-1,dtype=np.float64)[None],5,0)
    acc=(pred==tokens[:,1:][None]).mean(-1);refacc=np.repeat((ref_pred==tokens[:,1:]).mean(-1)[None],5,0)
    rng=np.random.default_rng(seed);si=rng.integers(0,5,size=(10000,5));bi=rng.integers(0,16,size=(10000,16))
    def resample(delta):
        by_bank=delta.reshape(5,16,8).mean(-1)
        across_seeds=np.take(by_bank,si,axis=0).mean(1)
        return np.take_along_axis(across_seeds,bi,axis=1).mean(1)
    dc=resample(loss-ref);da=resample(acc-refacc);upper=float(np.quantile(dc,.975));lower=float(np.quantile(da,.025))
    point=bool((loss-ref).mean()<=np.log(1.05) and (acc-refacc).mean()>=-.01)
    return {'samples':t,'cross_entropy_mean':float(loss.mean()),'relative_perplexity':float(np.exp((loss-ref).mean())),
        'token_accuracy_mean':float(acc.mean()),'accuracy_difference':float((acc-refacc).mean()),'point_tolerance_passed':point,
        'bootstrap_CE_upper':upper,'bootstrap_relative_upper':float(np.exp(upper)),'bootstrap_accuracy_difference_lower':lower,
        'passes':bool(point and upper<=np.log(1.05) and lower>=-.01)},dc,da

def main():
    out=OUT;p=json.loads((out/'protocol.json').read_text());final=json.loads((out/'manifest.json').read_text());selection=json.loads((out/'selection.json').read_text());assert final['config']==p
    prep=Path('results/thrml_spw/quarter_conservative_preparation');assert sha(prep/'protocol.json')==p['predeclared_design_sha256'] and sha(prep/'reservation_audit.json')==p['reservation_audit_sha256']
    for key,value in json.loads((prep/'protocol.json').read_text()).items():assert p[key]==value
    for name,key in [('quarter_conservative.py','driver_source_sha256'),('sequential_transformer.py','runner_source_sha256'),('conservative_selection.py','selection_source_sha256')]:assert sha(Path('research/thrml_spw')/name)==p[key]
    assert not selection['test_scores_used_in_selection'] and final['no_test_selection'];assert p['sample_search']==[1024,2048]
    assert sha(p['input_data_path'])==p['input_export_sha256'];data=np.load(p['input_data_path'])
    weights=np.load(out/'weights.npz');assert sha(out/'weights.npz')==p['weights_sha256'];assert sum(weights[f'weight_{i}'].size for i in range(12))==21233664
    properties=[finite_properties(weights[f'weight_{i}'],8,4.) for i in range(12)]
    baseline=read(out/'baseline_metrics.csv');baseline_raw=np.load(out/'baseline_tokens.npz');assert len(baseline)==8
    for row in baseline:
        split,c=row['split'],row['condition'];check(baseline_raw[f'{split}_{c}_nll'],baseline_raw[f'{split}_{c}_predictions'],data[split+'_tokens'],row)
    rows=read(out/'native_metrics.csv');prefixes=read(out/'prefix_metrics.csv');times=[r['samples'] for r in selection['profiles']]
    assert times==p['sample_search'][:len(times)]
    assert {(r['split'],int(r['samples']),int(r['sampling_seed'])) for r in rows}=={('validation',t,s) for t in times for s in range(5)}|{('test',selection['selected_samples'],s) for s in range(5)}
    assert len(rows)==final['run_count'] and len(prefixes)==13*len(rows);count=0
    for row in rows:
        split,t,seed=row['split'],int(row['samples']),int(row['sampling_seed']);tokens=data[split+'_tokens'];assert tokens.shape==(128,65)
        folder=out/'runs'/f'{split}_T{t}_seed{seed}';rec=json.loads((folder/'run.json').read_text());raw=np.load(folder/'prefix_tokens.npz');assert rec['status']=='complete'
        assert len(rec['stages'])==12 and rec['exact_prefix_input_checks']==int(row['exact_prefix_input_checks'])==128*78
        check(raw['prefix12_nll'],raw['prefix12_predictions'],tokens,row);total=0
        for i,stage in enumerate(rec['stages']):
            qw,var,free,clipped=properties[i];meta=json.loads(Path(stage['metadata']).read_text());sig=meta['signature'];ctrl=meta['controls']
            assert sig['seed']==seed and sig['stage']==i and sig['samples']==t and sig['field_bits']==8 and sig['field_bound']==4. and sig['contexts_per_batch']==8
            assert sig['domain']==p['validation_domain' if split=='validation' else 'test_domain'] and sig['weights_sha256']==p['weights_sha256']
            assert sig['jax_backend']=='gpu' and sig['matmul_precision']=='highest'
            for path,digest in sig['source_sha256'].items():assert sha(path)==digest
            assert sha(stage['inputs'])==sig['inputs_sha256'] and sha(stage['reference'])==sig['reference_sha256'] and sha(stage['output'])==meta['output_sha256']
            assert stage['prior_native_output_sha256']==[sha(folder/f'stage{k}_native_output.npy') for k in range(i)]
            x,ref,y=[np.load(stage[k],mmap_mode='r') for k in ['inputs','reference','output']]
            exact=x@weights[f'weight_{i}']+weights[f'bias_{i}'];assert np.max(np.abs(exact-ref))/(1+np.max(np.abs(ref)))<2e-5
            mu=x.astype(np.float64)@qw+weights[f'bias_{i}'];bias=float(np.mean((mu-ref)**2));prediction=float(np.mean(np.mean(x.astype(np.float64)**2,axis=(0,1))@var)/t)
            for value,key in [(bias,'conditional_control_bias_mse'),(prediction,'predicted_sampling_mse'),(bias+prediction,'predicted_total_mse'),(float(np.mean((y.astype(np.float64)-ref)**2)),'conditional_linear_mse'),(float(np.mean((y.astype(np.float64)-mu)**2)),'sampling_mse_vs_programmed_expectation')]:np.testing.assert_allclose(value,meta[key],rtol=1e-9,atol=1e-14)
            n=weights[f'weight_{i}'].size;assert ctrl['free_pbits']==free and ctrl['range_clipped_fields']==clipped and ctrl['metadata_bits']==n//128*64
            assert meta['bank_batches']==16 and meta['free_node_updates']==16*t*free and meta['logical_weight_readouts']==16*t*n and meta['binary_product_terms']==128*65*t*n
            total+=meta['free_node_updates'];count+=1
        assert total==rec['free_node_updates']==int(row['free_node_updates'])
        for i in range(13):
            r=next(r for r in prefixes if r['split']==split and int(r['samples'])==t and int(r['sampling_seed'])==seed and int(r['converted_prefix'])==i)
            check(raw[f'prefix{i}_nll'],raw[f'prefix{i}_predictions'],tokens,r)
    draw_count=0
    for split,t,record,bootseed in [('validation',t,r,p['selection_bootstrap']['seed']) for t,r in zip(times,selection['profiles'])]+[('test',selection['selected_samples'],final['confirmation'],20261112)]:
        raw=[np.load(out/'runs'/f'{split}_T{t}_seed{s}'/'prefix_tokens.npz') for s in range(5)]
        independent,dc,da=independent_profile(np.stack([r['prefix12_nll'] for r in raw]),np.stack([r['prefix12_predictions'] for r in raw]),baseline_raw[split+'_FP32_nll'],baseline_raw[split+'_FP32_predictions'],data[split+'_tokens'],t,bootseed)
        for key,value in independent.items():
            if isinstance(value,bool):assert record[key]==value
            else:np.testing.assert_allclose(record[key],value,rtol=1e-12,atol=1e-15)
        saved=np.load(out/f'{split}_T{t}_selection_draws.npz');np.testing.assert_allclose(saved['CE_difference'],dc,rtol=1e-12,atol=1e-15);np.testing.assert_allclose(saved['accuracy_difference'],da,rtol=1e-12,atol=1e-15);draw_count+=20000
    passing=[r for r in selection['profiles'] if r['passes']];assert selection['gate_passed']==bool(passing);assert selection['selected_samples']==(passing[0]['samples'] if passing else times[-1])
    hook=json.loads((out/'independent_hook_audit.json').read_text());assert hook['passed'] and hook['completed_runs_audited']==len(rows)
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoModelForCausalLM
    from .sequential_transformer import SequentialRunner
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{p['revision']}"
    runner=SequentialRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval(),p['targets_in_forward_order'])
    for i,w in enumerate(runner.weights):np.testing.assert_array_equal(w,weights[f'weight_{i}'])
    for condition in ['FP32','affine4_block128','affine8_block128','programmed_mean8']:
        try:
            if condition!='FP32':
                with torch.no_grad():
                    for i,(module,w) in enumerate(zip(runner.modules,runner.weights)):
                        q=properties[i][0].astype(np.float32) if condition=='programmed_mean8' else affine(w,4 if condition=='affine4_block128' else 8);module.weight.copy_(torch.from_numpy(q))
            for split in ['validation','test']:
                result=runner.evaluate(data[split+'_tokens']);np.testing.assert_array_equal(result['nll'],baseline_raw[f'{split}_{condition}_nll']);np.testing.assert_array_equal(result['predictions'],baseline_raw[f'{split}_{condition}_predictions'])
        finally:runner.restore()
    report={'passed':True,'native_runs':len(rows),'operator_checkpoints':count,'independent_native_hook_runs':len(rows),'independent_deterministic_replays':8,
        'independent_selection_bootstrap_values':draw_count,'validation_targets':8192,'confirmation_targets':8192,'source_sha256':sha(__file__),
        'scope':'Frozen prospective design, all causal inputs/outputs, finite controls and conditional errors, raw scores/counts, independent conservative selection and every selection draw, plus independent actual model replay. No physical-system proof.'}
    (out/'evidence_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
