"""Raw finite-controller audit: actual field lattice, bias floor and token evidence."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path('results/thrml_spw/transformer_control')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def quantize(w,bits,bound):
    blocks=w.reshape(-1,128,w.shape[1]);lower=np.repeat(blocks.min(1),128,0);span=np.repeat(blocks.max(1),128,0)-lower
    p=np.zeros_like(w);np.divide(w-lower,span,out=p,where=span!=0);p=np.clip(p,0,1).astype(np.float64)
    free=(p>0)&(p<1);requested=((np.log(p[free])-np.log1p(-p[free]))/2).astype(np.float32).astype(np.float64)
    levels=2**bits-1;code=np.rint((np.maximum(-bound,np.minimum(bound,requested))+bound)*levels/(2*bound))
    field=(-bound+code*(2*bound/levels)).astype(np.float32).astype(np.float64)
    q=p.copy();q[free]=1/(1+np.exp(-2*field))
    return lower+span*q,span**2*q*(1-q),int(free.sum()),int(np.sum(np.abs(requested)>bound))

def check_scores(nll,pred,tokens,row):
    ce=float(np.mean(nll.astype(np.float64)));acc=float(np.mean(pred==tokens[:,1:]))
    np.testing.assert_allclose(ce,float(row['cross_entropy']),atol=1e-12)
    np.testing.assert_allclose(acc,float(row['token_accuracy']),atol=1e-12)
    np.testing.assert_allclose(np.exp(ce),float(row['perplexity']),rtol=1e-12)
    return ce,acc

def main():
    out=ROOT;protocol=json.loads((out/'protocol.json').read_text());final=json.loads((out/'manifest.json').read_text());selection=json.loads((out/'selection.json').read_text())
    assert protocol['test_labels_used_in_selection'] is False and selection['test_labels_used_in_selection'] is False
    assert sha('research/thrml_spw/transformer_control.py')==protocol['driver_source_sha256']
    assert sha(out/'weights.npz')==final['weights_sha256']
    data=np.load('data/thrml_spw/transformer_projection/export.npz');assert sha('data/thrml_spw/transformer_projection/export.npz')==protocol['input_export_sha256']
    w=np.load(out/'weights.npz');rows=read(out/'native_metrics.csv');prefixes=read(out/'prefix_metrics.csv');base=read(out/'baseline_metrics.csv')
    bits_seen=[p['field_bits'] for p in selection['profiles']]
    assert bits_seen==protocol['field_bit_search'][:len(bits_seen)]
    assert {(r['split'],int(r['field_bits']),int(r['sampling_seed'])) for r in rows}=={('validation',bits,seed) for bits in bits_seen for seed in range(5)}|{('test',bits,seed) for bits in {4,selection['selected_bits']} for seed in range(5)}
    assert len(rows)==final['run_count'];properties={};count=0
    raw_base=np.load(out/'baseline_tokens.npz')
    for r in base:
        split=r['split'];check_scores(raw_base[split+'_FP32_nll'],raw_base[split+'_FP32_predictions'],data[split+'_tokens'],r)
        np.testing.assert_array_equal(raw_base[split+'_FP32_nll'],data[split+'_reference_token_nll'])
    for r in rows:
        split,bits,seed=r['split'],int(r['field_bits']),int(r['sampling_seed']);tokens=data[split+'_tokens']
        folder=out/'runs'/f'{split}_B{bits}_T256_seed{seed}';record=json.loads((folder/'run.json').read_text());raw=np.load(folder/'prefix_tokens.npz')
        assert record['status']=='complete';check_scores(raw['prefix4_nll'],raw['prefix4_predictions'],tokens,r)
        assert record['exact_prefix_input_checks']==int(r['exact_prefix_input_checks'])==10*len(tokens)
        updates=0
        for stage,s in enumerate(record['stages']):
            key=(bits,stage)
            if key not in properties:properties[key]=quantize(w[f'weight_{stage}'],bits,4.)
            qw,var,free,clipped=properties[key];meta=json.loads(Path(s['metadata']).read_text());sig=meta['signature'];ctrl=meta['controls']
            assert sig['field_bits']==bits and sig['field_bound']==4. and sig['stage']==stage and sig['seed']==seed and sig['samples']==256
            assert sig['weights_sha256']==final['weights_sha256'];assert sig['domain']==protocol['calibration_domain' if split=='validation' else 'test_domain']
            for path,digest in sig['source_sha256'].items():assert sha(path)==digest
            for k,signature_key in [('inputs','inputs_sha256'),('reference','reference_sha256')]:assert sha(s[k])==sig[signature_key]
            assert sha(s['output'])==meta['output_sha256']
            x,ref,y=[np.load(s[k]) for k in ('inputs','reference','output')]
            exact=x@w[f'weight_{stage}']+w[f'bias_{stage}']
            assert np.max(np.abs(exact-ref))/(1+np.max(np.abs(ref)))<2e-5
            mean=x.astype(np.float64)@qw+w[f'bias_{stage}'];pred=float(np.mean(np.mean(x.astype(np.float64)**2,axis=(0,1))@var)/256)
            bias=float(np.mean((mean-ref)**2));mse=float(np.mean((y.astype(np.float64)-ref)**2));centered=float(np.mean((y.astype(np.float64)-mean)**2))
            for actual,k in [(pred,'predicted_sampling_mse'),(bias,'conditional_control_bias_mse'),(mse,'conditional_linear_mse'),(centered,'sampling_mse_vs_programmed_expectation'),(pred+bias,'predicted_total_mse')]:np.testing.assert_allclose(actual,meta[k],rtol=1e-9,atol=1e-14)
            assert ctrl['field_bits']==bits and ctrl['range_clipped_fields']==clipped and ctrl['free_pbits']==free
            assert ctrl['metadata_bits']==w[f'weight_{stage}'].size//128*64
            assert ctrl['hypothetical_packed_controller_bits']==bits*free
            assert ctrl['hypothetical_endpoint_mode_bits']==2*w[f'weight_{stage}'].size
            batches=(len(tokens)+7)//8
            assert meta['free_node_updates']==batches*256*free
            assert meta['logical_weight_readouts']==batches*256*w[f'weight_{stage}'].size
            updates+=meta['free_node_updates'];count+=1
            assert s['prior_native_output_sha256']==[sha(folder/f'stage{k}_native_output.npy') for k in range(stage)]
        assert updates==int(r['free_node_updates'])==record['free_node_updates']
        for p in range(5):
            row=next(a for a in prefixes if a['split']==split and int(a['field_bits'])==bits and int(a['sampling_seed'])==seed and int(a['converted_prefix'])==p)
            check_scores(raw[f'prefix{p}_nll'],raw[f'prefix{p}_predictions'],tokens,row)
    ref=next(r for r in base if r['split']=='validation')
    for profile in selection['profiles']:
        group=[r for r in rows if r['split']=='validation' and int(r['field_bits'])==profile['field_bits']]
        ce=np.mean([float(r['cross_entropy']) for r in group]);acc=np.mean([float(r['token_accuracy']) for r in group])
        passed=bool(ce-float(ref['cross_entropy'])<=np.log(1.05) and float(ref['token_accuracy'])-acc<=.01)
        assert passed==profile['passes'];np.testing.assert_allclose(ce,profile['cross_entropy_mean'],atol=1e-12)
    passing=[p for p in selection['profiles'] if p['passes']]
    assert selection['gate_passed']==bool(passing)
    assert selection['selected_bits']==(passing[0]['field_bits'] if passing else bits_seen[-1])
    for row in read(out/'summary.csv'):
        group=[r for r in rows if r['split']==row['split'] and r['field_bits']==row['field_bits']];ce=np.array([float(r['cross_entropy']) for r in group]);margin=1.96*ce.std(ddof=1)/np.sqrt(5)
        np.testing.assert_allclose(ce.mean(),float(row['cross_entropy_mean']),atol=1e-12);np.testing.assert_allclose(margin,float(row['cross_entropy_ci95_margin']),atol=1e-12)
    # Independent expected-control replay through the real frozen downstream model.
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoModelForCausalLM
    from .transformer_block_gpu import BlockRunner
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    snapshot=Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{protocol['revision']}"
    runner=BlockRunner(AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,attn_implementation='eager').float().eval())
    expected=np.load(out/'expected_tokens.npz');expected_rows=read(out/'expected_metrics.csv')
    assert len(expected_rows)==10
    for bits in protocol['field_bit_search']:
        with torch.no_grad():
            for stage,module in enumerate(runner.modules):module.weight.copy_(torch.from_numpy(quantize(w[f'weight_{stage}'],bits,4.)[0].astype(np.float32)))
        for split in ('validation','test'):
            result=runner.evaluate(data[split+'_tokens']);np.testing.assert_array_equal(result['nll'],expected[f'{split}_B{bits}_nll']);np.testing.assert_array_equal(result['predictions'],expected[f'{split}_B{bits}_predictions'])
            row=next(r for r in expected_rows if r['split']==split and int(r['field_bits'])==bits);check_scores(result['nll'],result['predictions'],data[split+'_tokens'],row)
    hook=json.loads((out/'independent_hook_audit.json').read_text());assert hook['passed'] and hook['completed_runs_audited']==len(rows)
    result={'passed':True,'native_runs':len(rows),'native_operators':count,'independent_native_hook_runs':len(rows),'independent_expected_model_replays':10,'scope':'Declared finite-bias calibration/test schedules, actual field lattice, conditional bias+sampling decomposition, raw token losses and independent model replays; not physical controller validation'}
    (out/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
