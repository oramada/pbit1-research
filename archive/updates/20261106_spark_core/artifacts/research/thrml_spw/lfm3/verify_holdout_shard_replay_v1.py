"""Exact already-closed nonzero-batch replay, never an extra MC trial."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from .stage_checkpoint_v1 import StageStore
from .lossless_logits import read_logits
from .holdout_shard_contract_v1 import sha,validate_transport


def deterministic_receipt(record):
    # Only timing and transient file locations may vary. All numeric/resource,
    # key/call/program/shape/rounding fields must remain identical.
    return {k:v for k,v in record.items()if k not in {'elapsed_seconds','input','output'}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--transport',type=Path,required=True);args=ap.parse_args()
    tp=json.loads(args.transport.read_text());protocol=Path(tp['scientific_protocol']);core=json.loads(protocol.read_text());seed=core['seeds'][0]
    assignment=validate_transport(args.transport,protocol,seed,'proof4')
    root=Path(assignment['output_root'])/str(seed)/'proof4';r=json.loads((root/'results.json').read_text());rp=json.loads((root/'protocol.json').read_text())
    assert r['status']=='reserved_batch_shard_produced_pending_merge_and_independent_audit' and r['execution_shard']==assignment
    assert r['targets']==512 and r['articles']==2 and len(r['outputs'])==len(r['metrics'])==4
    old=Path('results/thrml_spw/lfm3_1p7b/holdout_native_v4')/str(seed);oldp=json.loads((old/'protocol.json').read_text())
    for key in ['seed','samples','selection','shared_protocol_sha256','package_sha256']:assert rp[key]==oldp[key]
    old_store=StageStore(old,oldp);new_store=StageStore(root,rp);checked=[]
    for stage in ['BF16','same_fit_BF16','nominal_controls','native']:
        a=old_store.load(4,stage);b=new_store.load(4,stage);assert a is not None and b is not None
        assert a['module_calls']==b['module_calls'] and len(b['module_calls'])==287
        assert len(a['operator_calls'])==len(b['operator_calls'])
        for one,two in zip(a['operator_calls'],b['operator_calls']):assert deterministic_receipt(one)==deterministic_receipt(two)
        x,y=read_logits(old/a['output']['path']),read_logits(root/b['output']['path'])
        np.testing.assert_array_equal(x.view(np.uint32),y.view(np.uint32))
        np.testing.assert_allclose(a['metrics']['token_NLL'],b['metrics']['token_NLL'],rtol=0,atol=1e-12)
        assert next(v for v in r['outputs']if v['stage']==stage)==b['output_item']
        assert next(v for v in r['metrics']if v['stage']==stage)==b['metrics']
        checked.append({'stage':stage,'raw_float32_values':int(x.size),'reference_file_sha256':a['output']['sha256'],
            'new_file_sha256':b['output']['sha256'],'shape_and_data_sha256':b['output_item']['shape_and_data_sha256'],
            'operator_receipts_checked':len(b['operator_calls']),'max_abs_token_loss_difference':float(np.max(np.abs(np.asarray(a['metrics']['token_NLL'])-np.asarray(b['metrics']['token_NLL']))))})
        del x,y
    result={'status':'closed_nonzero_batch_all4raw_stage_outputs_and_receipts_bitwise_replayed','source_sha256':sha(__file__),
        'transport_sha256':sha(args.transport),'shared_protocol_sha256':sha(protocol),'seed':seed,'batch':4,
        'source_case_results_sha256':sha(root/'results.json'),'reference_closed_checkpoints_sha256':tp['closed_reference_checkpoints_sha256'],
        'stages':checked,'scope':'SameoriginalMCkeyandglobalcall indices, batch4onlyalreadyclosed/exposed. Verificationworknotnewtrial. Proves nonzero-batchindependencefortheseactualshortinputs andunchangedcoremath; completefuturecasesstillneedoriginalfullcase/statisticalaudits.'}
    with (root/'replay_audit.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
