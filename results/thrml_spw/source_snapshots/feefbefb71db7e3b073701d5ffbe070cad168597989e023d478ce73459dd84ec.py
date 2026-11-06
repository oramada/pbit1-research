"""Separate transport/byte/coverage proof after the original full-case audit."""
from pathlib import Path
import argparse,json
from .stage_checkpoint_v1 import StageStore
from .holdout_shard_contract_v1 import sha,validate_transport
from .holdout_batch_layout import assemble_losses

STAGES=['BF16','same_fit_BF16','nominal_controls','native']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--transport',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    t=json.loads(args.transport.read_text());core_path=Path(t['scientific_protocol']);p=json.loads(core_path.read_text())
    out=Path('results/thrml_spw/lfm3_1p7b/holdout_native_v4')/str(args.seed)
    r=json.loads((out/'results.json').read_text());audit=json.loads((out/'independent_audit.json').read_text());rp=json.loads((out/'protocol.json').read_text())
    assert audit['status']=='full_reserved_case_raw_losses_controls_coverage_and_resources_independently_verified'
    assert audit['source_sha256']==p['independent_case_verifier_sha256'] and audit['shared_protocol_sha256']==sha(core_path)
    assert audit['results_sha256']==sha(out/'results.json') and audit['seed']==args.seed
    marker=json.loads((out/'transport_merge_contract.json').read_text())
    assert marker['transport_sha256']==r['execution_transport']['transport_sha256']==sha(args.transport)
    assert r['execution_transport']['merge_contract_sha256']==sha(out/'transport_merge_contract.json')
    assert marker['seed']==r['seed']==args.seed and r['targets']==10240 and r['articles']==40 and r['samples']==8192
    assert r['execution_transport']['extra_sampling_for_merge']==0
    full=StageStore(out,rp);seen=set();bindings={};all_calls=[];records={s:[]for s in STAGES}
    for shard in ['0','1','2','3']:
        a=validate_transport(args.transport,core_path,args.seed,shard);root=Path(a['output_root'])/str(args.seed)/shard
        ss=json.loads((root/'protocol.json').read_text());sr=json.loads((root/'results.json').read_text())
        assert ss['execution_shard']==sr['execution_shard']==a
        assert sr['same_fit_BF16_weight_hashes']==r['same_fit_BF16_weight_hashes']
        assert sr['worker_ready']['ready'] and any('cuda'in x.lower()or'gpu'in x.lower()for x in sr['worker_ready']['devices'])
        for key in ['seed','samples','selection','shared_protocol_sha256','package_sha256']:assert ss[key]==rp[key]
        source=StageStore(root,ss);binding={'results_sha256':sha(root/'results.json'),'protocol_sha256':sha(root/'protocol.json')}
        expected=next(x for x in marker['source_shards']if x['shard']==shard)
        assert all(binding[k]==expected[k]for k in binding);bindings[shard]=binding
        for batch in list(range(int(shard),16,4)):
            for stage in STAGES:
                key=(batch,stage);assert key not in seen;seen.add(key)
                original=source.load(batch,stage);merged=full.load(batch,stage);assert original is not None and merged is not None
                assert original['output']['sha256']==merged['output']['sha256']
                assert original['metrics']==merged['metrics'] and original['module_calls']==merged['module_calls']
                assert original['operator_calls']==merged['operator_calls'] and original['legacy_replay']==merged['legacy_replay']is None
                item=[x for x in r['outputs']if x['batch']==batch and x['stage']==stage];assert item==[merged['output_item']]
                metrics=[x for x in r['metrics']if x['batch_index']==batch and x['stage']==stage];assert metrics==[merged['metrics']]
                records[stage].append({'batch_index':batch,'token_NLL':merged['metrics']['token_NLL']})
    assert seen=={(i,s)for i in range(16)for s in STAGES}
    for stage in STAGES:assemble_losses(p['batch_plan'],records[stage])
    for batch in range(16):
        for stage in STAGES:all_calls.extend(full.load(batch,stage)['operator_calls'])
    assert all_calls==r['operator_calls'] and len(all_calls)==9184
    for batch in range(16):
        for stage in ['nominal_controls','native']:
            group=[v for v in all_calls if v['batch']==batch and v['stage']==stage];assert len(group)==287
            for i,rec in enumerate(group):assert rec['call']==287+batch*287+i
    proof=json.loads(Path(t['replay_proof']).read_text());assert sha(t['replay_proof'])==t['replay_proof_sha256']
    proof_t=Path(t['proof_transport']);assert sha(proof_t)==t['proof_transport_sha256'] and proof['transport_sha256']==sha(proof_t)
    original_transport=json.loads(proof_t.read_text())
    for source in ['research/thrml_spw/lfm3/holdout_native_case_shard_v1.py','research/thrml_spw/lfm3/holdout_shard_contract_v1.py']:
        assert t['sources_sha256'][source]==original_transport['sources_sha256'][source]==sha(source)
    result={'status':'four_shard_transport_merged_raw_bytes_fullcoverage_originalcaseaudit_verified','seed':args.seed,
        'source_sha256':sha(__file__),'transport_sha256':sha(args.transport),'shared_protocol_sha256':sha(core_path),
        'results_sha256':sha(out/'results.json'),'original_case_audit_sha256':sha(out/'independent_audit.json'),
        'merge_contract_sha256':sha(out/'transport_merge_contract.json'),'source_shards':bindings,
        'raw_stages_verified':64,'global_call_receipts_verified':9184,'MC_seeds_represented':1,'additional_merge_sampling':0,
        'scope':'Originalfullcaseauditorunchangedandpassed; all4shardbytes/checkpoints/coverage/callindicesbound independently. FoursoftwarebatchworkersarenotextraMCseedsorpbitsperparameter. Originalfive-seedstatisticsstillrequired; no hardwarelatencyorqualityclaim.'}
    full._atomic_json(out/'transport_audit_v1.json',result);print(json.dumps(result))

if __name__=='__main__':main()
