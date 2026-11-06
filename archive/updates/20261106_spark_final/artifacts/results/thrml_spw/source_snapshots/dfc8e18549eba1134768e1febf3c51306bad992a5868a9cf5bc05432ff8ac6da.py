"""Lossless 4-shard merge into the original complete-case audit contract.

No sampling, no model forward, no metric changes, no incomplete-case admission.
Original full-case and statistical auditors remain mandatory after merging.
"""
from pathlib import Path
import argparse,copy,json,os,shutil,time
import numpy as np
from .stage_checkpoint_v1 import StageStore,canonical
from .holdout_shard_contract_v1 import sha,validate_transport,validate_subset
from .holdout_batch_layout import assemble_losses

STAGES=['BF16','same_fit_BF16','nominal_controls','native']
CORE_KEYS=['seed','samples','selection','shared_protocol_sha256','package_sha256']


def collect(root,transport,seed):
    root=Path(root);tp=json.loads(Path(transport).read_text());core_path=Path(tp['scientific_protocol']);core=json.loads(core_path.read_text());plan=core['batch_plan']
    records={};shards=[];contract=None;weight_hashes=None;calls=[]
    for shard in ['0','1','2','3']:
        assignment=validate_transport(transport,core_path,seed,shard)
        directory=Path(assignment['output_root'])/str(seed)/shard
        rp=json.loads((directory/'protocol.json').read_text());r=json.loads((directory/'results.json').read_text())
        assert r['status']=='reserved_batch_shard_produced_pending_merge_and_independent_audit'
        assert r['execution_shard']==rp['execution_shard']==assignment
        cp={key:rp[key]for key in CORE_KEYS}
        if contract is None:contract=cp;weight_hashes=r['same_fit_BF16_weight_hashes']
        else:assert cp==contract and r['same_fit_BF16_weight_hashes']==weight_hashes
        assert r['seed']==seed and r['samples']==8192 and r['pbits_per_parameter']==1
        assert r['shared_protocol_sha256']==sha(core_path) and r['unique_parameter_coordinates']==1728801536
        assert r['unique_programs']==286 and r['weighted_modules']==287
        assert r['targets']==2560 and r['articles']==10 and len(r['outputs'])==len(r['metrics'])==16
        store=StageStore(directory,rp)
        for stage in STAGES:
            selected=[v for v in r['metrics']if v['stage']==stage]
            validate_subset(plan,assignment['batches'],[{'batch_index':v['batch_index'],'token_NLL':v['token_NLL']}for v in selected])
            for batch in assignment['batches']:
                key=(batch,stage)
                if key in records:raise ValueError('Overlapping source shard stages')
                closed=store.load(batch,stage);assert closed is not None
                matches=[v for v in r['outputs']if v['batch']==batch and v['stage']==stage]
                metric=[v for v in r['metrics']if v['batch_index']==batch and v['stage']==stage]
                assert matches==[closed['output_item']] and metric==[closed['metrics']]
                assert closed['legacy_replay']is None and len(closed['module_calls'])==287
                expected=[v for v in r['operator_calls']if v['batch']==batch and v['stage']==stage]
                assert closed['operator_calls']==expected and len(expected)==(287 if stage in ['nominal_controls','native']else 0)
                for index,receipt in enumerate(expected):assert receipt['call']==287+batch*287+index
                records[key]=(store,closed)
        shards.append({'shard':shard,'path':str(directory),'results_sha256':sha(directory/'results.json'),
            'protocol_sha256':sha(directory/'protocol.json'),'core_elapsed_seconds':r['elapsed_seconds'],
            'worker_ready':r['worker_ready']})
    if set(records)!={(i,s)for i in range(16)for s in STAGES}:raise ValueError('Incomplete full case')
    for stage in STAGES:assemble_losses(plan,[{'batch_index':i,'token_NLL':records[(i,stage)][1]['metrics']['token_NLL']}for i in range(16)])
    return core,contract,weight_hashes,records,shards


def copy_immutable(source,destination):
    # A copy is deliberate: final audited artifacts cannot depend on mutable
    # external paths or cross-directory hardlinks. Never overwrite any bytes.
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb')as src,destination.open('xb')as dest:shutil.copyfileobj(src,dest,16*1024*1024)
    assert sha(source)==sha(destination)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--transport',type=Path,required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--case-started-unix',type=float,required=True);args=ap.parse_args();begin=time.monotonic()
    root=Path('results/thrml_spw/lfm3_1p7b');core,contract,weight_hashes,records,shards=collect(root,args.transport,args.seed)
    out=(root/'holdout_native_v4'/str(args.seed)).resolve();out.mkdir(parents=True,exist_ok=True)
    merge_contract={'seed':args.seed,'shared_protocol_sha256':contract['shared_protocol_sha256'],
        'transport_sha256':sha(args.transport),'source_shards':shards,'merge_source_sha256':sha(__file__),
        'case_submission_unix':args.case_started_unix}
    marker=out/'transport_merge_contract.json'
    if marker.exists():assert json.loads(marker.read_text())==merge_contract
    else:
        if (out/'contract.json').exists()or list(out.glob('stage_*.json')):raise ValueError('Preexisting unbound/sequential case; no merge overwrite')
        StageStore(out/'merge_receipt',merge_contract)._atomic_json(marker,merge_contract)
    store=StageStore(out,contract)
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==contract
    else:store._atomic_json(out/'protocol.json',contract)
    if (out/'results.json').exists():
        old=json.loads((out/'results.json').read_text());assert old['execution_transport']['merge_contract_sha256']==sha(marker);return
    attempt=store.new_attempt();outputs=[];metrics=[];calls=[];transfers=[]
    for batch in range(16):
        for stage in STAGES:
            source_store,source_record=records[(batch,stage)];closed=store.load(batch,stage)
            if closed is None:
                source=source_store.root/source_record['output']['path'];target=attempt/f'batch{batch}_{stage}.npz'
                copy_immutable(source,target);item=copy.deepcopy(source_record['output_item']);item['path']=str(target)
                rec=copy.deepcopy(source_record);rec['output']={'path':str(target.relative_to(out)),'sha256':sha(target)};rec['output_item']=item
                store.commit(batch,stage,rec);closed=store.load(batch,stage)
                transfers.append({'batch':batch,'stage':stage,'bytes':target.stat().st_size,'sha256':sha(target)})
            assert closed['output']['sha256']==source_record['output']['sha256'] and closed['metrics']==source_record['metrics']
            assert closed['operator_calls']==source_record['operator_calls']
            outputs.append(closed['output_item']);metrics.append(closed['metrics']);calls.extend(closed['operator_calls'])
    r={'status':'full_reserved_native_case_produced_pending_independent_audit','seed':args.seed,'samples':8192,'pbits_per_parameter':1,
        'unique_programs':286,'weighted_modules':287,'unique_parameter_coordinates':1728801536,'targets':10240,'articles':40,
        'outputs':outputs,'metrics':metrics,'operator_calls':calls,'same_fit_BF16_weight_hashes':weight_hashes,
        'worker_ready':shards[0]['worker_ready'],'shared_protocol_sha256':contract['shared_protocol_sha256'],
        'elapsed_seconds':time.time()-args.case_started_unix,
        'elapsed_seconds_definition':'Submission-to-mergewallincludingqueue/setup/parallelworkers/copy,notlegacycore-onlywallandnotphysicalhardwarelatency.',
        'execution_transport':{'transport_sha256':sha(args.transport),'merge_contract_sha256':sha(marker),
            'merge_elapsed_seconds':time.monotonic()-begin,'shard_core_seconds':[s['core_elapsed_seconds']for s in shards],
            'sum_shard_core_seconds':sum(s['core_elapsed_seconds']for s in shards),'extra_sampling_for_merge':0,'new_transfers':transfers,
            'scope':'FoursoftwarebatchreplicasofonefixedMCseed,notfourpbitsperparameterorextraseeds. Everyglobalbatch/target/callindexcoveredexactlyonce. Fullrawcaseauditorandtransportauditorstillrequired.'},
        'recovery':{'checkpoint_contract_sha256':store.contract_sha256,'attempt_directories':[str(p.relative_to(out))for p in sorted((out/'attempts').iterdir())],
            'resource_accounting':'Acceptedfulltrajectory unchanged. Partial/replayedworkretainedinoriginalshardattempts; source_shardsboundinmergecontract. Copyworkseparate,nevernewMCseeds.'},
        'scope':'Completefixedteacherforcedholdoutassembledlosslesslyfrom4globalbatchshards, pendingoriginalindependentrawcaseauditandtransportbindingaudit. No physicalhardware/numericalBF16/autoregressivequalityclaim.'}
    store._atomic_json(out/'results.json',r)
    print(json.dumps({'status':r['status'],'seed':args.seed,'outputs':len(outputs),'operator_calls':len(calls),'extra_sampling':0}))

if __name__=='__main__':main()
