"""Freeze complete R1 quality execution after all actual-model/sampler gates."""
from pathlib import Path
from datetime import datetime,timezone
import copy,json,hashlib
from research.thrml_spw.broad.r1_io_v1 import BASE,sha
from .layout import APP_NAME,PROTOCOL,OUTPUT,LOG_DIRECTORY


def main():
    preflight_root=BASE/'r1_model_preflight_v2';preflight_protocol=preflight_root/'protocol.json'
    preflight=json.loads(preflight_protocol.read_text());pp_hash=sha(preflight_protocol)
    for key in ['gpt2','distilbert']:
        gate=json.loads((preflight_root/(key+'_results.json')).read_text())
        assert gate['status']=='actual_model_all7controls_and_fullparameter_T1native_neural_replay_passed_synthetic_only'
        assert gate['model']==key and gate['protocol_sha256']==pp_hash and len(gate['cases'])==8
        expected={name for name,c in preflight['cases'].items() if c['model_key']==key}
        assert {c['case'] for c in gate['cases']}==expected
        for a in gate['cases']:
            assert a['synthetic_only'] and a['protocol_sha256']==pp_hash and a['evaluation_batches']==1
            assert a['source_sha256']==preflight['sources_sha256']['research/thrml_spw/broad/verify_r1_case_v1.py']
    for path,digest in preflight['sources_sha256'].items():assert sha(path)==digest,path
    stat_gate=json.loads((BASE/'r1_statistics_v1_preflight/results.json').read_text())
    batch_gate=json.loads((BASE/'r1_batches_v1_preflight/results.json').read_text())
    assert stat_gate['status']=='complete_cohorts_dual_control_weighted_bootstrap_oracle_and_forgery_rejections_passed'
    assert batch_gate['status']=='all_real_task_batch_indices_masks_labels_targets_and_synthetic_loss_oracles_verified'
    for gate in [stat_gate,batch_gate]:
        for source,digest in gate['sources_sha256'].items():assert sha(source)==digest,source
    package_gate=json.loads((BASE/'additional_parameter_packages_v2_audit.json').read_text())
    assert package_gate['status']=='both_complete_parameter_packages_and_every_code_range_BF16point_alias_and_allocation_independently_verified'
    selection_path=BASE/'additional_model_inventory/selection_protocol_v1.json';selection=json.loads(selection_path.read_text())
    sources=dict(preflight['sources_sha256'])
    for name in ['r1_statistics_v1.py','verify_r1_statistics_v1.py','preflight_r1_statistics_v1.py','preflight_r1_batches_v1.py','task_statistics_v1.py']:
        path=Path('research/thrml_spw/broad')/name;sources[str(path)]=sha(path)
    for path in [Path(__file__),Path(__file__).with_name('app.py'),Path(__file__).with_name('layout.py'),Path(__file__).with_name('client.py')]:
        path=path.relative_to(Path.cwd()) if path.is_absolute() else path;sources[str(path)]=sha(path)
    inputs=dict(preflight['inputs_sha256']);transfers={}
    def bind(remote,local=None):
        remote=Path(remote);local=Path(local) if local is not None else remote
        assert not remote.is_absolute();inputs[str(remote)]=sha(local)
        if remote==local:transfers[str(remote)]=str(local)
    for path in [selection_path,preflight_protocol,BASE/'r1_statistics_v1_preflight/results.json',BASE/'r1_batches_v1_preflight/results.json',
                 BASE/'additional_assets_v1_independent_data_audit.json',preflight_root/'gpt2_results.json',preflight_root/'distilbert_results.json']:
        bind(path)
    data_mirror=BASE/'additional_assets_v1_mirror';models={};cases={};controls=[];primary=[]
    for item in selection['models']:
        key='gpt2' if item['id']=='openai-community/gpt2' else 'distilbert';spec=copy.deepcopy(preflight['models'][key])
        spec['unique_learned_scalars']=item['unique_learned_scalars'];spec['seeds_by_budget']=item['seeds_by_budget']
        spec['bootstrap_key']=item['bootstrap_key'];spec['margin']=list(item['primary_margin'].values())[0]
        arrays=BASE/'additional_assets_v1'/(key+'_tokens.npz');layout=BASE/'additional_assets_v1'/(key+'_layout.json')
        bind(arrays,data_mirror/arrays);bind(layout,data_mirror/layout)
        spec['data']={'kind':'full_task','arrays':str(arrays),'layout':str(layout)}
        spec['baseline_scope']='Uniform uncalibrated minmaxblock128 INT2/4/8 across every learnedparameter, includingnorms/biases; not optimizedGPTQ/AWQ or hardwarekernel performance.'
        bind(Path(spec['checkpoint'])/'config.json',data_mirror/Path(spec['checkpoint'])/'config.json')
        models[key]=spec
        for arm in ['original','mean1','mean4','mean8','int2','int4','int8']:
            slug=key+'__'+arm;cases[slug]={'model_key':key,'mode':'control','arm':arm};controls.append(slug)
        for budget in [1024,4096,8192]:
            seeds=spec['seeds_by_budget'][str(budget)];assert len(seeds)==len(set(seeds))==5
            for bits in [1,4,8]:
                for seed in seeds:
                    slug=f'{key}__K{bits}_C{budget}_seed{seed}'
                    cases[slug]={'model_key':key,'mode':'native','bits':bits,'budget':budget,'samples':budget//bits,'seed':seed}
                    if bits==1 and budget==8192:primary.append(slug)
    # Interleave architectures within each fixed stage; completion/quality never
    # changes condition membership, seeds, sample budgets or the later queue.
    controls=[f'{m}__{a}' for a in ['original','mean1','mean4','mean8','int2','int4','int8'] for m in ['gpt2','distilbert']]
    primary=[f'{m}__K1_C8192_seed{models[m]["seeds_by_budget"]["8192"][i]}' for i in range(5) for m in ['gpt2','distilbert']]
    remainder=[s for s in cases if s not in set(controls+primary)];order=controls+primary+remainder
    assert len(cases)==len(order)==len(set(order))==104 and len(controls)==14 and len(primary)==10
    assert all(not Path(path).is_absolute() for path in [*sources,*inputs])
    p={'status':'complete_R1_execution_and_statistics_frozen_before_scoring','recorded_utc':datetime.now(timezone.utc).isoformat(),
        'output':OUTPUT,'app':APP_NAME,'logs':LOG_DIRECTORY,'models':models,'cases':cases,
        'case_order':order,'control_cases':controls,'primary_cases':primary,'sources_sha256':sources,'inputs_sha256':inputs,
        'preflight_protocol_sha256':pp_hash,'sampler_tile_size':65536,'source_point':'Originalpinnedcheckpoint BF16rounded, alllearnedscalarsincludingbiases/norms, tiesonce',
        'readout':'Exactint64positionalnumerator, FP64division andlower+span*p, castFP32thenBF16onCPU, move toH100',
        'statistics':selection['statistics'],'bootstrap_replicates':10000,'MC_critical':2.7764451051977987,
        'primary_rule':'K1C8192bothmodels. BothconditionalMCt4andcrossedseed/data upper95percentbounds must meet existingmargin vsoriginalBF16andK1finiteprogrammeanBF16. No additionalfavorablekeys, ensemblepredictions, filteringorfitting.',
        'resource_unit':'Transitions percompleteweightreconstructiondraw, reused foralltestexamples; baselinebuffersandBF16modelstorage retained. Not per-inputR2costorphysicalmemorysavings.',
        'execution':{'gpu':'H100!','parallel_case_limit':4,'case_timeout_seconds':86400,'user_sampling_retries':0,
            'infrastructure_recovery':'StableFunctionCallownership, skip closedparameters/evaluationbatches, same-keypartialreplaychargedseparately',
            'all_outcomes_retained':True,'queue_independent_of_quality':True},
        'evidence':'AllcountsandBF16weightpoint hashes retained; allclassifierlogits saved, fullGPT2batchlogit hashes and targetlosscomponents retained and everybatch independently replayed without sampling.',
        'scope':'Fullfixedtask R1quality only. Approximate confidence intervals, no universal numericalBF16equivalence, R2or physicalhardwareadvantage claim.'}
    root=Path(OUTPUT);root.mkdir(exist_ok=False)
    with Path(PROTOCOL).open('x') as f:json.dump(p,f,indent=2);f.write('\n')
    transfers[PROTOCOL]=PROTOCOL
    with (root/'transfer_manifest.json').open('x') as f:json.dump({'protocol_sha256':sha(PROTOCOL),'new_metadata_files':transfers},f,indent=2);f.write('\n')
    for path,digest in sources.items():
        target=Path('results/thrml_spw/source_snapshots',digest+'.py')
        if not target.exists():target.write_bytes(Path(path).read_bytes())
    print(json.dumps({'status':'full104case_R1_execution_and_statistics_frozen','protocol_sha256':sha(PROTOCOL),'sources':len(sources),'cases':len(cases)}))

if __name__=='__main__':main()
