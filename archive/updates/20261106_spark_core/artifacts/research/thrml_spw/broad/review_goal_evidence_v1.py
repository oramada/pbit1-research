"""Point-in-time requirement/evidence gap review, never a completion oracle."""
from pathlib import Path
import hashlib,json,time
import modal

BASE=Path('results/thrml_spw/broad_verification_20261105')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    evidence={}
    def read(path):
        p=Path(path);evidence[str(p)]=sha(p);return json.loads(p.read_text())
    parent=read(BASE/'r1_execution_v1/protocol.json');primary=read(BASE/'r1_execution_v1/remote_primary.json');audit=read(BASE/'r1_execution_v1/remote_primary_audit.json')
    assert audit['statistics_sha256']==sha(BASE/'r1_execution_v1/remote_primary.json') and audit['cases_checked']==24
    assert primary['models']['distilbert']['primary_quality_gate'] and not primary['models']['gpt2']['primary_quality_gate'] and not primary['primary_both_models_pass']
    exposure=read(BASE/'prior_holdout_exposure.json');assert exposure['prior_BF16_control_targets']==512 and exposure['previously_unexposed_targets']==9728
    holdout=read('results/thrml_spw/lfm3_1p7b/holdout_native_v4_protocol.json');assert holdout['targets']==10240 and holdout['articles']==40 and len(holdout['seeds'])==5
    correlation=read(BASE/'correlation_independent_audit_v1/results.json');assert correlation['conditions']==108 and correlation['models']==9
    nonideal=read(BASE/'nonideality_evidence_v1/results.json');assert [x['original_crossed_gate_met']for x in nonideal['calibration']['conditions']]==[True,False,False]
    counts=read(BASE/'primary_count_moments_v1_results.json');assert counts['all_aggregate_moment_gates_met']
    high=read(BASE/'highT_execution_v1/protocol.json');assert high['sampling_budget']==262144 and len(high['cases'])==5
    timing=read(BASE/'highT_preparation_v1/H100_tile_benchmark_results.json');assert timing['tile_size']==65536 and len(timing['records'])==2
    metadata=read(BASE/'additional_parameter_packages_v2_audit.json')
    read('results/thrml_spw/lfm3_1p7b/language_resource_model/independent_audit.json')
    read('results/thrml_spw/lfm3_1p7b/published_hardware_context/independent_audit.json')
    matched=read('publish/pbit1/benchmarks/matched_resources_20261105.audit.json')
    read(BASE/'statistics_v3_preflight/results.json');read(BASE/'cloud_recovery_preflight_v1/recovery20261105a/results.json')
    live=[];studies={}
    for folder in ['r1_execution_v1','highT_execution_v1']:
        s=json.loads((BASE/folder/'observer/state.json').read_text())
        studies[folder]={'state_label':s['status'],'audited_cases':len([k for k in s['completed']if not k.startswith('statistics')])}
        for name,cid in s['calls'].items():
            if name in s['completed']:continue
            try:r=modal.FunctionCall.from_id(cid).get(timeout=0)
            except modal.exception.FunctionTimeoutError as error:
                live.append({'study':folder,'case':name,'call_id':cid,'observed_status':'terminal_function_timeout','message':str(error)})
            except (TimeoutError,modal.exception.TimeoutError):
                live.append({'study':folder,'case':name,'call_id':cid,'observed_status':'existing_call_live_or_waiting'})
            else:live.append({'study':folder,'case':name,'call_id':cid,'observed_status':'returned','result_status':r.get('status')})
    cid='fc-01M6PST9TAEYR7RM5T4H4C69Q3'
    try:r=modal.FunctionCall.from_id(cid).get(timeout=0)
    except modal.exception.FunctionTimeoutError as error:live.append({'study':'LFM3','call_id':cid,'observed_status':'terminal_function_timeout','message':str(error)})
    except (TimeoutError,modal.exception.TimeoutError):live.append({'study':'LFM3','call_id':cid,'observed_status':'existing_call_live_or_waiting'})
    else:live.append({'study':'LFM3','call_id':cid,'observed_status':'returned','result_status':r.get('status')})
    rows=[
        {'requirement':'Preserve accepted evidence, failures and fixed protocols','assessment':'verified_provenance_controls_within_recorded_scope','evidence':['r1_execution_v1/remote_primary_audit.json','highT_execution_v1/protocol.json','cloud_recovery_preflight_v1/recovery20261105a/results.json'],'boundary':'Original manifests/failednamespaces retained; new append-only events and stable remote handles. Final campaign archive still pending.'},
        {'requirement':'Repair frontend and account for previously exposed holdout data','assessment':'validated_execution_gates_and_explicit_exposure_ledger','evidence':['prior_holdout_exposure.json','statistics_v3_preflight/results.json'],'boundary':'512previouslyexposedtargets separated from9728unexposedtargets. Live full native execution is beyond repaired frontend; complete fresh-data quality remains pending.'},
        {'requirement':'Complete LFM3 five-seed full10240target/40article study','assessment':'incomplete_live_study','evidence':['holdout_v5_monitor/state.json'],'boundary':'Need all5independentcaseaudits and complete short/long/pooled full40 and unexposed38article MC/data-bootstrap gates. Firstcase still live; no fresh-data success conclusion.'},
        {'requirement':'Additional architectures/tasks with predefined quality criteria','assessment':'complete_primary_with_mixed_quality_outcomes','evidence':['r1_execution_v1/remote_primary.json','r1_execution_v1/remote_primary_audit.json'],'boundary':'DistilBERT/IMDb passes2ppmargin; GPT2/WikiText2 misses2percentPPLmargin atT8192. Alllearnedparameters converted, originalandfinitemeancontrols audited. Combinedprimary failed.'},
        {'requirement':'Conventional quantization and matched-resource multi-pbit comparisons','assessment':'legacy45case_comparison_complete_additional104case_study_incomplete','evidence':['publish/pbit1/benchmarks/matched_resources_20261105.audit.json','r1_execution_v1/protocol.json'],'boundary':'Internal K4/K8 controls, not a proprietaryExtropicrepresentation. Additionalfull104case statistics must complete before final newmodelPareto conclusions.'},
        {'requirement':'Verify whether more samples close GPT2gap','assessment':'distinct_frozen_followup_running_not_a_replacement_primary','evidence':['highT_execution_v1/protocol.json','highT_preparation_v1/model_preflight/old_new_bridge_audit.json'],'boundary':'5newMCkeys,T262144,32timesoriginalwork,samealreadyexposedtext; no result yet. All5caseandstatisticsaudits required.'},
        {'requirement':'Probability-control precision and metadata accounting','assessment':'bounded_controller_and_parameter_program_evidence_available','evidence':['additional_parameter_packages_v2_audit.json','nonideality_evidence_v1/results.json'],'boundary':'12controlbits/fullmodelscalar plus2FP32rangevalues/group; states/counters/readout additional. Component8bitcontrollerresults do not prove physical DAC resolution or general one-bit storage.'},
        {'requirement':'Temporal/cross-device correlation','assessment':'bounded_small_operator_characterization_complete','evidence':['correlation_independent_audit_v1/results.json'],'boundary':'108conditions/9smallIsingmodels,full spatial/temporal covariance. Does not show fullmodelrobustness or measure physicalcoupling.'},
        {'requirement':'Calibration error and drift','assessment':'bounded_native_studies_complete_with_explicit_negative_quality_result','evidence':['nonideality_evidence_v1/results.json'],'boundary':'SelectedquarterTransformer calibration improvesbutfails5percentqualitygate.150clockedoperatorqueries/10seeds characterize drift/refreshandcost. Fulladditionalmodelnonideal robustness and hardware lifetime unproven.'},
        {'requirement':'Separate measured simulator throughput from hardware projections','assessment':'primitive_timing_and_analytical_payload_available_final_campaign_timings_pending','evidence':['highT_preparation_v1/H100_tile_benchmark_results.json','results/thrml_spw/lfm3_1p7b/language_resource_model/independent_audit.json'],'boundary':'Count-only65536laneH100timing separate from neuralruntime/I/O. Packedpayload and componentarraybytes not totalmeasuredRAM or physicalarea. E_s,physicalf_s and full hardwareenergy/latency unmeasured.'},
        {'requirement':'Reproducible code, independent audits and published findings','assessment':'current_scoped_results_published_final_synthesis_pending','evidence':['r1_execution_v1/remote_primary_audit.json','highT_execution_v1/protocol.json'],'boundary':'PublicinstallableApache2SDK and aggregateevidence; detailedarchiveprivate. Completedrunningstudies require final immutablearchive, report and fullmatchedqualitysummary.'},
        {'requirement':'Physical hardware validation','assessment':'external_dependency_not_available_in_this_simulator_study','evidence':[],'boundary':'No physicalX0/XTR0deviceaccessormeasuredenergy/latency/calibration/correlations. No vendorinferiority or numericalBF16equivalence claim; this is not a blocker to remaining simulator work.'},
    ]
    assert all(sha(path)==h for path,h in evidence.items())
    result={'status':'point_in_time_full_goal_scope_and_evidence_gaps_reviewed','source_sha256':sha(__file__),'observed_unix':time.time(),
        'goal_complete':False,'requirements':rows,'closed_evidence_sha256':evidence,'studies':studies,'authoritative_call_observations':live,
        'next_required_results':['All5LFM3full/unexposedshort/long/pooledaudits','Complete104caseR1matchedresourceanalysis','Complete5seedhighTfollowupandindependentstatistics','Finalscopedquality/costParetoandpublication'],
        'scope':'A requirement/evidence gap inventory, not a newexperiment, formalcompletionoracle or automatic proof from passingflags. Exact numerical/calibration/drift checks are in separatelyboundaudits and exporter. Allpendingcampaignsmustfinish; physicalclaims require actualhardware.'}
    out=BASE/'goal_evidence_review_v1';out.mkdir(exist_ok=False);(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'goal_complete':False,'requirements':len(rows),'closed_files_bound':len(evidence),'live_call_observations':len(live),'studies':studies}))

if __name__=='__main__':main()
