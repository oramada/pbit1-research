"""Freeze distinct post-primary high-T follow-up only after all execution gates."""
from pathlib import Path
import ast,copy,hashlib,json,math,datetime
from research.thrml_spw.broad.highT_statistics_v1 import cohort

BASE=Path('results/thrml_spw/broad_verification_20261105')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    candidate=json.loads((BASE/'highT_preparation_v1/candidate.json').read_text())
    assert candidate['status']=='candidate_pending_execution_and_statistical_gates_not_launched'
    assert candidate['samples_per_parameter']==262144 and candidate['seeds']==list(range(202611800,202611805))
    parent_path=BASE/'r1_execution_v1/protocol.json';parent=json.loads(parent_path.read_text())
    primary_path=BASE/'r1_execution_v1/remote_primary.json';primary=json.loads(primary_path.read_text())
    audit=json.loads((BASE/'r1_execution_v1/remote_primary_audit.json').read_text());assert audit['statistics_sha256']==sha(primary_path)
    assert not primary['primary_both_models_pass'] and not primary['models']['gpt2']['primary_quality_gate']
    gates=['highT_preparation_v1/CPU_H100_bridge_audit.json','highT_preparation_v1/model_preflight/old_new_bridge_audit.json',
        'highT_preparation_v1/model_preflight/remote_audit.json','highT_preparation_v1/statistics_preflight.json',
        'highT_preparation_v1/H100_tile_benchmark_results.json','primary_count_moments_v1_results.json']
    bridge,model_bridge,model_audit,stat,timing,moments=[json.loads((BASE/x).read_text())for x in gates]
    assert bridge['total_bits_checked_per_backend']==6488064 and bridge['uint16_overflow_detected_and_uint32_roundtrip_passed']
    assert model_bridge['unique_parameters']==124439808 and model_bridge['canonical_parameters']==148
    assert model_audit['synthetic_only'] and model_audit['source_sha256']==sha('research/thrml_spw/broad/verify_highT_case_v1.py')
    assert sha(BASE/'highT_preparation_v1/model_preflight/remote_audit.json')==model_bridge['new_independent_replay_sha256']
    assert stat['status']=='highT_complete_binding_success_failure_missing_cohort_and_forgery_fixtures_passed' and len(stat['checks'])==9
    assert stat['producer_sha256']==sha('research/thrml_spw/broad/highT_statistics_v1.py')
    assert stat['verifier_sha256']==sha('research/thrml_spw/broad/verify_highT_statistics_v1.py')
    assert moments['all_aggregate_moment_gates_met'] and timing['tile_size']==65536
    assert timing['records'][1]['samples']==262144 and timing['records'][1]['median_warm_seconds']<10
    sources=copy.deepcopy(parent['sources_sha256'])
    new_sources=['research/thrml_spw/broad/'+x for x in ['highT_case_v1.py','highT_worker_v1.py','verify_highT_case_v1.py',
        'highT_statistics_v1.py','verify_highT_statistics_v1.py','preflight_highT_statistics_v1.py']]
    new_sources+=['cloud/highT_execution_v1/'+x for x in ['layout.py','app.py','client.py','freeze.py']]
    for name in new_sources:
        ast.parse(Path(name).read_text());h=sha(name);sources[name]=h
        dest=Path('results/thrml_spw/source_snapshots')/(h+'.py')
        if dest.exists():assert sha(dest)==h
        else:dest.write_bytes(Path(name).read_bytes())
    assert all(sha(path)==h for path,h in sources.items())
    inputs=copy.deepcopy(parent['inputs_sha256'])
    for path in [parent_path,primary_path,BASE/'r1_execution_v1/remote_primary_audit.json',BASE/'highT_preparation_v1/candidate.json',*[BASE/x for x in gates]]:inputs[str(path)]=sha(path)
    controls={path:h for path,h in primary['inputs_sha256'].items()if '/cases/gpt2__'in path and path.split('/cases/gpt2__')[1].split('/')[0]in ['original','mean1','mean4','mean8','int2','int4','int8']}
    assert len(controls)==14
    spec=copy.deepcopy(parent['models']['gpt2']);spec['seeds_by_budget']={'262144':candidate['seeds']};spec['bootstrap_key']=202611810
    assert spec['margin']==math.log(1.02)
    root=BASE/'highT_execution_v1';root.mkdir(exist_ok=False)
    cases={f'gpt2__K1_C262144_seed{s}':{'model_key':'gpt2','mode':'native','bits':1,'budget':262144,'samples':262144,'seed':s}for s in candidate['seeds']}
    p={'status':'complete_R1_execution_and_statistics_frozen_before_scoring','study_kind':'distinct_post_primary_highT_followup',
        'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output':str(root),'models':{'gpt2':spec},
        'cases':cases,'case_order':list(cases),'sources_sha256':sources,'inputs_sha256':inputs,
        'control_protocol':str(parent_path),'control_protocol_sha256':sha(parent_path),'control_inputs_sha256':controls,
        'sampling_budget':262144,'logical_pbits_per_scalar':1,
        'cost_scope':'32timesoriginal8192logicalretainedwork per reconstructedmodeldraw, samecellcount.65536lanewarmcountingprojection~51minutesperdraw,notfullruntime/hardwarelatency.',
        'decision':'Both originalBF16 and finite-programmean1 contrasts must satisfy unchangedlog1.02margin with MC95t4 and crossedseed/articlebootstrap95 upperbounds, allfivefixedseeds. No outcome-based cohort edits.',
        'data_scope':'Same previouslyexposed283225WikiText2targets/62articles atcontext1024stride512. New MCkeys, not new-textholdout. No training or test-based parameter/range updates.',
        'original_primary_preservation':{'statistics_sha256':sha(primary_path),'gpt2_passed':False,'joint_primary_passed':False},
        'failure_policy':'All5cases, failures, counts and runtimeattempts retained. No stochastic redraw after qualityinspection, no relaxation ofmargin, no renamingoriginalprimaryaspassed.',
        'scope':'One logicalpbitperlearnedscalar, includingallbias/normparameters andtiedparameteronce. R1reconstructionreusedacrossfulldataset, five independentdraws separatelyevaluated. No R2, numericalBF16equivalence or physicaldensity/energy/latencyclaim.'}
    assert cohort(p)==p['case_order']
    with (root/'protocol.json').open('x')as f:json.dump(p,f,indent=2);f.write('\n')
    print(json.dumps({'status':'complete_highT_followup_frozen_before_new_task_scoring','protocol_sha256':sha(root/'protocol.json'),'cases':5,'samples':262144}))

if __name__=='__main__':main()
