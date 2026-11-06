"""Freeze transport/checkpoint recovery, preserving all prospective science."""
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from research.thrml_spw.lfm3.holdout_native_case_v4 import validate_contract

ROOT=Path('results/thrml_spw/lfm3_1p7b')
BROAD=Path('results/thrml_spw/broad_verification_20261105')
SOURCE=Path('research/thrml_spw/lfm3')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    old_path=ROOT/'holdout_native_v2_protocol.json';old=json.loads(old_path.read_text())
    original_analysis=ROOT/'holdout_native_statistics_v2_protocol.json'
    old_analysis=json.loads(original_analysis.read_text())
    fixture=BROAD/'statistics_v4_preflight/results.json';tested=json.loads(fixture.read_text())
    cloud_fixture=BROAD/'cloud_recovery_preflight_v1/recovery20261105a/results.json'
    assert json.loads(cloud_fixture.read_text())['status']=='deployed_durable_calls_atomic_duplicate_claim_and_cross_container_recovery_passed'
    assert tested['status']=='full40_and_unexposed38_analysis_and_independent_auditor_fixtures_passed'
    assert tested['parameters']=={**old_analysis['parameters'],'control_loss_absolute_tolerance':1e-12}
    analysis=copy.deepcopy(old_analysis)
    analysis.update(recorded_utc=datetime.now(timezone.utc).isoformat(),
        supersedes_transport_only_analysis_sha256=sha(original_analysis),
        parameters=tested['parameters'],preflight_sha256=sha(fixture),sources_sha256=tested['sources_sha256'])
    analysis['engineering_change']='Only cross-seed FP64 metric-reduction guard changes to1e-12absolute. Keep every seed loss, raw control tensor bit equality, all frozen2percentcriteria, bootstrap keys and exposure definition.'
    analysis_path=ROOT/'holdout_native_statistics_v4_protocol.json';write(analysis_path,analysis)
    p=copy.deepcopy(old)
    p['recorded_utc']=datetime.now(timezone.utc).isoformat()
    p['statistics_protocol_sha256']=sha(analysis_path)
    p['sources_sha256'].update(tested['sources_sha256'])
    for name in ['holdout_native_case_v4.py','verify_holdout_native_case_v4.py','stage_checkpoint_v1.py','preflight_case_checkpoint_v4.py']:
        p['sources_sha256'][name]=sha(SOURCE/name)
    p['independent_case_verifier_sha256']=p['sources_sha256']['verify_holdout_native_case_v4.py']
    p['inputs_sha256']['holdout_native_statistics_v4_protocol.json']=sha(analysis_path)
    inventory=BROAD/'legacy_inventory_v3.json';legacy=json.loads(inventory.read_text())
    p['recovery']={'supersedes_interrupted_execution_protocol_sha256':sha(old_path),
        'legacy_outputs':{'202611150':{name:v['sha256'] for name,v in legacy['files'].items()}},
        'policy':'Same-key replay of legacy6closed outputs to regenerate missing receipts, exact raw-bit comparison, retained old files and separately charged replay. Completed v3stages skip only after bound payload/metadata validation. No extra MC seed or score selection.',
        'distributed_ownership':'Deployed durable FunctionCall IDs, atomic Dict.put(skip_if_exists=True), reattach original owner. Observer timeout never restarts.',
        'checkpoints':'Per closed stage, outputSHA, token losses, module coverage and operator receipts. Unique attempt logs, periodic20secondVolumecommit. Actual program failures terminal.',
        'cloud_recovery_fixture_sha256':sha(cloud_fixture),'legacy_inventory_sha256':sha(inventory),'preserved_v3_path_failure_sha256':sha(BROAD/'holdout_v3_failure/inventory.json')}
    extra=[fixture,cloud_fixture,inventory,BROAD/'preemption_event_v1.json',BROAD/'control_reduction_diagnostic_v1.json',
           Path('cloud/broad_validation_v4/app.py'),Path('cloud/broad_validation_v4/client.py'),Path('cloud/broad_validation_v4/freeze.py'),
           Path('cloud/broad_validation_v2/app.py'),BROAD/'case_checkpoint_v4_preflight/results.json',BROAD/'holdout_v3_failure/inventory.json']
    for f in extra:p['auxiliary_inputs_sha256'][str(f)]=sha(f)
    p['execution']={'gpu':'H100!','max_parallel_cases':4,'sampling_retries':0,'case_timeout_seconds':86400,
        'first_seed_audited_before_remaining_four':True,'deployed_app':'pbit1-broad-heldout-v3','observer':'local durable handles',
        'infrastructure_preemption':'Resume closed stages within same logical call. Partial stage same-key replay charged separately.'}
    p['scope']=old['scope']+' Transport recovery v4 preserves original seeds/data/model/criteria; original v2 artifacts remain immutable.'
    assert all(p[k]==old[k] for k in ['samples','pbits_per_parameter','arm','seeds','stages','targets','articles','field_bits','activation_bits','batch_plan','eligibility_policy_sha256'])
    selection={'status':'eligible_after_full_execution_protocol_freeze','pbits_per_parameter':1,'arm':'mean','samples':p['samples']}
    for seed in p['seeds']:validate_contract(p,selection,p['batch_plan'],seed)
    for name,d in p['sources_sha256'].items():assert sha(SOURCE/name)==d,name
    destination=ROOT/'holdout_native_v4_protocol.json';write(destination,p)
    # Existing inputs remain on the shared volume; include all small input files
    # so the deployed job rejects incomplete transfer before opening tokens.
    old_assets=json.loads((BROAD/'holdout_assets_v2.json').read_text())
    files=copy.deepcopy(old_assets['files'])
    for f in [destination,analysis_path,*extra]:files[str(f)]={'sha256':sha(f),'bytes':f.stat().st_size}
    asset=BROAD/'holdout_assets_v4.json';write(asset,{'status':'frozen_recovery_inputs_science_unchanged',
        'execution_protocol_sha256':sha(destination),'files':files})
    snapshot=Path('results/thrml_spw/source_snapshots');snapshot.mkdir(exist_ok=True)
    for name in p['sources_sha256']:
        f=SOURCE/name;d=sha(f);target=snapshot/(d+'.py')
        if not target.exists():target.write_bytes(f.read_bytes())
    for f in extra:
        if f.suffix=='.py':
            target=snapshot/(sha(f)+'.py')
            if not target.exists():target.write_bytes(f.read_bytes())
    print(json.dumps({'status':'recovery_v4_frozen','protocol_sha256':sha(destination),'sources':len(p['sources_sha256']),
        'seeds':p['seeds'],'samples':p['samples'],'asset_files':len(files)}))

if __name__=='__main__':main()
