"""Transport-only batch assignment; original scientific contract stays fixed."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path('results/thrml_spw/lfm3_1p7b')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for data in iter(lambda:f.read(16*1024*1024),b''):h.update(data)
    return h.hexdigest()


def validate_subset(plan,batches,records):
    if not batches or len(batches)!=len(set(batches))or any(type(i)is not int or not 0<=i<16 for i in batches):
        raise ValueError('Unique frozen global batch indices required')
    if len(records)!=len(batches):raise ValueError('Missing or extra shard batch')
    seen=set()
    for r in records:
        i=r['batch_index']
        if i not in batches or i in seen:raise ValueError('Wrong or duplicate shard batch')
        seen.add(i);x=np.asarray(r['token_NLL'],np.float64)
        if list(x.shape)!=plan[i]['loss_shape']or not np.isfinite(x).all():raise ValueError('Shard loss geometry differs')
    if seen!=set(batches):raise ValueError('Incomplete shard')


def validate_transport(path,core_path,seed,shard):
    p=json.loads(Path(path).read_text());core=json.loads(Path(core_path).read_text())
    if p['scientific_protocol_sha256']!=sha(core_path):raise ValueError('Original scientific protocol changed')
    if p['parent_case_sha256']!=sha(Path(__file__).with_name('holdout_native_case_v4.py')):raise ValueError('Original producer changed')
    for name,h in p['sources_sha256'].items():
        if sha(name)!=h:raise ValueError('Transport source changed: '+name)
    if p['mode']=='same_key_closed_batch_replay_proof':
        if seed!=core['seeds'][0]or shard!='proof4' or p['assignments']!={'proof4':[4]}:raise ValueError('Only closed nonzero batch4 proof authorized')
        for name,h in p['closed_reference_checkpoints_sha256'].items():
            if sha(name)!=h:raise ValueError('Closed proof reference changed')
        expected=[ROOT/'holdout_native_v4'/str(seed)/f'stage_4_{s}.json'for s in ['BF16','same_fit_BF16','nominal_controls','native']]
        if set(p['closed_reference_checkpoints_sha256'])!=set(map(str,expected)):raise ValueError('Complete4stage reference required')
    elif p['mode']=='batch_parallel_remaining_four_seeds_after_first_case_audit':
        if seed not in core['seeds'][1:]or shard not in ['0','1','2','3']:raise ValueError('Only original remainingfourkeys/shards authorized')
        if p['assignments']!={str(i):list(range(i,16,4))for i in range(4)}:raise ValueError('Frozen roundrobin coverage changed')
        proof=Path(p['replay_proof']);a=json.loads(proof.read_text())
        if sha(proof)!=p['replay_proof_sha256']or a['status']!='closed_nonzero_batch_all4raw_stage_outputs_and_receipts_bitwise_replayed':raise ValueError('Exact replay proof required')
        first=ROOT/'holdout_native_v4'/str(core['seeds'][0]);audit=json.loads((first/'independent_audit.json').read_text())
        if audit['status']!='full_reserved_case_raw_losses_controls_coverage_and_resources_independently_verified':raise ValueError('First complete independentcaseaudit required')
        if audit['results_sha256']!=sha(first/'results.json')or audit['shared_protocol_sha256']!=sha(core_path):raise ValueError('First case binding differs')
        if audit['source_sha256']!=core['independent_case_verifier_sha256']:raise ValueError('First case auditor differs')
    else:raise ValueError('Unknown transport mode')
    if p['status']!='transport_frozen_before_its_execution':raise ValueError('Frozen transport required')
    batches=p['assignments'][shard]
    expected_root=ROOT/('holdout_sharding_proof_v1'if p['mode']=='same_key_closed_batch_replay_proof'else'holdout_batch_shards_v1')
    if p['output_root']!=str(expected_root):raise ValueError('Unexpected shard namespace')
    return {'transport_sha256':sha(path),'mode':p['mode'],'shard':shard,'batches':batches,'output_root':p['output_root']}
