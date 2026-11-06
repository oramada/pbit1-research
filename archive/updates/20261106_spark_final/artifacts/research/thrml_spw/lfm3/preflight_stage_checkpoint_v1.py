"""Real-process interruption fixtures with official THRML, no text or models."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .stage_checkpoint_v1 import StageStore, CheckpointIntegrityError, canonical, sha

CONTRACT = {'fixture': 'official_unary_probability_counts', 'seed': 202611901,
    'samples': 512, 'logical_pbits': 65, 'stages': 2}


def worker(root, interruption):
    from research.thrml_spw.native import NativeProbabilityBank
    store = StageStore(root, CONTRACT)
    bank = NativeProbabilityBank(np.linspace(.01, .99, 65))
    for batch in range(2):
        if store.load(batch, 'native') is not None:
            continue
        attempt = store.new_attempt()
        with (store.root/'sampling_events.jsonl').open('a') as f:
            f.write(json.dumps({'batch': batch, 'seed': CONTRACT['seed']+batch})+'\n')
            f.flush(); os.fsync(f.fileno())
        values = bank.sample(CONTRACT['seed']+batch, 512, 1).mean(axis=(0, 1))
        path = attempt/f'batch{batch}.npy'
        with path.open('xb') as f: np.save(f, values)
        if batch == 0 and interruption == 'before_commit': os._exit(74)
        record = {'output': {'path': str(path.relative_to(store.root)), 'sha256': sha(path)},
            'metrics': {'mean': float(values.mean())}, 'module_calls': ['synthetic_unary_bank'],
            'operator_calls': [{'logical_pbits': 65, 'samples': 512}]}
        store.commit(batch, 'native', record)
        if batch == 0 and interruption == 'after_commit': os._exit(73)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--root', type=Path)
    p.add_argument('--interrupt', choices=['none','before_commit','after_commit'], default='none')
    args = p.parse_args()
    if args.worker:
        worker(args.root, args.interrupt)
        return
    out = Path('results/thrml_spw/broad_verification_20261105/stage_checkpoint_v1_preflight')
    out.mkdir(exist_ok=False)
    def run(path, interruption, expected):
        r = subprocess.run([sys.executable, '-m', 'research.thrml_spw.lfm3.preflight_stage_checkpoint_v1',
            '--worker', '--root', str(path), '--interrupt', interruption], capture_output=True, text=True)
        assert r.returncode == expected, r.stderr[-2000:]
    run(out/'uninterrupted', 'none', 0)
    run(out/'after_commit', 'after_commit', 73)
    run(out/'after_commit', 'none', 0)
    run(out/'before_commit', 'before_commit', 74)
    run(out/'before_commit', 'none', 0)
    reference = StageStore(out/'uninterrupted', CONTRACT)
    for name in ['after_commit','before_commit']:
        actual = StageStore(out/name, CONTRACT)
        for batch in range(2):
            a, b = actual.load(batch,'native'), reference.load(batch,'native')
            np.testing.assert_array_equal(np.load(actual.root/a['output']['path']), np.load(reference.root/b['output']['path']))
    def events(name): return [json.loads(line) for line in (out/name/'sampling_events.jsonl').read_text().splitlines()]
    assert [x['batch'] for x in events('after_commit')] == [0,1]
    assert [x['batch'] for x in events('before_commit')] == [0,0,1]
    assert len(list((out/'before_commit/attempts').glob('*/batch0.npy'))) == 2
    rejected=[]
    for corruption in ['contract','payload','truncated_metadata','metadata_digest','path_escape']:
        target=out/('corrupt_'+corruption)
        shutil.copytree(out/'after_commit',target)
        try:
            if corruption=='contract':
                StageStore(target,{**CONTRACT,'seed':1})
            else:
                store=StageStore(target,CONTRACT);path=store.root/'stage_0_native.json';record=json.loads(path.read_text())
                if corruption=='payload': (store.root/record['record']['output']['path']).write_bytes(b'corrupted')
                elif corruption=='truncated_metadata':path.write_text('{')
                elif corruption=='metadata_digest':
                    record['record']['metrics']['mean']=999;path.write_text(json.dumps(record))
                else:
                    record['record']['output']['path']='../escape.npy'
                    record['record_sha256']=hashlib.sha256(canonical(record['record'])).hexdigest()
                    path.write_text(json.dumps(record))
                store.load(0,'native')
        except CheckpointIntegrityError: rejected.append(corruption)
        else: raise AssertionError('Corruption accepted: '+corruption)
    accepted=StageStore(out/'after_commit',CONTRACT)
    h=sha(accepted.root/'stage_0_native.json')
    try:accepted.commit(0,'native',accepted.load(0,'native'))
    except FileExistsError:pass
    else:raise AssertionError('Completed stage overwritten')
    assert sha(accepted.root/'stage_0_native.json')==h
    result={'status':'real_process_interruptions_stage_skip_same_key_replay_and_corruption_rejection_passed',
        'sources_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['preflight_stage_checkpoint_v1.py','stage_checkpoint_v1.py']},
        'auxiliary_sources_sha256':{'research/thrml_spw/native.py':sha(Path('research/thrml_spw/native.py'))},
        'contract':CONTRACT,'after_committed_stage_sampling_events':events('after_commit'),
        'before_commit_sampling_events_including_charged_replay':events('before_commit'),
        'corruptions_rejected':rejected,'orphan_attempt_payloads_preserved':True,
        'scope':'Actual subprocess exit/restart and officialTHRML deterministic replay on local disk. No Modal distributed locking/commit proof, model inference, new neural MC trial, or fresh-data outcome.'}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
