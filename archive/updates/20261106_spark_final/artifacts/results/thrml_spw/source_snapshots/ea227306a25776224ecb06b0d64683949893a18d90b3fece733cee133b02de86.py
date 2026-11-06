"""Audit completed pilot producers; verify live PID identity while waiting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (proc / 'cmdline').read_bytes().split(b'\0')
        return {'start_ticks': fields[19], 'state': fields[0], 'argv': [v.decode() for v in argv if v]}
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--producer-pid', type=int, required=True)
    args = parser.parse_args()
    root = Path('results/thrml_spw/lfm3_1p7b')
    out = root / 'adaptation_pilot'
    producer = identity(args.producer_pid)
    assert producer is not None and 'research.thrml_spw.lfm3.adaptation_pilot_pipeline' in producer['argv']
    assert producer['state'] != 'Z'
    protocol = {'source_sha256': sha(Path(__file__)),
        'verifier_sha256': sha(Path(__file__).with_name('verify_adaptation_pilot.py')),
        'shared_protocol_sha256': sha(root / 'adaptation_pilot_protocol.json'),
        'producer_pid': args.producer_pid, 'producer_identity_at_start': producer,
        'scope': 'Wait on actual producer PID identity and completed-child receipts; independently replay each produced arm. No automatic retry or native quality inference.'}
    with (out / 'audit_pipeline_protocol.json').open('x') as f:
        json.dump(protocol, f, indent=2)
    state = {'status': 'waiting_on_verified_producer', 'pid': os.getpid(), 'audited_arms': []}
    def save():
        (out / 'audit_pipeline.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    while True:
        pipeline = None
        for _ in range(5):
            try:
                pipeline = json.loads((out / 'pipeline.json').read_text())
                break
            except json.JSONDecodeError:
                time.sleep(.1)
        assert pipeline is not None
        assert pipeline['protocol_sha256'] == protocol['shared_protocol_sha256']
        pending = [arm for arm in pipeline['produced_arms'] if arm not in state['audited_arms']]
        if pending:
            arm = pending[0]
            log = (out / (arm + '.audit.log')).open('x')
            process = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.verify_adaptation_pilot', '--arm', arm],
                                       stdout=log, stderr=log)
            state.update(status='auditing', current_arm=arm, child_pid=process.pid)
            save()
            code = process.wait()
            log.close()
            if code:
                state.update(status='audit_failed_preserved_no_restart', child_exit_code=code)
                save()
                raise SystemExit(code)
            audit = json.loads((out / arm / 'independent_audit.json').read_text())
            assert audit['status'] == 'all_coordinates_Adam_trajectory_programs_and_validation_losses_verified'
            state['audited_arms'].append(arm)
            state.update(current_arm=None, child_pid=None)
            save()
            continue
        if pipeline['status'] == 'all_three_producers_complete_pending_independent_audits':
            assert set(state['audited_arms']) == {'mean', 'binomial', 'baseline'}
            state.update(status='all_three_arms_independently_verified_pending_native_quality')
            save()
            break
        if pipeline['status'] == 'failed_preserved_no_restart':
            state.update(status='producer_failed_preserved_no_restart', producer_state=pipeline)
            save()
            raise SystemExit(1)
        current = identity(args.producer_pid)
        assert current is not None and current['start_ticks'] == producer['start_ticks']
        assert current['argv'] == producer['argv'] and current['state'] != 'Z'
        state.update(status='waiting_on_verified_producer', last_verified_producer_unix=time.time(),
                     producer_current_arm=pipeline.get('current_arm'))
        save()
        time.sleep(30)


if __name__ == '__main__':
    main()
