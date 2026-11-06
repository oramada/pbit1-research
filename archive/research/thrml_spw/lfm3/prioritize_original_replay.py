"""Bounded, reversible scheduling hold of the secondary replay frontend only."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from .adaptation_native_v2_pipeline import identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--secondary-pid', required=True, type=int)
    parser.add_argument('--original-pid', required=True, type=int)
    parser.add_argument('--seconds', required=True, type=int)
    args = parser.parse_args()
    assert 1 <= args.seconds <= 180
    root = Path('results/thrml_spw/lfm3_1p7b')
    expected = identity(args.secondary_pid)
    original = identity(args.original_pid)
    assert expected and expected['state'] == 'S'
    assert 'research.thrml_spw.lfm3.batched_native_replay' in expected['argv']
    assert original and original['state'] != 'Z'
    assert 'research.thrml_spw.lfm3.adaptation_native_v2_pipeline' in original['argv']
    assert (Path('/proc') / str(args.secondary_pid) / 'wchan').read_text().strip() == 'anon_pipe_read'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = root / 'batched_native_replay' / ('scheduling_priority_' + stamp + '.json')
    report = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'secondary_pid': args.secondary_pid, 'secondary_identity': expected,
        'original_pipeline_pid': args.original_pid, 'original_pipeline_identity': original,
        'maximum_hold_seconds': args.seconds, 'status': 'prepared',
        'scope': 'Operational scheduling only, no scientific-source/data/seed/sample-budget mutation or restart. Stop only hostfrontend whileitsworkerfinishescurrentrequest; boundedautomaticCONT evenifupstreamfails. No goalpause or performanceclaim.'}
    with path.open('x') as f: json.dump(report, f, indent=2)
    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')
    def same(actual, prior):
        return actual is not None and actual['start_ticks'] == prior['start_ticks'] and actual['argv'] == prior['argv']
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    start = time.monotonic(); stopped = False
    try:
        assert same(identity(args.secondary_pid), expected)
        os.kill(args.secondary_pid, signal.SIGSTOP); stopped = True
        report.update(status='secondary_frontend_temporarily_held', started_unix=time.time()); save()
        print(json.dumps({'status': report['status'], 'receipt': str(path)}), flush=True)
        while time.monotonic() - start < args.seconds:
            actual = identity(args.secondary_pid)
            assert same(actual, expected) and actual['state'] != 'Z'
            upstream = identity(args.original_pid)
            if not same(upstream, original) or upstream['state'] == 'Z':
                report['resume_reason'] = 'original_pipeline_terminal'; break
            if (root / 'adaptation_native_v2/initial_replay/independent_audit.json').exists():
                report['resume_reason'] = 'original_replay_audit_available'; break
            report.update(last_verified_unix=time.time(), observed_secondary_state=actual['state']); save()
            time.sleep(min(15, max(.01, args.seconds - (time.monotonic() - start))))
    finally:
        actual = identity(args.secondary_pid)
        if stopped and same(actual, expected) and actual['state'] != 'Z':
            os.kill(args.secondary_pid, signal.SIGCONT)
            report.update(status='secondary_frontend_resumed', resumed_unix=time.time(), hold_elapsed_seconds=time.monotonic() - start)
            report.setdefault('resume_reason', 'bounded_deadline_or_coordinator_exit')
        else:
            report.update(status='secondary_terminal_or_identity_changed_no_signal_sent')
        save(); print(json.dumps({'status': report['status'], 'receipt': str(path)}), flush=True)


if __name__ == '__main__':
    main()
