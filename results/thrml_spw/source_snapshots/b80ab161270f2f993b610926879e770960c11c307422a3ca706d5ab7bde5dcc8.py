"""Sequential producers for the prospectively frozen probability adaptation pilot.

Producer completion is not independent audit acceptance. No automatic retry.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    protocol_path = root / 'adaptation_pilot_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert sha(Path(__file__)) == protocol['sources_sha256'][Path(__file__).name]
    out = root / 'adaptation_pilot'
    out.mkdir(exist_ok=True)
    path = out / 'pipeline.json'
    state = {'status': 'running', 'protocol_sha256': sha(protocol_path), 'produced_arms': [],
             'scope': 'Producer scheduling only; independent audit acceptance separate.'}
    with path.open('x') as f:
        json.dump(state, f, indent=2)
    for arm in protocol['producer_order']:
        log = (out / (arm + '.log')).open('x')
        process = subprocess.Popen([sys.executable, '-m', 'research.thrml_spw.lfm3.adaptation_pilot', '--arm', arm],
                                   stdout=log, stderr=log)
        state.update(current_arm=arm, child_pid=process.pid, child_started_unix=time.time())
        path.write_text(json.dumps(state, indent=2) + '\n')
        code = process.wait()
        log.close()
        state['child_exit_code'] = code
        if code:
            state['status'] = 'failed_preserved_no_restart'
            path.write_text(json.dumps(state, indent=2) + '\n')
            raise SystemExit(code)
        report = json.loads((out / arm / 'results.json').read_text())
        assert report['status'] == 'all_parameter_probability_adaptation_and_validation_surrogate_complete'
        state['produced_arms'].append(arm)
        path.write_text(json.dumps(state, indent=2) + '\n')
    state.update(status='all_three_producers_complete_pending_independent_audits', current_arm=None, child_pid=None)
    path.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
