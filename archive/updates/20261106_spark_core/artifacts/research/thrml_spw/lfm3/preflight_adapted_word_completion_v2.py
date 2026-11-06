"""Completion-receipt negative controls; does not launch a sampler or model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from .adapted_word_validation_pipeline_v2 import validate_completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for name in ['adapted_word_validation_pipeline_v2.py', 'preflight_adapted_word_completion_v2.py']:
        p = Path(__file__).with_name(name)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert (Path('results/thrml_spw/source_snapshots') / (digest + '.py')).read_bytes() == p.read_bytes()
        sources[name] = digest
    state = {'status': 'complete', 'protocol_sha256': 'p', 'statistics_audit_sha256': 'a'}
    audit = {'status': 'verified', 'source_sha256': 'v', 'statistics_sha256': 's'}
    assert validate_completion(state, audit, 's', 'a', 'complete', 'verified', 'p', 'v')
    rejected = 0
    for field, in_state in [('status', True), ('protocol_sha256', True), ('statistics_audit_sha256', True),
                            ('status', False), ('source_sha256', False), ('statistics_sha256', False)]:
        s, a = copy.deepcopy(state), copy.deepcopy(audit)
        (s if in_state else a)[field] = 'changed'
        try:
            validate_completion(s, a, 's', 'a', 'complete', 'verified', 'p', 'v')
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('Incomplete/forged completion accepted')
    assert rejected == 6
    with args.output.open('x') as handle:
        json.dump({'status': 'valid_completion_and_six_incomplete_or_forged_receipts_verified',
                   'sources_sha256': sources, 'modified_receipts_rejected': rejected,
                   'scope': 'Pure scheduling-receipt fixture only; no model result or confidence-coverage claim.'}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
