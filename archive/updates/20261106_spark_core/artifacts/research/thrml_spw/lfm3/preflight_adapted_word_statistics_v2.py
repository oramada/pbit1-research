"""Synthetic positive/negative controls for the paired statistics, no model scores."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from .adapted_word_statistics_v2 import analyze
from .verify_adapted_word_statistics_v2 import verify_math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for name in ['adapted_word_statistics_v2.py', 'verify_adapted_word_statistics_v2.py', 'preflight_adapted_word_statistics_v2.py']:
        p = Path(__file__).with_name(name)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert (Path('results/thrml_spw/source_snapshots') / (digest + '.py')).read_bytes() == p.read_bytes()
        sources[name] = digest
    critical = '2.77644510519779866907365875095'
    margin = 0.01980262729617973
    verified = rejected = 0
    for values in [[3.99] * 5, [4.1, 4.2, 4.3, 4.4, 4.5], [4.0, 4.001, 3.999, 4.002, 3.998]]:
        data = {'common_controls': {'BF16': 4.1, 'same_fit_BF16': 4.0},
                'methods': {f'K{k}': {'native_NLL': [v + delta for v in values], 'nominal_NLL': 3.99 + delta}
                            for k, delta in [(1, 0), (4, -.01), (8, -.02)]}}
        candidate = analyze(data, float(critical), margin)
        verify_math(candidate, data, critical, margin)
        verified += 1
        for change in ['gate', 'interval', 'contrast']:
            bad = copy.deepcopy(candidate)
            if change == 'gate':
                bad['methods']['K1']['conditional_BF16_gate_passed'] = not bad['methods']['K1']['conditional_BF16_gate_passed']
            elif change == 'interval':
                bad['methods']['K4']['native_NLL']['MC95_t4_interval'][1] += .5
            else:
                del bad['paired_NLL_contrasts']['K1_minus_K8']
            try:
                verify_math(bad, data, critical, margin)
            except AssertionError:
                rejected += 1
            else:
                raise AssertionError('Modified statistics were accepted')
    assert verified == 3 and rejected == 9
    with args.output.open('x') as handle:
        json.dump({'status': 'pass_fail_zero_variance_and_nine_tampered_statistics_cases_verified',
                   'sources_sha256': sources, 'valid_fixtures': verified, 'modified_reports_rejected': rejected,
                   'scope': 'Synthetic arithmetic/schema controls only, no model evidence or claimed confidence coverage.'}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
