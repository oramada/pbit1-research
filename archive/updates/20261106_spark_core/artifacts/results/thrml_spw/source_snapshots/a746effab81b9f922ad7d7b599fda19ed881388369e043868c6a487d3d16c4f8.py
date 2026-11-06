"""Label-free ideal IID state-change accounting, distinct from hardware energy."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
PARENT = ROOT/'representation_moment_evidence'
OUT = ROOT/'nominal_state_activity_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    assert not OUT.exists()
    closed = json.loads((PARENT/'completion.json').read_text())
    assert closed['passed'] and closed['weight_conditions_checked'] == 19640
    for path, expected in closed['artifact_sha256'].items():
        assert sha(path) == expected, path
    source_protocol = json.loads((PARENT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, expected in source_protocol[group].items():
            assert sha(path) == expected, path
    OUT.mkdir()
    protocol = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_sha256': {str(Path(__file__)): sha(__file__)},
                'input_sha256': {str(PARENT/'completion.json'): sha(PARENT/'completion.json'),
                                 str(PARENT/'arrays.npz'): sha(PARENT/'arrays.npz')},
                'scope': 'Nominal F64probability law of programmedF32fields; ideal independent stationary observations. Uses only all5frozen RGB model probabilities, no labels/native draws. Expected state changes are not measured physical flips/energy or native F32empirical moments. All3fixed matched budgets retained.'}
    write(OUT/'protocol.json', protocol)
    arrays = np.load(PARENT/'arrays.npz')
    rows, enumerated = [], 0
    for mi in range(5):
        ps = np.concatenate([arrays[f'model{mi}_layer{li}_single_probabilities'].ravel() for li in range(3)])
        pw = np.concatenate([arrays[f'model{mi}_layer{li}_word_probabilities'].ravel() for li in range(3)])
        assert len(ps) == 3928 and len(pw) == 15712
        quantities = []
        for probabilities in [ps, pw]:
            per_device = 2*probabilities*(1-probabilities)
            independent = []
            for p in probabilities:
                states = [(previous, current, (p if previous else 1-p)*(p if current else 1-p))
                          for previous in [0, 1] for current in [0, 1]]
                assert abs(math.fsum(float(mass) for _, _, mass in states)-1) < 2e-15
                independent.append(math.fsum(float(mass) for previous, current, mass in states if previous != current))
                enumerated += 1
            np.testing.assert_allclose(independent, per_device, rtol=2e-14, atol=2e-16)
            quantities.append({'stationary_changes_per_update_round': math.fsum(independent),
                               'initial_zero_warmup_changes': math.fsum(float(p) for p in probabilities)})
        for single_t, word_t in [(64, 16), (256, 64), (1024, 256)]:
            one, word = quantities
            u1, u4 = 3928*single_t, 15712*word_t
            assert u1 == u4
            f1 = one['stationary_changes_per_update_round']*single_t
            f4 = word['stationary_changes_per_update_round']*word_t
            rows.append({'model': mi, 'single_T': single_t, 'word_T': word_t,
                'retained_scheduled_updates_each_per8imagebank': u1,
                'single_expected_retained_update_state_changes': f1,
                'word_expected_retained_update_state_changes': f4,
                'single_over_word_expected_retained_changes': f1/f4,
                'single_expected_initial_zero_warmup_changes': one['initial_zero_warmup_changes'],
                'word_expected_initial_zero_warmup_changes': word['initial_zero_warmup_changes'],
                'single_expected_changes_between_retained_samples': one['stationary_changes_per_update_round']*(single_t-1),
                'word_expected_changes_between_retained_samples': word['stationary_changes_per_update_round']*(word_t-1)})
    assert enumerated == 98200 and len(rows) == 15
    write(OUT/'activity.json', {'rows': rows, 'device_conditions_four_state_enumerated': enumerated, 'scope': protocol['scope']})
    report = ROOT.parent/'CIFAR_CNN_NOMINAL_ACTIVITY_REPORT.md'
    assert not report.exists()
    lines = ['# Nominal state-change accounting at matched RGB update budgets', '',
        'For independent stationary Bernoulli observations, the probability of a state change in one update is2p(1-p). All98200device conditions (five models×3928single nodes plus five×15712word nodes) are checked by separately enumerating the four previous/current-state possibilities. Probabilities are the already verified nominal F64laws of programmedF32fields. No native trajectories, labels or physical measurements are added.', '',
        '| Single T | Word T | Scheduled retained updates each / bank | Expected single changes / bank | Expected word changes / bank | Single / word changes |', '|---:|---:|---:|---:|---:|---:|']
    for ts, tw in [(64, 16), (256, 64), (1024, 256)]:
        r = [v for v in rows if v['single_T'] == ts]
        one = np.mean([v['single_expected_retained_update_state_changes'] for v in r])
        word = np.mean([v['word_expected_retained_update_state_changes'] for v in r])
        lines.append(f'|{ts}|{tw}|{3928*ts}|{one:.6f}|{word:.6f}|{one/word:.6f}|')
    lines += ['', 'The internal word comparator programs every node near a deterministic endpoint (finitefields±4). Single-node variable probabilities generally fluctuate more. Equal retained scheduled updates therefore do not imply equal expected state changes. This is a property of these internal ideal representations, not an assertion about Extropic hardware weight encoding or measured device switching.', '',
        'The table counts Tretained updates after the warmup state is stationary, including the transition from warmup to the first retained state. Changes strictly between retained samples useT−1instead; both are retained in activity.json. Software initial all-zero-to-warmup changes are separately stored. Physical hardware may retain state across inputs and need not reset like this simulator.', '',
        'A conditional cost model may distinguish U scheduled updates and Fstate changes, e.g. U*E_update+F*E_change plus programming/holding/readout/accumulation/calibration. These coefficients and even this additive model require device validation. State-change counts alone are not energy estimates; sampling may dissipate energy without a visible bit flip. No energy/latency/density winner or quality/sample-saving claim follows.', '',
        '[All15model/budget rows](cifar_cnn_native/nominal_state_activity_evidence/activity.json) · [Protocol and assumptions](cifar_cnn_native/nominal_state_activity_evidence/protocol.json) · [Prior scalar moment analysis](CIFAR_CNN_REPRESENTATION_MOMENTS_REPORT.md). The frozen quality miss and all native/public quality evidence remain unchanged.']
    report.write_text('\n'.join(lines)+'\n')
    paths = [OUT/'protocol.json', OUT/'activity.json', report]
    write(OUT/'completion.json', {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'device_conditions_four_state_enumerated': enumerated, 'native_calls': 0, 'task_metrics': 0,
          'artifact_sha256': {str(path): sha(path) for path in paths}})
    print('All98200nominaldevice conditions/all15matchedbudget rows checked; no physical energy claim', flush=True)


if __name__ == '__main__':
    main()
