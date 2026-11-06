"""Independent receipt/schema/range/code/resource review of learned-word validation."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.cifar_cnn_native.probability_cohort.common import (
    ROOT, checkpoint, sha, guard, seal)


def main():
    out = ROOT/'learned_word_witness_v2_evidence'
    assert not (out/'review_completion.json').exists()
    assert (ROOT.parent/'source_snapshots'/(sha(__file__)+'.py')).read_bytes() == Path(__file__).read_bytes()
    seal(out/'completion.json')
    protocol = json.loads((out/'protocol.json').read_text()); guard(protocol)
    audit = json.loads((out/'evidence_audit.json').read_text())
    assert audit['passed'] and audit['protocol_sha256'] == sha(out/'protocol.json')
    assert audit['labels_used'] is False and len(audit['rows']) == 40
    assert audit['counter_entries_exact'] == 3941280 and audit['bank_arrays_exact'] == 1120
    assert audit['native_updates'] == audit['public_updates'] == 427366400
    assert audit['warmup_updates_per_pass'] == 1256960
    expected = {(m, mi, t) for m in range(2) for mi in range(5) for t in [16, 64, 256, 1024]}
    seen = set(); entries = 0; arrays = 0; weight_coordinates = 0
    for method in range(2):
        for mi in range(5):
            model = checkpoint(mi, method)
            for prefix in ['conv1', 'conv2', 'head']:
                w = model[prefix+'_weight'].reshape(len(model[prefix+'_weight']), -1)
                c, r = model[prefix+'_center'], model[prefix+'_radius']
                assert np.all(r > 0)
                # Derive positional codes from exported means, independently of command-logistic replay.
                code = np.rint(15*np.minimum(1, np.maximum(0, ((w-c[:, None])/r[:, None]+1)/2))).astype(np.uint8)
                reconstructed_code = sum(((code >> bit) & 1)*(2**bit) for bit in range(4))
                np.testing.assert_array_equal(code, reconstructed_code)
                weight_coordinates += code.size
    for row in audit['rows']:
        item = row['method'], row['model'], row['word_T']
        assert item not in seen and item in expected
        seen.add(item)
        assert row['matched_single_T'] == 4*row['word_T'] and row['all14arrays_exact']
        path = Path(row['path']); assert sha(path) == row['sha256']
        with np.load(path) as raw:
            assert len(raw.files) == 14
            count = 0
            for li, ni in enumerate([27, 72, 256]):
                bound = 255*ni*15*row['word_T']
                assert bound < 2**31 and row['counter_bounds'][li] == bound
                gate, col = raw[f'gate_{li}'], raw[f'column_{li}']
                assert gate.dtype == col.dtype == np.int32
                assert gate.min() >= 0 and gate.max() <= bound
                assert col.min() >= 0 and col.max() <= ni*15*row['word_T']
                assert np.isfinite(raw[f'output_{li}']).all()
                count += gate.size+col.size
            assert raw['output_2'].shape == (2, 8, 10)
            assert count == row['counter_entries_exact'] == 98532
            entries += count; arrays += len(raw.files)*2
    assert seen == expected and entries == 3941280 and arrays == 1120 and weight_coordinates == 39280
    failure = ROOT/'execution_checks/learned_word_first_attempt_terminal_failure.json'
    prior = json.loads(failure.read_text())
    assert prior['first_attempt_terminal_returncode'] == 1
    assert prior['failure_log_sha256'] == sha(ROOT/'learned_word_witness_evidence/run.log')
    report = ROOT.parent/'CIFAR_CNN_LEARNED_WORD_VALIDATION_REPORT.md'; assert not report.exists()
    report.write_text('''# Same-fit four-p-bit RGB validation

All ten previously evaluated probability-finetuned fits pass native/public validation at word samples 16, 64, 256 and 1024, corresponding to single-p-bit samples 64, 256, 1024 and 4096 at equal retained stochastic-node evaluations. The original per-output centers/radii, deterministic biases and ACT8 spans are retained. Word codes round the learned stationary mean into 16 positional levels; all four nodes remain free with finite fields plus/minus4 and coefficients1,2,4,8 (denominator15). This is an internal research comparator, not an asserted Extropic hardware weight format.

All40cases use the first16validation images in two8imagebanks. All1120bank arrays and3941280integer counters match native/public-state execution exactly. Each pass charges427366400retained plus1256960warmup schedulednodeupdates. The largest signed counter bound is1002700800, below2^31. There are15712freeweightnodes versus3928for one-p-bit inference. Original biases, range metadata, activation discretization, counters and F64 readout remain separate resources. Warmup work is larger for the four-node baseline even when retained work is matched; no actual state-change/energy/latency/area improvement is claimed.

The first validation attempt stopped at an exact learned-mean export identity check because reciprocal multiplication differed from the export's division order by up to1.11e-16. Its frozen source/protocol/log are preserved. The corrected version uses the same export division order; no output or counter tolerance was relaxed. The failed attempt had502784retained plus31424warmup native scheduled updates and no public-state draws; it is separate from successful validation costs.

No labels, test accuracy, or model selection were used. This proves compatibility and causal execution for the matched-weight comparator; full matched-work task accuracy, statistical comparisons and nominal-word controls remain required before any representation advantage can be claimed. Both training methods and all lower-budget outcomes will remain included. The broader research goal remains active.

[Protocol](cifar_cnn_native/learned_word_witness_v2_evidence/protocol.json) · [Exact native/public validation](cifar_cnn_native/learned_word_witness_v2_evidence/evidence_audit.json) · [Preserved first failure](cifar_cnn_native/execution_checks/learned_word_first_attempt_terminal_failure.json)
''')
    review = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': sha(__file__), 'cases_checked': 40, 'weight_coordinates_code_checked': 39280,
        'counter_entries_schema_bounds_checked': entries, 'bank_arrays_schema_checked': arrays,
        'protocol_source_input_hashes_verified': True,
        'artifact_sha256': {str(p): sha(p) for p in [out/'completion.json', out/'run.log', failure, report]},
        'scope': 'Root immutable-evidence/schema/counter-bound/resource review following native/public exact validation; no labels, new quality replicas, hardware result or full-goal completion.'}
    (out/'root_review.json').write_text(json.dumps(review, indent=2)+'\n')
    (out/'review_completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {
        str(out/'root_review.json'): sha(out/'root_review.json'), str(report): sha(report)}}, indent=2)+'\n')
    print('All40same-fit word validation cases reviewed; full task comparison remains open')


if __name__ == '__main__':
    main()
