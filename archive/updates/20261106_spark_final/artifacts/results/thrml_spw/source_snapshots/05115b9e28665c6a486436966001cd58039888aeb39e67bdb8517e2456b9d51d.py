"""Root review of the new validation-only sample-budget extension."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'probability_high_budget_witness_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'review_completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, digest in p[group].items():
            assert sha(path) == digest, path
    c = json.loads((OUT/'completion.json').read_text())
    assert c['passed'] and c['status'] == 'all10_T4096_validation_cases_exact_goal_active'
    for path, digest in c['artifact_sha256'].items():
        assert sha(path) == digest, path
    a = json.loads((OUT/'evidence_audit.json').read_text())
    assert a['passed'] and a['protocol_sha256'] == sha(OUT/'protocol.json')
    assert len(a['rows']) == 10
    assert {(r['method'], r['model']) for r in a['rows']} == {(method, model) for method in range(2) for model in range(5)}
    assert a['native_updates'] == a['public_updates'] == 321781760
    assert a['counter_entries_exact'] == 985320 and a['warmup_updates_per_pass'] == 78560
    for row in a['rows']:
        assert sha(row['path']) == row['sha256'] and row['all14arrays_exact']
        assert row['counter_entries_exact'] == 98532
        assert row['retained_node_updates_per_pass'] == 2*3928*4096
        with np.load(row['path']) as d:
            assert len(d.files) == 14
            assert sum(d[f'gate_{li}'].size+d[f'column_{li}'].size for li in range(3)) == 98532
            for li, ni in enumerate(row['input_dimensions']):
                assert row['signed32_worst_case_counter_bounds'][li] == 255*ni*4096 < 2**31
                assert d[f'gate_{li}'].dtype == d[f'column_{li}'].dtype == np.int32
                assert d[f'codes_{li}'].dtype == np.uint8
                assert d[f'output_{li}'].dtype == np.float64 and np.isfinite(d[f'output_{li}']).all()
                assert d[f'gate_{li}'].min() >= 0
            assert d['output_2'].shape == (2, 8, 10)
    old = Path('results/thrml_spw/fashion_cnn_native/allocated_native/operator.py').read_text().split('\n\ndef recover(')[0]+'\n'
    new = (ROOT/'probability_high_budget/operator.py').read_text()
    old = old.replace('Per-layer sample counts; native counters and independent public-state replay.',
                      'Larger-budget derivative of the frozen allocated native operator; no reconstruction.')
    old = old.replace("1 <= t <= 1024", "1 <= t <= 4096")
    assert old == new, 'The operator must change only docstring, cap and omission of unused recovery function'
    report = ROOT.parent/'CIFAR_CNN_HIGH_BUDGET_NATIVE_WITNESS_REPORT.md'
    assert not report.exists()
    report.write_text('''# Validation of 4096-sample one-p-bit inference

All ten learned RGB models (five starting models, both training methods) passed a validation-only test at 4096 samples/layer on 16 images. All 140 native/public intermediate arrays and 985320 integer counter entries match exactly. Public replay reconstructs learned commands, original ranges/biases, patches and counters separately from the native observer.

The derivative operator retains exactly the previous forward implementation, changing its sample cap from 1024 to 4096. The underlying native gate still checks signed32 worst-case accumulation bounds. Layer fan-ins 27/72/256 give worst-case gated sums 28200960/75202560/267386880, all below 2^31. The saved counters remain int32 and external outputs F64; this is not increased controller precision or extra weight p-bits.

Each model still uses 3928 free weight p-bits, original fixed per-output ranges, 34 deterministic biases, unchanged learned 8-bit commands and ACT8 activation spans. Native and independent public validation each use 321781760 retained scheduled node updates plus 78560 warmup updates. Those counts are not physical bit flips, energy or measured device latency.

The budget was chosen after the consumed 1024-sample study to test whether lower temporal variance closes the remaining BF16 task-quality gap. These validation cases use no labels and establish operator correctness only. They do not establish new test accuracy, BF16 equivalence, fresh-data confirmation, or a systems advantage. Existing lower-budget failures/results remain unchanged.

The next required experiment is a separately frozen full task cohort at the larger budget, retaining both methods, five models and ten MC seeds, followed by complete public replay and direct BF16 comparisons. No new task scores are accepted from this witness.

[Validation protocol](cifar_cnn_native/probability_high_budget_witness_evidence/protocol.json) · [Native/public audit](cifar_cnn_native/probability_high_budget_witness_evidence/evidence_audit.json) · [Root review](cifar_cnn_native/probability_high_budget_witness_evidence/root_review.json).

The broader research goal remains active.
''')
    review = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'case_coverage': 10, 'arrays_checked': 140,
              'counter_entries': 985320, 'operator_diff_checked': True,
              'artifact_sha256': {str(path): sha(path) for path in [OUT/'completion.json', report]},
              'scope': 'Root source/input/completion/case-schema/bounds review of validation-only larger sampling budget, not new task quality.'}
    (OUT/'root_review.json').write_text(json.dumps(review, indent=2)+'\n')
    (OUT/'review_completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {
        str(OUT/'root_review.json'): sha(OUT/'root_review.json'), str(report): sha(report)}}, indent=2)+'\n')
    print('All10 T4096 validation cases/bounds/operator diff reviewed and report sealed')


if __name__ == '__main__':
    main()
