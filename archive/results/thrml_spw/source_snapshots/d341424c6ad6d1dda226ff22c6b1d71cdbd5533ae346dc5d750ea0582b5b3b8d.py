"""Seal bounded RGB word evidence and derive prospective matched-update costs."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'word_witness_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'review_completion.json').exists()
    completion = json.loads((OUT/'completion.json').read_text())
    for path, expected in completion['artifact_sha256'].items():
        assert sha(path) == expected, path
    p = json.loads((OUT/'protocol.json').read_text())
    for kind in ['source_sha256', 'input_sha256']:
        for path, expected in p[kind].items():
            assert sha(path) == expected, path
    d = json.loads((OUT/'evidence_audit.json').read_text())
    assert d['passed'] and d['protocol_sha256'] == sha(OUT/'protocol.json')
    assert {(r['model'], r['T_word']) for r in d['rows']} == {(mi, t) for mi in range(5) for t in [1, 4, 16]}
    assert len(d['rows']) == 15
    entries = 0
    for row in d['rows']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path'], allow_pickle=False) as a:
            assert len(a.files) == 14
            for li, gate, column in [(0, (2, 4096, 8), (2, 8)),
                                      (1, (2, 1024, 16), (2, 16)),
                                      (2, (2, 8, 10), (2, 10))]:
                assert a[f'gate_{li}'].shape == gate
                assert a[f'column_{li}'].shape == column
                assert a[f'gate_{li}'].dtype == a[f'column_{li}'].dtype == np.int32
                assert np.all(a[f'gate_{li}'] >= 0)
                assert np.isfinite(a[f'output_{li}']).all()
                entries += a[f'gate_{li}'].size+a[f'column_{li}'].size
    assert entries == 1477980 == d['counter_entries_exact']
    assert d['native_retained_sample_updates'] == d['public_retained_sample_updates'] == 2*15712*5*21
    assert d['warmup_updates_per_pass'] == 2*15712*15
    future_updates = 5*10*1250*15712*(16+64+256)
    assert future_updates == 5*10*1250*3928*(64+256+1024) == 329952000000
    report = ROOT.parent/'CIFAR_CNN_WORD_WITNESS_REPORT.md'
    assert not report.exists()
    report.write_text('''# Bounded RGB four-p-bit reference validation

All15 CPU validation cases passed: five compact RGB CIFAR checkpoints, first16validation inputs in two banks of8, and word sample counts1/4/16. Native observer outputs and separately recovered public states agree exactly for all1477980integer counters and420canonical bank arrays. Each pass uses3299520retained-sample scheduled node updates plus471360warmup updates. These are validation correctness checks, without task scoring or physical measurements.

The internal positional reference uses four free THRML nodes per normalized weight, coefficients1/2/4/8 and finite fields±4. There are15712weight nodes versus3928single nodes. Both share per-output centers/radii, deterministic biases and ACT8 activation ranges; integer output accumulation precedes externalF64readout and nonlinearities. This comparator is not an asserted Extropic hardware weight format.

A future full comparison must pair single samples64/256/1024 with word samples16/64/256, on the same five models/all10000images/ten inference seeds. Each full production pass would use329952000000retained updates. This is prospective accounting, not executed word task work. Four times as many nodes and one quarter as many rounds do not imply equal energy or latency. Word warmup would be2946000000updates per full pass, versus736500000single; charge it separately.

Logical word target commands require1964bytes at4bits/weight, plus544bytes ranges,136bytes biases and48bytes activation ranges:2692bytes. Single commands use3928bytes, giving4656bytes with identical metadata. Dynamic states are separately1964word versus491single bytes. Shared fixed-field descriptions, physical controllers/retention, calibration, buffers and readout are additional; this is not a demonstrated total physical density improvement. Both software observers retain197064bytes of int32counters per8-imagebank. At the largest proposed word sample count256, the largest integer bound is250675200, below signed32bit limits.

Full-budget GPU validation, all150word task cases, their public replay, deterministic word-mean controls and paired statistical verification remain required before an RGB quality/resource comparison. The single-node RGB cohort is still running. The broad research goal remains active.

[Frozen witness protocol](cifar_cnn_native/word_witness_evidence/protocol.json) · [Native/public audit](cifar_cnn_native/word_witness_evidence/evidence_audit.json) · [Root review](cifar_cnn_native/word_witness_evidence/root_review.json)
''')
    paths = [OUT/'completion.json', OUT/'protocol.json', OUT/'evidence_audit.json',
             ROOT/'word_witness.log', report]
    review = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'cases_checked': 15, 'counter_entries': entries,
              'future_matched_retained_updates_per_pass': future_updates,
              'scope': 'Bounded native witness closure and prospective resources only; no full word task quality or physical systems claim.',
              'artifact_sha256': {str(path): sha(path) for path in paths}}
    (OUT/'root_review.json').write_text(json.dumps(review, indent=2)+'\n')
    (OUT/'review_completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {
        str(OUT/'root_review.json'): sha(OUT/'root_review.json'), str(report): sha(report)}}, indent=2)+'\n')
    print('All15RGB word witness files/source/input seals reviewed; prospective matched budgets derived.')


if __name__ == '__main__':
    main()
