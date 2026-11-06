"""Seal validation-only learned RGB native witness; never claims task quality."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'probability_native_witness_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'review_completion.json').exists()
    c = json.loads((OUT/'completion.json').read_text())
    for path, expected in c['artifact_sha256'].items():
        assert sha(path) == expected, path
    p = json.loads((OUT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, expected in p[group].items():
            assert sha(path) == expected, path
    audit = json.loads((OUT/'evidence_audit.json').read_text())
    assert audit['passed'] and audit['all_weight_nodes_free']
    assert audit['protocol_sha256'] == sha(OUT/'protocol.json')
    assert len(audit['rows']) == 30
    assert {(r['model'], r['method'], r['T']) for r in audit['rows']} == {
        (mi, method, t) for mi in range(5) for method in p['methods'] for t in [64, 256, 1024]}
    entries = 0
    for row in audit['rows']:
        assert sha(row['path']) == row['sha256']
        assert row['all28_bank_arrays_exact']
        with np.load(row['path'], allow_pickle=False) as arrays:
            assert len(arrays.files) == 14
            for li, gshape, cshape in [(0, (2, 4096, 8), (2, 8)), (1, (2, 1024, 16), (2, 16)), (2, (2, 8, 10), (2, 10))]:
                g, c = arrays[f'gate_{li}'], arrays[f'column_{li}']
                assert g.shape == gshape and c.shape == cshape
                assert g.dtype == c.dtype == np.int32
                assert np.all(g >= 0) and np.all(c >= 0)
                assert arrays[f'output_{li}'].dtype == np.float64 and np.isfinite(arrays[f'output_{li}']).all()
                entries += g.size+c.size
    assert entries == audit['counter_entries_exact'] == 2955960
    assert audit['canonical_bank_arrays_exact'] == 840
    assert audit['native_retained_sample_updates'] == audit['public_retained_sample_updates'] == 2*3928*10*(64+256+1024)
    assert audit['warmup_updates_per_pass'] == 2*3928*30 == 235680
    report = ROOT.parent/'CIFAR_CNN_PROBABILITY_NATIVE_WITNESS_REPORT.md'
    assert not report.exists()
    report.write_text('''# Learned-command RGB native validation

All30validation-only CPU THRML cases passed: five starting models, two trained probability methods, T64/256/1024 and two banks of8validation inputs. Native observers and independently recovered public raw states agree exactly for2955960integer counters and840canonical bank arrays, including activation codes/padding masks/intermediate outputs. All3928weight nodes remain free, with no activation p-bits.

The native adapter retains trained8bitcommands/F32fields and originalF64per-output centers/radii/biases. It never re-fits ranges to learned expected weights. Public recovery independently reconstructs original ranges, latent-to-command rounding and fields, then uses separate patch/readout contractions. All saved case/source/input/completion hashes and integer schemas are checked. This establishes bounded learned-command execution correctness, not task quality.

CPU native/public passes each105584640retained-sample scheduled node updates plus235680warmup updates. Integer counters precede externalF64readout/ReLU/maxpool. Official THRML0.1.4/clean pinned source and CPUJAXF32sampling are used; logical one-p-bit encoding still requires finite probability controls/range metadata/readout. No physical energy, latency, area or whole-systemmemory advantage is measured.

The earlier ten-fit surrogate training/validation preparation remains separate. Its original frozenRGB2.6936point quality-target miss is retained. Extra epochs confound comparisons to the originalFP32reference; biased STE/binomial-mean training is not native or unbiased-gradient training. No testlabels/taskmetrics were used here. GPUfull-budget validation, all full native task cases, matched controls, separate public replay and statistical verification remain required before any trained RGB quality claim. All methods/budgets must be retained.

[Frozen native validation protocol](cifar_cnn_native/probability_native_witness_evidence/protocol.json) · [All30cases and native/public checks](cifar_cnn_native/probability_native_witness_evidence/evidence_audit.json) · [Training preparation](CIFAR_CNN_PROBABILITY_FINETUNE_PREPARATION_REPORT.md). The broad research goal remains active.
''')
    paths = [OUT/'protocol.json', OUT/'evidence_audit.json', OUT/'completion.json', ROOT/'probability_native_witness.log', report]
    review = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'cases_checked': 30, 'counter_entries_checked': entries,
              'artifact_sha256': {str(path): sha(path) for path in paths},
              'scope': 'Root bounded learned-command CPU native correctness review only; no GPU/test quality or systems conclusion.'}
    (OUT/'root_review.json').write_text(json.dumps(review, indent=2)+'\n')
    (OUT/'review_completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {
        str(OUT/'root_review.json'): sha(OUT/'root_review.json'), str(report): sha(report)}}, indent=2)+'\n')
    print('All30learned RGB witness files/seals/counts reviewed; test quality remains open', flush=True)


if __name__ == '__main__':
    main()
