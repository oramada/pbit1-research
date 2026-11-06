"""Root closure review and unit-checked reporting of all RGB contrasts."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'closure_review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    reviews = {}
    reports = []
    for name, terminal, contrast_count, report_name in [
        ('cohort_evidence', 'all150compactRGBnative_cases_independently_audited_goal_active', 30, 'CIFAR_CNN_NATIVE_REVIEW.md'),
        ('precision_evidence', 'FP16_BF16_RGB_controls_audited_goal_active', 12, 'CIFAR_CNN_PRECISION_REVIEW.md')]:
        folder = ROOT/name
        completion = json.loads((folder/'completion.json').read_text())
        assert completion['status'] == terminal
        for path, h in completion['artifact_sha256'].items():
            assert sha(path) == h, path
        p = json.loads((folder/'protocol.json').read_text())
        for kind in ['source_sha256', 'input_sha256']:
            for path, h in p[kind].items():
                assert sha(path) == h, path
        pipeline = json.loads((folder/'pipeline.json').read_text())
        assert pipeline['status'] == terminal and all(r['returncode'] == 0 and r['status'] == 'completed' for r in pipeline['stages'])
        s = json.loads((folder/'statistics.json').read_text())
        arrays = np.load(folder/'statistical_arrays.npz')
        assert len(s['contrasts']) == contrast_count
        parent_arrays = np.load(ROOT/'cohort_evidence/statistical_arrays.npz')
        native = parent_arrays['values']
        assert native.shape == (3, 5, 10, 2)
        baselines = arrays['baseline_metrics'] if name == 'cohort_evidence' else arrays['control_metrics']
        seen = set()
        lines = ['# '+('Native compact RGB CIFAR cohort review' if name == 'cohort_evidence' else 'RGB FP16/BF16 supplement review'), '',
                 'All completion-bound hashes, frozen source/input seals and successful pipeline stages have been checked. Five independently trained compact16x16RGB CNNs,3928weights/34deterministic biases, all10000fixed historically consumed test images. Descriptive five-model/ten-MC-seed intervals are not fresh generalization confirmation, equivalence tests or physical hardware evidence.', '',
                 '| Native samples | Accuracy (%) | Mean CE |', '|---:|---:|---:|']
        for bi, t in enumerate([64, 256, 1024]):
            mean = native[bi].mean((0, 1))
            lines.append(f'|{t}|{100*mean[0]:.4f}|{mean[1]:.8f}|')
        lines += ['', '| Samples | Native minus control | Accuracy difference (pp) | Descriptive95%interval (pp) | CE difference | Descriptive95%CE interval |',
                  '|---:|---|---:|---|---:|---|']
        for c in s['contrasts']:
            bi, ci = c['budget_index'], c['control_index']
            assert (bi, ci) not in seen
            seen.add((bi, ci))
            assert c['T'] == [64, 256, 1024][bi]
            point = (native[bi]-baselines[ci, :, None]).mean((0, 1))
            np.testing.assert_allclose(point, c['difference'], rtol=2e-10, atol=2e-12)
            dist = arrays[f'budget{bi}_control{ci}']
            assert dist.shape == (5000, 2)
            interval = np.quantile(dist, [.025, .975], axis=0).T
            np.testing.assert_allclose(interval, c['descriptive_95_interval'], rtol=2e-10, atol=2e-12)
            lines.append(f"|{c['T']}|{c['control']}|{100*point[0]:+.4f}|[{100*interval[0, 0]:+.6f},{100*interval[0, 1]:+.6f}]|{point[1]:+.8f}|[{interval[1, 0]:+.8f},{interval[1, 1]:+.8f}]|")
        assert seen == {(bi, ci) for bi in range(3) for ci in range(10 if name == 'cohort_evidence' else 4)}
        if name == 'cohort_evidence':
            audit = json.loads((folder/'statistics_audit.json').read_text())
            native_audit = json.loads((folder/'native_audit.json').read_text())
            manifest = json.loads((folder/'manifest.json').read_text())
            assert audit['passed'] and audit['controls_checked'] == 50 and audit['bootstrap_scalar_values_checked'] == 300000
            assert native_audit['passed'] and native_audit['complete'] and native_audit['manifest_sha256'] == sha(folder/'manifest.json')
            assert len(native_audit['rows']) == len(manifest['rows']) == 150
            assert native_audit['counter_entries'] == 9237375000 and native_audit['scheduled_updates'] == 329952000000
            assert all(r['all_receipts_and_final_logits_exact'] for r in native_audit['rows'])
            fp = next(c for c in s['contrasts'] if c['T'] == 1024 and c['control'] == 'FP32_F64')
            assert fp['difference'][0] < -.02 and fp['descriptive_95_interval'][0][1] < -.02
            lines += ['', 'At the largest predetermined budget, native53.5644%versusFP3256.2580% gives a2.6936percentage-point loss (descriptive95%loss interval2.304985–3.081805points). This frozen RGB cohort misses the approximate1–2point target; implementation correctness does not imply application success. Lower budgets are substantially worse and are retained above.', '',
                'AtT1024 native exceeds the dequantized min-max INT4control51.6980%by1.8664points; the descriptive interval is[0.038975,3.680270]points. This is one of30retained contrasts, without multiplicity adjustment or a new outcome-selected success gate. T64/T256native remain below that control. INT8 and FP16/BF16 retain near-FP32quality; no representation, energy or latency superiority is established.', '',
                'Native production/publicreplay each329952000000retained updates plus736500000warmup updates. All9237375000counter entries/15millionfinal logits were publicly regenerated; all50deterministic controls/300000bootstrap scalar values separately checked. Logical payload4656bytes and197064bytes/bank int32counters are software accounting, not measured area/energy/latency. Biases/activation quantization/readout and physical controller information remain costs. The internal four-node matched-resource comparison is separate and still in progress.']
        else:
            audit = json.loads((folder/'evidence_audit.json').read_text())
            assert audit['passed'] and audit['full_logits_checked'] == 2000000 and audit['bootstrap_values_checked'] == 120000
            lines += ['', 'All20full control paths/2millionlogits and120000bootstrap scalar values passed independent verification. CPU-low-precision module/functional outputs match exactly. Weight-only controls use independently rounded weights and separate NumPyF64graphs. Observed output dtypes do not establish internal accumulator widths.', '',
                'Mean accuracy: FP16weight-only56.2400%, BF16weight-only56.2660%, FP16CPU56.2400%, BF16CPU56.2940%. No new native draws; CPUsoftware arithmetic is not a probabilistic-device performance measurement. Nominal7856weightbytes plus136FP32biasbytes or68low-precisionbiasbytes exclude activations/framework/readout buffers.']
        lines += ['', 'Every planned model/budget/control contrast is retained. The broader research goal remains active; physical calibration/correlation/drift, energy/latency/area and broader fresh-data/model confirmation remain open.', '',
                  f'[Full statistics]({name.replace(name, "cifar_cnn_native/"+name)}/statistics.json) · [Completion receipt](cifar_cnn_native/{name}/completion.json)']
        report = ROOT.parent/report_name
        assert not report.exists()
        report.write_text('\n'.join(lines)+'\n')
        reports.append(report)
        reviews[name] = {'completion_sha256': sha(folder/'completion.json'),
                         'contrast_rows_checked': contrast_count, 'bootstrap_scalar_values_unit_checked': contrast_count*5000*2,
                         'pipeline_status': pipeline['status'], 'report_sha256': sha(report)}
    OUT.mkdir()
    record = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'passed': True, 'source_sha256': sha(__file__), 'reviews': reviews,
              'scope': 'Root completion/source/input/process-stage/contrast-unit review; not whole research completion or new independent task replicas.'}
    (OUT/'root_review.json').write_text(json.dumps(record, indent=2)+'\n')
    paths = [OUT/'root_review.json']+reports
    (OUT/'completion.json').write_text(json.dumps({'passed': True, 'artifact_sha256': {str(path): sha(path) for path in paths}}, indent=2)+'\n')
    print('RGBnative+precisionclosure hashes/stages/all42contrasts/420000bootstrap units reviewed', flush=True)


if __name__ == '__main__':
    main()
