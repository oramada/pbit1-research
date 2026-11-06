"""Seal full RGB production file integrity while independent replay is live."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'cohort_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'production_review.json').exists()
    protocol = json.loads((OUT/'protocol.json').read_text())
    for group in ['source_sha256', 'input_sha256']:
        for path, expected in protocol[group].items():
            assert sha(path) == expected, path
    pipeline = json.loads((OUT/'pipeline.json').read_text())
    stage = next(r for r in pipeline['stages'] if r['stage'] == 'production')
    assert stage['status'] == 'completed' and stage['returncode'] == 0
    assert stage['log_sha256'] == sha(OUT/'logs/production.log')
    replay = next(r for r in pipeline['stages'] if r['stage'] == 'public_native_replay')
    assert replay['status'] == 'running'
    m = json.loads((OUT/'manifest.json').read_text())
    assert m['complete'] and len(m['rows']) == 150
    assert m['protocol_sha256'] == sha(OUT/'protocol.json')
    assert m['test_data_sha256'] == sha(OUT/'test_data.npz')
    expected = [(mi, t, seed) for mi in range(5) for t in [64, 256, 1024] for seed in range(10)]
    assert [(r['model'], r['T'], r['seed']) for r in m['rows']] == expected
    updates, counters = 0, 0
    for row in m['rows']:
        name = f"model{row['model']}_T{row['T']}_seed{row['seed']}"
        assert row['name'] == name and Path(row['path']) == OUT/'cases'/(name+'.npz')
        assert sha(row['path']) == row['sha256']
        assert row['all_receipts_and_final_logits_exact'] is False
        with np.load(row['path'], allow_pickle=False) as a:
            assert set(a.files) == {'receipts', 'logits', 'clipped_coordinates'}
            assert a['receipts'].shape == (1250, 14, 32) and a['receipts'].dtype == np.uint8
            assert a['logits'].shape == (10000, 10) and a['logits'].dtype == np.float64
            assert np.isfinite(a['logits']).all()
            assert a['clipped_coordinates'].shape == (3,) and a['clipped_coordinates'].dtype == np.int64
            np.testing.assert_array_equal(a['clipped_coordinates'], row['clipped_coordinates'])
            assert np.all(a['clipped_coordinates'] >= 0)
        counters += 1250*49266
        updates += 1250*3928*row['T']
    assert updates == m['scheduled_updates'] == 329952000000
    assert counters == m['counter_entries'] == 9237375000
    live = []
    for base in Path('/proc').iterdir():
        if not base.name.isdigit():
            continue
        try:
            argv = (base/'cmdline').read_bytes().split(b'\0')
            if not argv or b'.venv-thrml-gpu/bin/python' not in argv[0]:
                continue
            if b'results.thrml_spw.cifar_cnn_native.cohort.run' not in argv or b'audit' not in argv:
                continue
            stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
            assert stat[0] not in ['Z', 'X']
            live.append({'pid': int(base.name), 'start_ticks': stat[19], 'state': stat[0]})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    assert len(live) == 1
    result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
              'checker_source_sha256': sha(__file__), 'case_files_checked': 150,
              'final_logit_values_finite': 15000000, 'retained_sample_updates': updates,
              'scheduled_counter_entries': counters, 'warmup_updates': 736500000,
              'actual_live_public_replay': live, 'production_stage': stage,
              'artifact_sha256': {str(path): sha(path) for path in [OUT/'protocol.json', OUT/'manifest.json', OUT/'logs/production.log']},
              'scope': 'Full production coverage/file hashes/schema/finite logits/resource counts and successful process exit. Public-state receipt correctness, task metrics and statistical/control acceptance remain pending.'}
    (OUT/'production_review.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'case_files_checked': 150, 'actual_live_public_replay': live}))


if __name__ == '__main__':
    main()
