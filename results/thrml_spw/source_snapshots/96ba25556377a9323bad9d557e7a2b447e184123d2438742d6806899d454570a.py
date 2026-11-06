"""Read-only partial production checks; never computes task quality or replays draws."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'cohort_evidence'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    for collection in ['source_sha256', 'input_sha256']:
        for path, expected in protocol[collection].items():
            assert digest(path) == expected, path
    # Read one atomic manifest version; the producer may append later cases.
    raw = (OUT/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    assert manifest['protocol_sha256'] == digest(OUT/'protocol.json')
    assert manifest['test_data_sha256'] == digest(OUT/'test_data.npz')
    expected_cases = [(mi, t, seed) for mi in range(5)
                      for t in [64, 256, 1024] for seed in range(10)]
    rows = manifest['rows']
    assert 0 < len(rows) <= 150
    assert [(r['model'], r['T'], r['seed']) for r in rows] == expected_cases[:len(rows)]
    assert manifest['complete'] == (len(rows) == 150)
    counters = 0
    updates = 0
    checked = []
    for row in rows:
        name = f"model{row['model']}_T{row['T']}_seed{row['seed']}"
        path = OUT/'cases'/(name+'.npz')
        assert row['name'] == name and Path(row['path']) == path
        assert digest(path) == row['sha256']
        assert row['all_receipts_and_final_logits_exact'] is False
        with np.load(path, allow_pickle=False) as case:
            assert set(case.files) == {'receipts', 'logits', 'clipped_coordinates'}
            assert case['receipts'].shape == (1250, 14, 32)
            assert case['receipts'].dtype == np.uint8
            assert case['logits'].shape == (10000, 10)
            assert case['logits'].dtype == np.float64
            assert np.isfinite(case['logits']).all()
            clips = case['clipped_coordinates']
            assert clips.shape == (3,) and clips.dtype == np.int64
            assert np.all(clips >= 0)
            # Inputs: 3x16x16; post-pool feature maps: 8x8x8,16x4x4.
            assert np.all(clips <= np.array([768, 512, 256])*10000)
            np.testing.assert_array_equal(clips, row['clipped_coordinates'])
        # Convolutions retain separate positive/negative activation gates;
        # column occupancy counters are per output, shared across the bank.
        counters += 1250*(2*8*16*16*8+8+2*8*8*8*16+16+8*10+10)
        updates += 1250*3928*row['T']
        checked.append({'path': str(path), 'sha256': row['sha256']})
    assert counters == manifest['counter_entries']
    assert updates == manifest['scheduled_updates']
    live = []
    for pid, ticks, module in [
        (3964689, '47733621', 'results.thrml_spw.cifar_cnn_native.cohort.pipeline'),
        (3965140, '47736133', 'results.thrml_spw.cifar_cnn_native.cohort.run')]:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert stat[19] == ticks and stat[0] not in ['Z', 'X']
        assert module.encode() in argv
        live.append({'pid': pid, 'start_ticks': ticks, 'state': stat[0], 'module': module})
    now = datetime.datetime.now(datetime.timezone.utc)
    record = {'recorded_utc': now.isoformat(), 'passed': True,
              'previous_turn_classification': 'verified_wait',
              'this_turn_classification': 'progress',
              'checker_sha256': digest(__file__),
              'manifest_snapshot_sha256': hashlib.sha256(raw).hexdigest(),
              'manifest_snapshot': manifest, 'case_files_checked': len(rows),
              'case_hashes_checked': checked, 'live_processes': live,
              'scope': 'Saved file hashes/schema/finite logits/coverage/accounting only; no task metrics, public replay, statistical acceptance, or cohort completion claim.'}
    directory = ROOT/'execution_checks'
    directory.mkdir(exist_ok=True)
    target = directory/(now.strftime('%Y%m%dT%H%M%SZ')+'_production_integrity.json')
    assert not target.exists()
    target.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'passed': True, 'cases_checked': len(rows), 'record': str(target)}))


if __name__ == '__main__':
    main()
