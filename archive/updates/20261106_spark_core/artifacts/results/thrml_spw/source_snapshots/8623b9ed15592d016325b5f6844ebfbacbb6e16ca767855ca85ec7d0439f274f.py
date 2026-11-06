"""Release complete audited validation metadata, irrespective of quality outcome."""
from pathlib import Path
import hashlib
import json
import time
import modal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/thrml_spw/lfm3_1p7b')
    directory = root / 'counted_mean_t8192'
    ap, sp = directory / 'statistics_audit.json', directory / 'statistics.json'
    deadline = time.time() + 86400
    while not ap.exists():
        assert time.time() < deadline
        time.sleep(30)
    audit = json.loads(ap.read_text())
    assert audit['status'] == 'all_five_mean_quality_statistics_and_workspace_resource_counts_verified'
    assert audit['source_sha256'] == 'dd9b5578c16fe25ecc3f53c7dc1c45fcd5d0e678e145fc19df0c890fc3ae99f5'
    assert audit['statistics_sha256'] == sha(sp)
    paths = [ap, sp, root / 'counted_mean_t8192_protocol.json']
    for seed in range(202610600, 202610605):
        rp, cp = directory / 'mean' / str(seed) / 'results.json', directory / 'mean' / str(seed) / 'independent_audit.json'
        case = json.loads(cp.read_text())
        assert case['source_sha256'] == 'f85318c765d6a6177c310c757cf0714c1cac0f849676dbb2e06d7c30da01a85d'
        assert case['results_sha256'] == sha(rp) and case['seed'] == seed
        paths.extend([rp, cp])
    manifest = {'status': 'all_five_C8192_reference_cases_independently_audited',
        'files': {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths}}
    out = Path('results/thrml_spw/modal_h100_v1/C8192_assets_v1.json')
    with out.open('x') as f:
        json.dump(manifest, f, indent=2)
    volume = modal.Volume.from_name('spw-h100-validation-v1')
    with volume.batch_upload(force=True) as batch:
        for p in paths:
            batch.put_file(p, '/' + str(p))
    with volume.batch_upload(force=True) as batch:
        batch.put_file(out, '/C8192_assets_v1.json')
    print('complete_C8192_metadata_released', flush=True)


if __name__ == '__main__':
    main()
