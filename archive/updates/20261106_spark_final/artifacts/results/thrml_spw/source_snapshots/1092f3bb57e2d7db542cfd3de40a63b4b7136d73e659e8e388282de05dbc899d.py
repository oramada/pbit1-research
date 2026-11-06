"""Upload only fixed inputs needed by the authorized H100 comparison."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import modal


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[2]
    r=root/'results/thrml_spw/lfm3_1p7b'
    p=json.loads((r/'adapted_word_validation_v2_protocol.json').read_text())
    files={r/name for name in p['inputs_sha256']}
    files.update([r/'adapted_word_validation_v2_protocol.json',r/'program_export_gpu_preflight/independent_audit.json',
        r/'adaptation_pilot/mean/module_aliases.json',r/'counted_mean_t8192_protocol.json'])
    for v in json.loads((r/'adaptation_pilot/mean/program_package.json').read_text())['programs']:
        files.add(root/v['program'])
    for v in json.loads((r/'adapted_word_package/package.json').read_text())['programs']:
        for item in v['outputs'].values():files.add(root/item['path'])
    for name,seeds in [('counted_mean_validation',range(202610350,202610355)),
            ('counted_mean_t4096',range(202610420,202610425))]:
        for seed in seeds:
            for suffix in ['results.json','independent_audit.json']:
                files.add(r/name/'mean'/str(seed)/suffix)
    for batch in range(4):files.add(r/f'counted_mean_validation/mean/202610350/batch{batch}.npz')
    # Existing accepted K8 first case provides nominal-control parity if needed.
    for suffix in ['results.json','independent_audit.json']:
        path=r/'adapted_word_validation_v2/C1024/K8/mean/202610350'/suffix
        if path.exists():files.add(path)
    manifest={'status':'frozen_asset_upload_manifest','created_utc':datetime.now(timezone.utc).isoformat(),
        'files':{str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)},
        'scope':'Pinned parameter programs, metadata, completed reference controls/audits only. No training gradients, credentials or heldout token arrays.'}
    out=root/'results/thrml_spw/modal_h100_v1'
    out.mkdir(exist_ok=True)
    mp=out/'asset_manifest.json'
    with mp.open('x') as f:json.dump(manifest,f,indent=2)
    total=sum(v['bytes'] for v in manifest['files'].values())
    print(json.dumps({'files':len(files),'bytes':total,'status':'upload_starting'}),flush=True)
    volume=modal.Volume.from_name('spw-h100-validation-v1',create_if_missing=True)
    with volume.batch_upload() as batch:
        for path in sorted(files):batch.put_file(str(path),'/'+str(path.relative_to(root)))
        batch.put_file(str(mp),'/asset_manifest_v1.json')
    print(json.dumps({'status':'fixed_inputs_uploaded','files':len(files),'bytes':total}),flush=True)


if __name__=='__main__':main()
