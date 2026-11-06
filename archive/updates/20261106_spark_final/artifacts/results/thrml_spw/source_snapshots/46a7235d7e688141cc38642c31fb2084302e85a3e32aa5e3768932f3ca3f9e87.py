"""Resume infrastructure transfer in bounded batches; never rerun sampling."""
from pathlib import Path
import json
import hashlib
import modal


def main():
    root=Path(__file__).resolve().parents[2]
    mp=root/'results/thrml_spw/modal_h100_v1/asset_manifest.json'
    manifest=json.loads(mp.read_text())
    volume=modal.Volume.from_name('spw-h100-validation-v1')
    entries=volume.listdir('/',recursive=True)
    remote={e.path.lstrip('/'):e.size for e in entries}
    paths=[name for name,rec in manifest['files'].items() if remote.get(name)!=rec['bytes']]
    print(json.dumps({'remaining_files':len(paths),'existing_entries':len(entries)}),flush=True)
    for start in range(0,len(paths),12):
        group=paths[start:start+12]
        with volume.batch_upload(force=True) as batch:
            for name in group:batch.put_file(str(root/name),'/'+name)
        print(json.dumps({'uploaded_files':min(start+12,len(paths)),'remaining_total':len(paths)}),flush=True)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(mp),'/asset_manifest_v1.json')
    print('fixed_input_transfer_complete_full_remote_hash_check_still_required',flush=True)


if __name__=='__main__':main()
