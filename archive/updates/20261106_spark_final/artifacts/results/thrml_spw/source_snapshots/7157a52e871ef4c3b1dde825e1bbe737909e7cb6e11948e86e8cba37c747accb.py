"""Resume fixed assets, transporting large raw references as lossless gzip."""
from pathlib import Path
import json
import modal


def main():
    root=Path(__file__).resolve().parents[2];base=root/'results/thrml_spw/modal_h100_v1'
    mp=base/'asset_manifest.json';manifest=json.loads(mp.read_text())
    compressed={v['original']:v for v in json.loads((base/'compressed_transfer_v3.json').read_text())['files']}
    volume=modal.Volume.from_name('spw-h100-validation-v1')
    remote={e.path.lstrip('/'):e.size for e in volume.listdir('/',recursive=True)}
    transfers=[]
    for name,rec in manifest['files'].items():
        if remote.get(name)==rec['bytes']:continue
        if name in compressed:
            rec=compressed[name];destination='compressed_inputs_v1/'+name+'.gz'
            if remote.get(destination)==rec['gzip_bytes']:continue
            transfers.append((rec['local_gzip'],destination))
        else:transfers.append((name,name))
    print(json.dumps({'status':'resume_lossless_transfer','files':len(transfers),'remote_entries':len(remote)}),flush=True)
    for start in range(0,len(transfers),48):
        with volume.batch_upload(force=True) as batch:
            for source,target in transfers[start:start+48]:batch.put_file(str(root/source),'/'+target)
        print(json.dumps({'uploaded':min(start+48,len(transfers)),'total':len(transfers)}),flush=True)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(base/'compressed_transfer_v3.json'),'/compressed_transfer_v3.json')
        batch.put_file(str(mp),'/asset_manifest_v1.json')
    print('all_fixed_inputs_uploaded_remote_reconstruction_and_hash_audit_required',flush=True)


if __name__=='__main__':main()
