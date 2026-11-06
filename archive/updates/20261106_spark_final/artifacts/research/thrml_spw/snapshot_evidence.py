"""Archive current research sources and refresh the raw-evidence checksum index.

Historical content-addressed source copies are retained. Run after experiments
finish, never treat this inventory alone as a scientific verification.
"""
from pathlib import Path
import hashlib,json


def main():
    root=Path('results/thrml_spw');root.mkdir(parents=True,exist_ok=True)
    path=root/'source_snapshot_manifest.json'
    manifest=json.loads(path.read_text()) if path.exists() else {}
    current={};archive=root/'source_snapshots';archive.mkdir(exist_ok=True)
    for source in sorted(Path('research/thrml_spw').glob('*.py')):
        data=source.read_bytes();digest=hashlib.sha256(data).hexdigest()
        target=archive/f'{digest}.py'
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
        current[str(source)]={'sha256':digest,'archive':str(target)}
    manifest['current_sources']=current
    manifest.setdefault('scope','Current sources and retained content-addressed history; does not certify that every historical source version was archived')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    index_path=root/'artifact_index.json';files=set()
    for folder in ['research/thrml_spw','results/thrml_spw','data/thrml_spw']:
        directory=Path(folder)
        if directory.exists():files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=index_path)
    for name in ['requirements-thrml.lock.txt','requirements-thrml-gpu.lock.txt','requirements-models.snapshot.txt','README.md','.gitignore']:
        if Path(name).exists():files.add(Path(name))
    index={}
    for source in sorted(files):
        digest=hashlib.sha256()
        with source.open('rb') as handle:
            for data in iter(lambda:handle.read(1024*1024),b''):digest.update(data)
        index[str(source)]={'bytes':source.stat().st_size,'sha256':digest.hexdigest()}
    index_path.write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({'current_source_modules':len(current),'indexed_files':len(index),'indexed_bytes':sum(r['bytes'] for r in index.values())}))

if __name__=='__main__':main()
