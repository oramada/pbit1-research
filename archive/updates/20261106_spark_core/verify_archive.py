from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'FILES.sha256.json').read_text())
actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p.name!='FILES.sha256.json'}
assert actual==set(manifest), 'Missing or unexpected archive files'
for name,expected in manifest.items():
 h=hashlib.sha256()
 with(root/name).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 assert h.hexdigest()==expected,name
print('Verified',len(manifest),'archive file hashes')
