"""Verify committed research artifacts against the snapshot manifest."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST.sha256.json').read_text())
tracked=set(subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0'))-{''}
assert tracked==set(manifest)|{'MANIFEST.sha256.json'},'Manifest/committed file coverage differs'
for name,digest in manifest.items():
 h=hashlib.sha256()
 with(root/name).open('rb')as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==digest,name
print('Verified',len(manifest),'research artifact hashes.')
