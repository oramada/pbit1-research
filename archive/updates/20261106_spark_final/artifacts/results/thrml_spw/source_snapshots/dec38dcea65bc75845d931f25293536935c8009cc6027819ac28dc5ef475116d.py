"""Seal native positional scalar protocol before sampling."""
import json
from .finite_positional_common import OUT,SOURCES,design,sha
from pathlib import Path
def main():
    assert not (OUT/'protocol.json').exists();OUT.mkdir(parents=True,exist_ok=True);p=design();p['source_sha256']={str(Path('research/thrml_spw')/s):sha(Path('research/thrml_spw')/s) for s in SOURCES};p['official_commit']='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9';(OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');print('Frozen before sampling:',sha(OUT/'protocol.json'))
if __name__=='__main__':main()
