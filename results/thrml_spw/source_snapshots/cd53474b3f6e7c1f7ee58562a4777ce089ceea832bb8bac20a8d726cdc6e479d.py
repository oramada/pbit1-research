"""Full fixed-point-activation cohort conventions, separate from frozen parent."""
from pathlib import Path
import json,hashlib
import numpy as np
PARENT=Path('results/thrml_spw/boolean_classifier')
OUT=Path('results/thrml_spw/integer_activation_native/cohort')
PREP=Path('results/thrml_spw/boolean_classifier_preparation_v2/preparation.json')
ROOT=Path('results/thrml_spw/integer_activation_native')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,value):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def atomic_npz(p,**arrays):
    q=p.with_suffix('.tmp')
    with q.open('wb') as f:np.savez_compressed(f,**arrays)
    q.replace(p)
def parent_rows():return [r for r in json.loads((PARENT/'manifest.json').read_text())['rows'] if r['regime']=='R1']
def guard(p):
    for file,h in p['source_sha256'].items():assert sha(file)==h,(file,'changed')
    assert sha(PREP)==p['preparation_sha256'];prep=json.loads(PREP.read_text());assert sha(prep['data_path'])==prep['data_sha256']
    for r in prep['models']:assert sha(r['model_path'])==r['model_sha256'] and sha(r['controls_path'])==r['controls_sha256']
    return prep
def parent_closed():
    assert json.loads((PARENT/'pipeline.json').read_text())['status']=='completed_and_independently_audited_goal_active'
    c=json.loads((PARENT/'completion.json').read_text())
    for file,h in c['artifact_sha256'].items():assert sha(file)==h
    assert json.loads((PARENT/'native_audit.json').read_text())['passed'] and json.loads((PARENT/'statistics_audit.json').read_text())['passed']
def case_path(r):return OUT/'cases'/(r['name'].replace('R1_','integer8_',1)+'.npz')
