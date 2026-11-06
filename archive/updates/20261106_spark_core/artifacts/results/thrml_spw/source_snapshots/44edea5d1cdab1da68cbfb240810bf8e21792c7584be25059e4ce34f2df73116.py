"""Causal streaming neural validation: same controls, matched node updates."""
from pathlib import Path
import hashlib,json,shutil
OUT=Path('results/thrml_spw/positional_student')
SOURCES=['positional_student_common.py','prepare_positional_student.py','positional_student.py','positional_student_worker.py','audit_positional_student.py','positional_student_audit_worker.py','positional_student_statistics.py','verify_positional_student_statistics.py','positional_operator_common.py','positional_operator_native.py','verify_positional_operators.py','native.py','controlled_positive_native.py','packed_float_outputs.py','provenance.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard(p):
    for f,h in p['source_sha256'].items():assert sha(f)==h,(f,h)
def capacity(p):
    free=shutil.disk_usage(OUT).free
    if free<p['minimum_free_before_scoring']:raise RuntimeError(f'Need {p["minimum_free_before_scoring"]} free persistent bytes before scoring; have {free}.')
def condition(p,name):return next(c for c in p['conditions'] if c['name']==name)
def check_prerequisites(p):
    guard(p)
    for f,h in p['operator_audits'].items():assert sha(f)==h and json.loads(Path(f).read_text())['passed']
    assert sha(p['tokens_path'])==p['tokens_sha256']
    for e in p['matrices']:
        for key in ['single_controls','positional_controls']:assert sha(e[key]['path'])==e[key]['sha256']
