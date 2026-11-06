"""Run the frozen public auditor with retained pure frequency-function factories.

The scientific observer, key schedule, fields and numerical gates are unchanged.
An unbounded factory cache avoids re-tracing identical shape/sample signatures.
"""
from pathlib import Path
from functools import lru_cache
import json
from . import verify_positional_operators as verifier
from .positional_operator_common import OUT,sha,guard
from .calibrated_mnist_path_audit import PublicLayer

def main():
    p=json.loads((OUT/'protocol.json').read_text());guard(p);m=json.loads((OUT/'manifest.json').read_text());r=json.loads((OUT/'statistics.json').read_text());assert m['source_commit']=='2d54cd8404df85bb65703d7e8b4e84e48c1b72a9' and r['native_operators']==960
    roster={(v['signature']['stage'],v['signature']['K'],v['signature']['T'],v['signature']['seed']) for v in r['rows']};expected={(stage,k,t,seed) for stage in range(24) for k,times in [(1,p['single_samples']),(4,p['positional_samples'])] for t in times for seed in p['seeds']};assert len(r['rows'])==len(roster)==len(expected)==960 and roster==expected
    for row in r['rows']:assert sha(row['output'])==row['output_sha256']
    record={'launcher_sha256':sha(__file__),'frozen_auditor_sha256':sha(Path(verifier.__file__)),'protocol_sha256':sha(OUT/'protocol.json'),'production_manifest_sha256':sha(OUT/'manifest.json'),'cohort_unique_cases_checked':960,'frequency_factory_cache':'unbounded retention of unchanged PublicLayer.frequencies.__wrapped__; identical public native sampler and IEEE/exact gates, no resampling-policy change','audit_completed':False};path=OUT/'audit_cache_execution.json';assert not path.exists();path.write_text(json.dumps(record,indent=2)+'\n');original=PublicLayer.frequencies;cached=lru_cache(maxsize=None)(original.__wrapped__);PublicLayer.frequencies=cached
    try:
        verifier.main();audit=json.loads((OUT/'evidence_audit.json').read_text());record.update({'audit_completed':True,'audit_passed':audit['passed'],'audit_sha256':sha(OUT/'evidence_audit.json'),'frequency_factory_cache_info':str(cached.cache_info())});path.write_text(json.dumps(record,indent=2)+'\n')
    finally:PublicLayer.frequencies=original
if __name__=='__main__':main()
