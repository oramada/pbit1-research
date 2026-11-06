"""Synthetic rejection fixtures for coverage and frozen relative-quality gate."""
import json
import numpy as np
from .audit_lfm3_benchmark_v1 import coverage,primary_statistics
from .prepare_v1 import OUT,sha

def main():
 offsets=np.array([0,5,9]);windows=[{'article':0,'context_start':0,'context_stop':5,'target_start':1,'target_stop':5},{'article':1,'context_start':0,'context_stop':4,'target_start':1,'target_stop':4}]
 np.testing.assert_array_equal(coverage(windows,offsets),[4,3]);rejected=[]
 for name,broken in [('duplicate',windows+[windows[0]]),('missing',windows[:1]),('future', [{**windows[0],'context_stop':4},windows[1]])]:
  try:coverage(broken,offsets)
  except AssertionError:rejected.append(name)
  else:raise AssertionError(name)
 assert primary_statistics(np.full(5,np.log(1.019)),0.)['pass']
 assert not primary_statistics(np.full(5,np.log(1.021)),0.)['pass']
 # A good mean alone is insufficient if seed uncertainty exceeds the margin.
 assert not primary_statistics(np.array([-.1,-.1,0.,.1,.1]),0.)['pass']
 output={'status':'coverage_and_relative_quality_gate_rejection_fixtures_passed','source_sha256':sha(__file__),'auditor_sha256':sha(__import__('pathlib').Path(__file__).with_name('audit_lfm3_benchmark_v1.py')),'rejected':rejected,'quality_fixture_count':3}
 (OUT/'lfm3_audit_helpers_verification_v1.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))
if __name__=='__main__':main()
