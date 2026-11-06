"""Verify hard gates cannot consume the reserved cohort or construct a model."""
import subprocess,sys,unittest
from pathlib import Path
class PreflightTests(unittest.TestCase):
    def isolated(self,body):
        code="from unittest.mock import patch\nimport sys\nfrom pathlib import Path\nfrom research.thrml_spw import fine_range_confirmation as c\nassert 'torch' not in sys.modules and 'jax' not in sys.modules\nbefore={str(p):(p.read_bytes() if p.exists() else None) for p in [c.OUT/'protocol.json',c.OUT/'tokens.npz',c.OUT/'execution.json',c.OUT/'baseline_tokens.npz']}\n"+body+"\nassert 'torch' not in sys.modules and 'jax' not in sys.modules\nassert all((Path(p).read_bytes() if Path(p).exists() else None)==v for p,v in before.items())\n"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
    def test_low_storage_precedes_all_model_and_native_imports(self):
        self.isolated("from collections import namedtuple\nUsage=namedtuple('Usage','total used free')\nwith patch('research.thrml_spw.fine_range_common.shutil.disk_usage',return_value=Usage(100,99,1)):\n try:c.preflight()\n except RuntimeError as e:assert 'persistent free bytes' in str(e)\n else:raise AssertionError('Storage gate bypassed')\n")
    def test_failed_scientific_gate_precedes_model_import(self):
        self.isolated("original=Path.read_text\ndef read(p,*args,**kwargs):\n if str(p)=='results/thrml_spw/range_localization/fixed_compilation/evidence_audit.json':return '{\"passed\":false}'\n return original(p,*args,**kwargs)\nwith patch.object(c,'require_capacity'),patch.object(Path,'read_text',read):\n try:c.preflight()\n except AssertionError as e:assert 'Fixed-compiler' in str(e)\n else:raise AssertionError('Scientific gate bypassed')\n")
if __name__=='__main__':unittest.main()
