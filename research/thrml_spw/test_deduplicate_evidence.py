"""Verify storage sharing preserves bytes, independent atomic replacement and undo."""
import hashlib,json,os,tempfile,unittest,io,contextlib
from unittest.mock import patch
from pathlib import Path
from . import deduplicate_evidence as dedup

class DeduplicationTest(unittest.TestCase):
    def setUp(self):
        self.old=Path.cwd();self.tmp=tempfile.TemporaryDirectory();os.chdir(self.tmp.name);folder=Path('results/thrml_spw/closed');folder.mkdir(parents=True);(folder/'manifest.json').write_text('{}');data=b'original evidence'*100000;self.paths=[folder/'stage0_inputs.npy',folder/'stage0_conditional_reference.npy']
        for p in self.paths:p.write_bytes(data);p.chmod(0o640)
        h=hashlib.sha256(data).hexdigest();Path('results/thrml_spw/artifact_index.json').write_text(json.dumps({str(p):{'sha256':h,'bytes':len(data)} for p in self.paths}));dedup.OUT.mkdir();self.digest=h
    def tearDown(self):os.chdir(self.old);self.tmp.cleanup()
    def test_share_and_restore(self):
        p=dedup.plan();r=dedup.apply(p);self.assertEqual(r['new_paths_linked'],1);self.assertTrue(os.path.samefile(*self.paths));self.assertEqual(self.paths[0].stat().st_mode&0o777,0o444)
        # Atomic replacement of one name never changes another name's bytes.
        tmp=self.paths[1].with_suffix('.tmp');tmp.write_bytes(self.paths[1].read_bytes());os.replace(tmp,self.paths[1]);self.assertFalse(os.path.samefile(*self.paths));self.assertEqual(dedup.sha(self.paths[0]),self.digest)
        dedup.restore(p);self.assertFalse(os.path.samefile(*self.paths))
        for path in self.paths:self.assertEqual(dedup.sha(path),self.digest);self.assertEqual(path.stat().st_mode&0o777,0o640)
    def test_completed_apply_is_idempotent(self):
        p=dedup.plan();dedup.save(dedup.OUT/'plan.json',p);dedup.apply(p);before=(dedup.OUT/'evidence_audit.json').read_bytes();out=io.StringIO()
        with patch('sys.argv',['deduplicate_evidence','--apply']),contextlib.redirect_stdout(out):dedup.main()
        self.assertTrue(json.loads(out.getvalue())['cached_complete']);self.assertEqual((dedup.OUT/'evidence_audit.json').read_bytes(),before)
    def test_mismatch_prevents_all_mutation(self):
        p=dedup.plan();self.paths[1].write_bytes(b'changed');self.assertRaises(AssertionError,dedup.apply,p);self.assertFalse(os.path.samefile(*self.paths));self.assertEqual(self.paths[0].stat().st_mode&0o777,0o640)

if __name__=='__main__':unittest.main()
