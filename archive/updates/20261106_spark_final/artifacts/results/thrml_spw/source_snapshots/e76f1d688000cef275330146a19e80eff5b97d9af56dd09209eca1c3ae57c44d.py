"""Meaningful fail-closed checks before expensive full-network scoring."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import positional_student_common as common
class PreflightTests(unittest.TestCase):
    def test_rejects_missing_duplicate_or_changed_matched_budget(self):
        correct={'conditions':[{'name':'single','K':1,'T':256},{'name':'positional','K':4,'T':64}]}
        for name in ['single','positional']:self.assertEqual(common.condition(correct,name)['K']*common.condition(correct,name)['T'],256)
        for p,name in [({'conditions':[]},'single'),({'conditions':[correct['conditions'][0]]*2},'single'),({'conditions':[{'name':'single','K':1,'T':512}]},'single'),(correct,'unknown')]:
            with self.assertRaises(ValueError):common.condition(p,name)
    def test_failed_or_altered_upstream_evidence_blocks_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);audit=folder/'audit.json';tokens=folder/'tokens.bin';tokens.write_bytes(b'fixture');audit.write_text(json.dumps({'passed':False}));p={'source_sha256':{},'operator_audits':{str(audit):common.sha(audit)},'tokens_path':str(tokens),'tokens_sha256':common.sha(tokens),'matrices':[]}
            with self.assertRaises(AssertionError):common.check_prerequisites(p)
            audit.write_text(json.dumps({'passed':True}))
            with self.assertRaises(AssertionError):common.check_prerequisites(p)
            p['operator_audits'][str(audit)]=common.sha(audit);common.check_prerequisites(p);tokens.write_bytes(b'changed')
            with self.assertRaises(AssertionError):common.check_prerequisites(p)
    def test_modified_sources_and_insufficient_space_block_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.py';source.write_text('pass\n');p={'source_sha256':{str(source):common.sha(source)}};common.guard(p);source.write_text('raise RuntimeError\n')
            with self.assertRaises(AssertionError):common.guard(p)
        with patch.object(common.shutil,'disk_usage',return_value=type('Usage',(),{'free':10})()):
            with self.assertRaises(RuntimeError):common.capacity({'minimum_free_before_scoring':20})
if __name__=='__main__':unittest.main()
