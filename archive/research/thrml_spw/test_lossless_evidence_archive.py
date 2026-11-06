"""Original-byte preservation, corruption refusal and interrupted-commit recovery."""
import io,json,stat,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import lossless_evidence_archive as store
from .npy_archive_codec import encode,decode,sha_bytes
from .verify_npy_archive import independently_restore
class ArchiveTests(unittest.TestCase):
    def test_all_float_bits_orders_headers_and_endianness(self):
        rng=np.random.default_rng(2026118101)
        for dtype in ['<f4','>f4','<f8','>f8']:
            width=np.dtype(dtype).itemsize;bits=rng.integers(0,256,(8,9,width),dtype=np.uint8);array=bits.reshape(-1).view(dtype).reshape(8,9)
            for order in ['C','F']:
                value=np.array(array,order=order);stream=io.BytesIO();np.save(stream,value);raw=stream.getvalue();blob,m=encode(raw,'results/thrml_spw/fixture/runs/x.npy',0o444,1000000000);restored,meta=decode(blob);self.assertEqual(raw,restored);self.assertEqual(meta,m)
                with tempfile.TemporaryDirectory() as td:
                    archive=Path(td)/'x.spwnpy';archive.write_bytes(blob);independent,info=independently_restore(archive);self.assertEqual(raw,independent);self.assertEqual(info,m)
        special=np.array([0x0000000000000000,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff0000000000001,1],np.uint64).view(np.float64);s=io.BytesIO();np.save(s,special);blob,_=encode(s.getvalue(),'special.npy',0o644,0);self.assertEqual(decode(blob)[0],s.getvalue())
    def test_unsupported_and_corrupt_data_refused(self):
        stream=io.BytesIO();np.save(stream,np.arange(100,dtype=np.int32))
        with self.assertRaises(ValueError):encode(stream.getvalue(),'x.npy',0o644,0)
        stream=io.BytesIO();np.save(stream,np.ones(1000,np.float64));raw=stream.getvalue();blob,_=encode(raw,'x.npy',0o644,0)
        for bad in [blob[:7],blob[:-3],b'BROKEN!!'+blob[8:],blob[:-1]+bytes([blob[-1]^1])]:
            with self.assertRaises(Exception):decode(bad)
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'bad';p.write_bytes(bad)
                with self.assertRaises(Exception):independently_restore(p)
        with self.assertRaises(ValueError):encode(raw+b'extra','x.npy',0o644,0)
    def fixture(self,td):
        root=Path(td);original='results/thrml_spw/fixture/runs/x.npy';p=root/original;p.parent.mkdir(parents=True);np.save(p,np.zeros((64,64),np.float64));p.chmod(0o444);s=p.stat();raw=p.read_bytes();out=root/'archives';row={'id':0,'original_path':original,'archive_path':'archives/payloads/x.spwnpy','original_sha256':sha_bytes(raw),'original_bytes':len(raw),'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns};return root,out,row,raw
    def test_durable_archive_idempotent_restore_and_release(self):
        with tempfile.TemporaryDirectory() as td:
            root,out,row,raw=self.fixture(td);r=store.archive_one(row,root,out);self.assertFalse((root/row['original_path']).exists());self.assertEqual(independently_restore(root/row['archive_path'])[0],raw);again=store.archive_one(row,root,out);self.assertEqual(r,again);record=out/'records/00000.json';store.restore_one(record,root,True);p=root/row['original_path'];self.assertEqual(p.read_bytes(),raw);self.assertEqual(stat.S_IMODE(p.stat().st_mode),row['mode']);self.assertEqual(p.stat().st_mtime_ns,row['mtime_ns']);self.assertFalse((root/row['archive_path']).exists());self.assertEqual(json.loads(record.read_text())['status'],'restored')
    def test_failed_journal_retains_raw_and_can_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root,out,row,raw=self.fixture(td)
            with patch.object(store,'durable_json',side_effect=OSError('simulated journal full')):
                with self.assertRaises(OSError):store.archive_one(row,root,out)
            self.assertEqual((root/row['original_path']).read_bytes(),raw);self.assertEqual(independently_restore(root/row['archive_path'])[0],raw);store.archive_one(row,root,out);self.assertFalse((root/row['original_path']).exists())
    def test_shared_links_and_wrong_hash_never_removed(self):
        with tempfile.TemporaryDirectory() as td:
            root,out,row,raw=self.fixture(td);p=root/row['original_path'];alias=p.with_name('shared.npy');alias.hardlink_to(p)
            with self.assertRaises(AssertionError):store.archive_one(row,root,out)
            self.assertEqual(p.read_bytes(),raw);alias.unlink();row['original_sha256']='0'*64
            with self.assertRaises(AssertionError):store.archive_one(row,root,out)
            self.assertEqual(p.read_bytes(),raw)
    def test_failed_archive_sync_retains_raw(self):
        with tempfile.TemporaryDirectory() as td:
            root,out,row,raw=self.fixture(td)
            with patch.object(store.os,'fsync',side_effect=OSError('simulated sync failure')):
                with self.assertRaises(OSError):store.archive_one(row,root,out)
            self.assertEqual((root/row['original_path']).read_bytes(),raw);self.assertFalse((root/(row['archive_path']+'.tmp')).exists())
if __name__=='__main__':unittest.main()
