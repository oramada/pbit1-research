import tempfile,unittest
from pathlib import Path
import numpy as np
from .packed_float_outputs import save_output,load_output
class PackedOutputTest(unittest.TestCase):
    def test_all_bit_patterns_roundtrip(self):
        # Include signed zero, infinities, quiet/signaling NaN payloads and
        # subnormals, then arbitrary bits. Compare bits rather than floatvalues.
        special=np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0x7f800001,1,0x80000001],np.uint32);random=np.random.default_rng(20261170).integers(0,2**32,1024,dtype=np.uint32);bits=np.r_[special,random].reshape(3,344)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'values.npz';save_output(path,bits.view(np.float32));restored=load_output(path);np.testing.assert_array_equal(restored.view(np.uint32),bits);self.assertEqual(restored.shape,(3,344));self.assertEqual(restored.dtype,np.float32)
if __name__=='__main__':unittest.main()
