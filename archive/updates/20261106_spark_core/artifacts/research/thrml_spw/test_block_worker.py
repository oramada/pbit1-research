"""The worker transport and resume invariants are part of scientific correctness."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class BlockWorkerChecks(unittest.TestCase):
    def test_native_constant_operator_and_cache_input_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            w = np.array([[.25, -.5], [.25, -.5]], dtype=np.float32)
            bias = np.array([.125, .25], dtype=np.float32)
            np.savez(root/"weights.npz", weight_0=w, bias_0=bias)
            x = np.array([[[1., 2.], [3., 4.]]], dtype=np.float32)
            np.save(root/"input.npy", x);np.save(root/"reference.npy", x@w+bias)
            process = subprocess.Popen([sys.executable, "-u", "-m", "research.thrml_spw.block_native_worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(json.loads(process.stdout.readline())["status"], "ready")
                request = {"weights": str(root/"weights.npz"), "inputs": str(root/"input.npy"),
                    "reference": str(root/"reference.npy"), "output": str(root/"output.npy"),
                    "stage": 0, "samples": 4, "seed": 3, "domain": 940730, "contexts_per_batch": 8}
                def call():
                    process.stdin.write(json.dumps(request)+"\n");process.stdin.flush()
                    response = json.loads(process.stdout.readline())
                    self.assertEqual(response["status"], "ok", response)
                    return response
                self.assertFalse(call()["cached"])
                np.testing.assert_array_equal(np.load(root/"output.npy"), x@w+bias)
                self.assertTrue(call()["cached"])
                x = x*2
                np.save(root/"input.npy", x);np.save(root/"reference.npy", x@w+bias)
                self.assertFalse(call()["cached"])
                np.testing.assert_array_equal(np.load(root/"output.npy"), x@w+bias)
                metadata = json.loads((root/"output.json").read_text())
                self.assertEqual(metadata["free_node_updates"], 0)
                self.assertEqual(metadata["logical_weight_readouts"], 16)
                self.assertEqual(metadata["controls"]["constant_groups"], 2)
                process.stdin.write('{"action":"quit"}\n');process.stdin.flush()
                self.assertEqual(process.wait(timeout=30), 0)
            finally:
                if process.poll() is None:process.terminate();process.wait(timeout=30)
                process.stdin.close();process.stdout.close();process.stderr.close()


if __name__ == "__main__":unittest.main()
