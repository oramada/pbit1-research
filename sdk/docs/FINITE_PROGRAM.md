# Finite probability programs (unreleased)

`FinitePBitProgram` exposes the NumPy reconstruction used in the DGX Spark
research. Install from a checkout containing this update with `python -m pip install .`.
The existing v0.1.0a1 release does not yet contain this API.

```python
import numpy as np
from pbit1 import FinitePBitProgram

weights = np.linspace(-0.5, 0.7, 256).reshape(16, 16)
program = FinitePBitProgram.encode(
    weights, group_size=128, layout="transposed_matrix"
)
result = program.reconstruct(32768, seed=7, parameter_index=0)
print(result.weights)          # FP32 reconstruction in original matrix layout
print(program.expectation())  # mean with the same finite controller
print(program.storage())      # controls and range metadata counted explicitly
```

The controller has 4,096 levels over the half-logit range [-4, 4]. Each contiguous
group in the selected layout stores an FP32 minimum and span. Transposed grouping
flattens the transposed matrix; groups can cross its row boundaries. For depthwise
filters with seven coefficients, `group_size=7` keeps each filter together.

The simulator draws exact independent binomial counts instead of emitting every
Bernoulli state. It uses uint16 counts through T=65,535 and uint32 above that.
`parameter_index` and `seed` identify independent parameter draws reproducibly.
The output is FP32; the research neural runners separately round to BF16. This
API does not install weights into a model, select layouts automatically, simulate
correlated devices, or perform native THRML execution.

Actual command arrays use two bytes per scalar, plus eight bytes per group.
`storage()` reports the ideal packed 12-bit alternative separately and adds the
one physical binary state per parameter as its own count. Sampling counters,
reconstructed weights, Python metadata, and device-control circuitry are not
included in the program-array total. One logical p-bit is not one bit of total
system storage.

The package tests cover input validation, reproducible sampling, constants,
layout restoration, and counter overflow. The separate research verification
compares exact count hashes and BF16-rounded outputs against the frozen sampler
on a group-boundary fixture, a transposed matrix, and three original LFM3 tensors.
