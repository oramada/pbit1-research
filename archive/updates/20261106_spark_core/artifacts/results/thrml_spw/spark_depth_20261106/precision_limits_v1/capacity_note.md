# Finite-controller capacity bound

Consider a fixed tensor shape and fixed grouping/permutation. The current model uses a 12-bit probability-control code per parameter, plus two FP32 range values per group of 128. For N parameters divisible by 128, this allows at most

    2^(12N + 64N/128) = 2^(12.5N)

possible encoded programs. Even allowing one extra persistent binary state per parameter gives at most 2^(13.5N) states. Fixed per-tensor layout and sample-count metadata adds only the explicitly counted policy bits.

An arbitrary tensor of finite BF16 values has 65,280 available finite bit patterns per coordinate (including signed zero), hence 65,280^N possibilities. Because log2(65,280) is about 15.994, this is strictly larger than the available program space. By the pigeonhole principle, a universal lossless mapping of arbitrary BF16 tensors into this fixed finite-controller format cannot exist. Increasing observation time does not distinguish identical stationary programs. If layout choices, residual values, controller precision, or side tables carry additional information, their storage must be included before applying this bound.

This is a worst-case statement about arbitrary numerical tensors. Learned neural weights can be redundant, and task quality can tolerate representation error. Consequently the bound does not preclude useful one-p-bit-per-parameter neural computation or model-specific compression.

The companion executable counterexample exhibits two different BF16 tensors that collide under the actual current encoder, rather than relying only on the counting argument. Its scope is the finite encoding used in these experiments, not every possible physical programmable p-bit architecture.
