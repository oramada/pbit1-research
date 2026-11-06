# What each comparison establishes

The active full-model benchmark compares one-p-bit encodings with identical 12-bit probability controls, group sizes, and per-parameter sample budgets. It isolates grouping direction. BF16 is the task-quality reference; these runs do not measure physical device energy.

The older multi-p-bit reference has a different precision contract. In `research/thrml_spw/broad/parameter_program_v1.py`, a K-bit word is quantized first. Its bitplanes use endpoint fields corresponding to probabilities δ or 1−δ, where δ = sigmoid(−8). Its expected unit value is

    δ + (1−2δ) × word / (2^K−1).

Its independent finite-sample variance is

    δ(1−δ) × sum(4^k, k=0..K−1) / [(2^K−1)^2 T].

Quantization bias remains as T increases. This is not the same baseline as K replicas all programmed to the target analog probability. Any future K4 comparison must use the actual declared reference and preserve that distinction.

The package's ideal `MultiPBit` reference instead uses deterministic binary states. Repeatedly reading an unchanged digital word does not improve its precision. It should also be compared as an ordinary low-bit representation, without assigning fictitious switching energy to retained deterministic bits.

| Comparison | What must be matched or counted |
|---|---|
| Selected K1 vs original-layout K1 | Same controller precision, samples, original model and evaluation targets |
| K1 vs finite-field K4 word | Different probability precision and quantization bias; count controller, range, state, sampling and bias-programming resources |
| K1 vs deterministic INT4 | Task quality and actual representation cost; digital weights require no Monte Carlo reconstruction |
| R1 vs direct stochastic execution | Match samples and numerical accumulation convention; account for reconstructed-weight storage and operator work separately |
| Physical-hardware benefit | Measured or explicit device model for bias retention, transition rate, readout, accumulation, calibration and interconnect |

“One p-bit per parameter” refers to the logical probabilistic element count. It does not mean one stored digital bit per parameter. Current ideal packed control/range/state accounting is about 13.5 bits per weight, and the R1 simulator additionally retains BF16 working weights and temporary counters. A hardware density, energy or latency improvement has not been demonstrated by these software quality tests.
