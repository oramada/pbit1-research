# API

`SinglePBit(eps=1e-6)` is the NumPy reference implementation for signed,
affine local-range encoding. `encode(weights, group_ids=None)` returns
probabilities and shared centers/radii. `sample(encoded, n_samples, rng)`
returns sampled affine weights; `decode(samples)` averages them.
`forward(inputs, encoded, n_samples, rng)` averages linear outputs, using
weight shape `(input_features, output_features)`. Supply a NumPy generator.
Explicit `eps=0` permits ideal endpoints. Constant groups have zero radius.

`NativeProbabilityBank(weights, beta=1)` uses official THRML. Weights must
be finite in `[0,1]`. `sample(seed, n_samples, n_chains=1)` returns binary
states with shape `(chains, samples, *weights.shape)` for vector input.
`controls()` reports logical/free/clamped nodes and float32 bias storage;
`realized_probabilities` reports the probability after bias rounding.
Exact endpoints are clamped nodes. The native bank defaults to CPU.

The default install has NumPy encoders and nonideality utilities.
The optional `thrml` extra installs the tested CPU simulator environment.
The LFM3 model conversion and large GPU worker live in the separate archive;
this small SDK is not a drop-in Transformers quantization plugin.
