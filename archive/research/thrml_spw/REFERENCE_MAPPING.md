# Reference mapping and baseline definitions

## Source evidence

The official paper's Section III maps p-bit states to sampled data and latent
variables, with J and h as model parameters. Appendix D.5.1 discusses memory for
programmed weights and biases. Appendix G expands multivalued data variables into
sums of binary states. These are distinct roles. Source:
[Extropic paper](https://arxiv.org/html/2510.23972v1), Sections III, D.5.1, G.

The exact energy and conditional update convention are also in the pinned source:
[Ising model](https://github.com/extropic-ai/thrml/blob/2d54cd8404df85bb65703d7e8b4e84e48c1b72a9/thrml/models/ising.py)
and [conditional sampler](https://github.com/extropic-ai/thrml/blob/2d54cd8404df85bb65703d7e8b4e84e48c1b72a9/thrml/conditional_samplers.py).
The README links a community DTM replication, not an official four-p-bits-per-learned-
weight implementation: [pinned README](https://github.com/extropic-ai/thrml/blob/2d54cd8404df85bb65703d7e8b4e84e48c1b72a9/README.md).

## Implication for this research

Our construction deliberately assigns a sampled node to each neural matrix weight,
then programs its unary field to obtain the required marginal probability. This is
a valid use of the simulator. It is a different mapping from identifying a neural
weight with an existing Ising coupling J. The current evidence does not establish
that Extropic previously needed four p-bits for each of those learned parameters.

The baselines therefore have explicit meanings:

| Baseline | Purpose | What it does not establish |
|---|---|---|
| Official five-node Ising example | Reproduce library semantics and sampling law | End-to-end Extropic application performance |
| K identical-probability sampled replicas | Compare space/time with the same observation budget | Positional low-bit encoding |
| Deterministic unsigned K-bit positional code | Conventional numerical precision/storage comparator | An official Extropic learned-weight encoding |
| Same trained positive-weight model in FP32/UINT4/UINT8 | Isolate representation/task-quality effects | Physical hardware energy, density, or latency |

Future system comparisons must specify an implementable controller, accumulator,
readout schedule, and actual reference architecture. A reduction in logical sampled
nodes alone cannot be promoted to a fourfold systems improvement.
