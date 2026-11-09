# The research in plain language

## The idea

A conventional weight stores a numerical value such as `0.7`. A probabilistic bit
has two possible outputs, but its chance of producing `1` can be programmed.

Set that chance to `0.7`. Over many observations, the fraction of ones approaches
`0.7`. That is the basic representation. A shared scale and offset handle weights
outside `[0, 1]`. Encoding numbers this way is established stochastic computing;
our study investigates its tradeoffs for pretrained networks.

**A p-bit stores a programmable distribution, and samples reveal an estimate.**
The probability controls and shared ranges still contain information and have
storage costs of their own.

## What the study asks

Can a pretrained neural network tolerate the estimation error well enough to
keep useful task quality? What sampling budget does it need? Which layers and
weight groups are sensitive? Do control and sampling costs erase the benefit of
using fewer stochastic elements?

## What was tested

Three original pretrained models were converted with one logical p-bit per learned
scalar: GPT-2, DistilBERT, and LiquidAI/LFM3-1.7B. Biases and normalization parameters
were included. Each selected full-model configuration used:

- 32,768 independent samples per weight;
- 12-bit probability commands and shared local ranges;
- five independent model weight draws;
- a complete fixed benchmark evaluation for each draw.

The full-model simulator generates binomial counts, reconstructs the weights,
and runs ordinary BF16 neural computation. It does not emit every individual bit
or simulate the complete model through THRML. Smaller native operator experiments
use Extropic's official THRML sampler.

## What succeeded

The selected configurations passed their predefined task-quality margins on the
evaluated data. GPT-2's perplexity increased about 0.66% and LFM3's about 0.26%.
DistilBERT's mean accuracy declined 0.0176 percentage points.

A separate GPT-2 study reduced nominal reconstruction sampling work by 41.7%
while retaining the same quality margin. This was a fixed research allocation,
not an automatic feature of the package.

## Why earlier configurations failed

Sharing a range across the wrong weights can put too much noise in directions
the network amplifies. GPT-2's output head was especially sensitive. Restoring
that output use largely removed the failure; selecting a better grouping repaired
the full-model comparison at the same sample count.

The best grouping differs across operators and models. There is no single rule
such as “transpose every matrix” that fixes all cases.

## What this does not prove

- Matching task quality is different from reproducing every BF16 number.
- The tested finite-control format has collisions: different BF16 values can
  become the same encoded program. More samples cannot distinguish them.
- One logical p-bit per parameter does not imply one bit of total model storage.
- Software simulation does not establish physical energy, area, or latency savings.
- Three model/task checks on previously inspected datasets do not establish
  generalization to every architecture or task.

The value of the study is a measured success regime, a diagnosed failure
mechanism, and a reproducible way to investigate the tradeoff.

## Read next

[Evidence map](EVIDENCE.md) · [Paper](../paper/main.pdf) ·
[Supplement](../paper/supplement.pdf) · [Reproduction](../REPRODUCING.md)
