# PBit1: a concrete spatial–temporal precision experiment

**Question:** Can each learned parameter be assigned to one programmable
p-bit, with precision supplied through time and a shared local range?

**Completed result:** All 1,728,801,536 unique learned parameters of
LiquidAI/LFM3-1.7B were converted to finite probabilistic commands. At 8,192
samples, five independent Monte Carlo seeds achieved mean perplexity
75.1547 versus 75.3147 for the same adapted weights executed in BF16.
The conditional 95% t4 perplexity-ratio interval is 0.98062–1.01544,
passing the original upper-bound criterion of 1.02 on 2,048 fixed targets.

**Implementation:** Extropic's official THRML factors and Gibbs sampler,
pinned at `2d54cd8404df85bb65703d7e8b4e84e48c1b72a9`.
Twelve-bit probability commands, FP32 grouped ranges, ACT12 inputs,
integer accumulation, and BF16 nonlinearities are explicitly retained.
Embedding/head aliases share one canonical parameter assignment.

**Cross-GPU result:** Native and nominal arrays replayed bit-for-bit across
GB10 and H100 in the completed spatial-control bridge. The same-fit BF16
reference shifted about 0.00515% in perplexity. The original strict bridge
failure is retained; an explicit later amendment uses H100-specific controls.

**Spatial motivation:** One logical element per parameter gives 4× fewer
logical p-bits than a four-element positional encoding. It trades spatial
replication for time; it does not establish a 4× energy or speed gain.
The measured T8192 run schedules 56.7 trillion retained node evaluations
per complete validation pass. Controllers, readout, ranges and counters matter.

**Completed matched-resource comparison (November 5):** an independent audit now covers 45 accepted cases. The lowest explored qualifying budget is C8192 for K1 and C1024 for K8: one-eighth the logical p-bits for K1, but eight times the retained scheduled work and 64 times the temporal samples. These are fixed-subset, backend-labelled comparisons. They establish a density/time tradeoff rather than a measured energy or latency advantage. [Full update](BROAD_VERIFICATION_20261105.md).

**What would be useful to investigate with Extropic:** map the unary
stationary-probability programming interface to actual devices; measure
calibration and effective independent switching rate; evaluate accumulation
and controller costs; compare matched quality and transitions against an
agreed hardware-native representation. Internal K4/K8 controls are not
a reproduction of a proprietary Extropic parameter representation.

**Current boundary:** This is an independent, installable research SDK and
a conditional validation result. The larger LFM3 reserved-data study is running with checkpointed recovery. The first full-task, five-seed R1 primaries are mixed: DistilBERT passes (87.532% versus 87.512% accuracy on IMDb), while GPT-2 misses (31.8679 versus 25.5024 perplexity on WikiText-2). The combined broad primary criterion failed; the complete original sweep continues. [Audited outcomes and scope](BROAD_VERIFICATION_20261105.md).
No physical hardware energy, latency, residency or numerical BF16-equivalence
claim is made. Probability encoding and stochastic computing are established;
the contribution investigated here is the measured model-level tradeoff.

[SDK](https://github.com/oramada/pbit1) ·
[Experiment archive](https://github.com/oramada/pbit1-extropic)
