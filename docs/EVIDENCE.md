# Results and evidence map

Use this page to connect a claim to the saved results that support it.
All paths below point to the accepted November 6 DGX Spark investigation.

## The main model comparison

| Model/task | Original BF16 | Selected K1 | Five-draw probability-quality ratio and MC95 interval |
|---|---:|---:|---|
| GPT-2 / 283,225 WikiText-2 targets | PPL 25.50149 | PPL 25.66953 | 1.00659 [0.99927, 1.01397] |
| DistilBERT / 25,000 IMDb reviews | Accuracy 87.512% | Mean accuracy 87.4944% | Categorical PPL ratio 0.99966 [0.99676, 1.00256] |
| LFM3 / 294,442 WikiText-2 targets | PPL 18.93015 | PPL 18.97918 | 1.00259 [0.98751, 1.01790] |

“PPL” means perplexity; lower is better. DistilBERT's accuracy is higher-is-better,
and its categorical perplexity is a classification probability metric. These
are different task metrics. The README's chart keeps their scales separate.

The selected configurations use one logical p-bit for every learned scalar,
finite 12-bit controls, shared ranges, and 32,768 independent samples. The primary
probability-quality criterion requires the upper five-seed t4 interval to be at
most 1.02 relative to BF16. DistilBERT also passes its two-percentage-point
accuracy-decline criterion. The upper confidence bound on its accuracy decline
is 0.08383 percentage points.

The intervals describe random-weight-draw variability on these fixed data.
These benchmarks had been inspected before; they are not newly unseen tasks.
Full-model execution reconstructs weights and then uses conventional BF16 operators.

## Find the supporting artifacts

| Claim | Result | Independent check |
|---|---|---|
| Three complete selected configurations pass | [Cross-model summary](../results/thrml_spw/spark_depth_20261106/cross_model_summary_v2/results.json) | [GPT-2 audit](../results/thrml_spw/spark_depth_20261106/benchmark_audit_v1/results.json), [DistilBERT audit](../results/thrml_spw/spark_depth_20261106/distilbert_benchmark_audit_v1/results.json), [LFM3 audit](../results/thrml_spw/spark_depth_20261106/lfm3_benchmark_audit_v1/results.json) |
| All LFM3 weight points reconstruct and selected windows replay | [Replay report](../results/thrml_spw/spark_depth_20261106/lfm3_benchmark_replay_v1/results.json) | Seven model points, 1,430 count hashes, 21 neural windows; not every forward pass |
| GPT-2 head/range grouping explains the tested failure | [Role interventions](../results/thrml_spw/spark_depth_20261106/tied_head_v1/results.json) | [Activation-weighted variance](../results/thrml_spw/spark_depth_20261106/activation_variance_v1/results.json) |
| GPT-2 uses 41.7% fewer nominal observations | [Budget summary](../results/thrml_spw/spark_depth_20261106/reduced_work_summary_v1/results.json) | [Full task/budget audit](../results/thrml_spw/spark_depth_20261106/reduced_work_gpu_v1_audit.json), [SDK/neural replay](../results/thrml_spw/spark_depth_20261106/reduced_work_replay_v1/results.json) |
| Layer sensitivity is not simply additive | [LFM3 causal study](../results/thrml_spw/spark_depth_20261106/lfm3_causal_depth_analysis_v1/results.json) | [Cumulative GPT-2 audit](../results/thrml_spw/spark_depth_20261106/gpt2_cumulative_v1_audit.json) |
| Neural errors can be correlated despite independent weight draws | [Covariance study](../results/thrml_spw/spark_depth_20261106/lfm3_error_covariance_v2/results.json) | [All 48 subset covariance matrices audited](../results/thrml_spw/spark_depth_20261106/lfm3_error_covariance_v2_audit.json) |
| Stronger multi-element controls can beat K1 operator error | [Native THRML comparison](../results/thrml_spw/spark_depth_20261106/native_coarse_fractional_v1_audit.json) | [Exact moment checks](../results/thrml_spw/spark_depth_20261106/native_coarse_fractional_moments_v1.json) |
| Controls and working weights must be counted | [Resource ledger](../results/thrml_spw/spark_depth_20261106/resource_ledger_final_v1.json) | Exact declared payload counts; not chip area or full process memory |
| Universal lossless BF16 mapping is impossible for this fixed format | [Executed collision](../results/thrml_spw/spark_depth_20261106/precision_limits_v1/results.json) | [Capacity argument](../results/thrml_spw/spark_depth_20261106/precision_limits_v1/capacity_note.md) |

## How to interpret the files

- **Protocol:** the fixed inputs, seeds, mapping, and evaluation rule.
- **Result:** the observations and statistics from a completed run.
- **Audit:** an independent recomputation or consistency check of saved outputs.
- **Replay:** a reconstruction or forward pass compared with the recorded run.
- **Source snapshot:** the exact executed source bytes, indexed by SHA-256.
- **Manifest:** an integrity check for files; it does not by itself establish correctness.

Source code for a proposed experiment is not evidence that it ran successfully.
Failed and interrupted experiments remain in `archive/`; they are not silently
counted as passes. Training-only or single-draw diagnostics are labeled separately
from complete five-draw task evaluations.

## What is and is not included

The repository includes compact reports, protocols, source snapshots, figures,
and audits. It omits pretrained weight files, token arrays, and most large raw
outputs, with hashes and restoration requirements preserved. A summary audit
report cannot substitute for the omitted arrays when rerunning an audit.

For broader discussion, read [the paper](../paper/main.pdf),
[the supplement](../paper/supplement.pdf), or [the detailed research report](../SPARK_RESEARCH.md).
