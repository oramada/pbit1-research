# DGX Spark: one p-bit per parameter

The long-horizon research goal is active. This investigation uses the local GB10, keeps cloud jobs stopped, and leaves the unrelated `uca32` workload running. Research forwards have a 2 GiB PyTorch allocator cap.

## Completed GPT-2 primary benchmark

The frozen one-p-bit encoding **passes the predefined 2% perplexity criterion** on the complete 283,225-target, 62-article benchmark. All 12 cases are complete, with five independent seeds for each stochastic layout. The saved-output, coverage, and statistical audit passed. All 12 weight points and all 36 prespecified neural replay windows matched exactly.

| Representation | Perplexity at mean NLL | Ratio to BF16 | 95% Monte Carlo ratio interval | Frozen criterion |
|---|---:|---:|---|---|
| BF16 | 25.501486 | 1 | — | Reference |
| Finite-control mean | 25.472793 | 0.998875 | — | Deterministic control |
| Operator-aware K1 | 25.669535 | 1.006590 | 0.999267–1.013967 | Pass |
| Original-layout K1, same budget | 27.022307 | 1.059637 | 1.042543–1.077010 | Fail |

Both stochastic methods use one logical p-bit for every one of 124,439,808 learned parameters, 12-bit probability controls, 128-weight ranges, and 32,768 samples. The joint article/seed bootstrap interval for the selected method is 1.002660–1.011716. This is a reconstruction-first (R1) ideal-sampling result on a previously inspected benchmark; it does not establish exact BF16 bit-pattern equality, full-model native THRML execution, or physical hardware savings.

[Complete audited results](results/thrml_spw/spark_depth_20261106/benchmark_audit_v1/results.json) · [Summary](results/thrml_spw/spark_depth_20261106/benchmark_summary_v1/README.md)

![Complete GPT-2 benchmark](results/thrml_spw/spark_depth_20261106/benchmark_summary_v1/full_benchmark.png)

## Completed DistilBERT cross-model check

All 12 cases on 25,000 IMDb reviews completed, and all saved logits were independently rescored. The one-p-bit policy passes both frozen criteria: classification accuracy and categorical perplexity. All 12 weight points and all 36 prespecified CUDA replay batches matched exactly.

| Metric | BF16 | One-p-bit policy |
|---|---:|---:|
| Accuracy | 87.5120% | 87.4944% |
| 95% upper bound on accuracy decline | — | 0.08383 percentage points |
| Categorical perplexity ratio | 1 | 0.999656 |
| 95% upper bound on perplexity ratio | — | 1.002563 |

The original grouping also passes on this classifier. It failed on GPT-2 at the matched budget; this difference is evidence for model-dependent sensitivity, not a universal grouping rule.

[Audited cross-model summary](results/thrml_spw/spark_depth_20261106/cross_model_summary_v1/results.json)

![Two completed model cohorts](results/thrml_spw/spark_depth_20261106/cross_model_summary_v1/tested_model_quality.png)

## What was measured

- Surveyed all 148 GPT-2 parameter tensors: 124,439,808 unique learned scalars, including its tied token embedding/output matrix.
- Traced all 12 transformer blocks, 144 attention heads, normalization outputs, MLP channels, residual streams, and logits on eight diagnostic training articles.
- Completed 152 causal/control forwards, including restoring each block individually and testing two range-group layouts.
- Completed another 264 forwards: BF16 and two layouts across five independent sampling seeds on 24 separate, previously reserved training articles. Every sampled point still represents every learned scalar with one logical p-bit.

### Result of the first candidate fix

The old layout groups weights in their original flat storage order. The candidate transposes two-dimensional tensors before forming 128-weight range groups, then reverses that permutation for neural execution. It keeps the same probability controller, group size, and 8,192 samples per parameter.

On the 16 reserved confirmation articles (4,096 prediction targets per seed):

| Representation | Mean perplexity / BF16 | 95% Monte Carlo interval | Frozen 2% gate |
|---|---:|---:|---|
| Original grouping | 1.22415 | 1.16862–1.28233 | Fail |
| Reoriented grouping | 1.02549 | 1.00398–1.04747 | Fail |

These intervals describe sampling variation across five model draws on fixed articles. They do not measure uncertainty across all language, architectures, or hardware. All these articles are from the training partition; this is not an official held-out benchmark score.

A further 64-forward tied-matrix experiment separates its input and output roles. Restoring only the output projection gives perplexity 46.10 versus BF16 45.80; restoring only its input use gives 59.66. Transposing only that matrix gives 45.27 on the single diagnostic seed. These are causal clues, not new multi-seed confirmation results. Cross-run logit hashes and saved token losses were checked.

### What the neuron analysis found

Restoring only the tied embedding/output matrix almost removes the degradation on the eight diagnostic articles. Restoring individual transformer blocks has much smaller effects. Channels 496, 430, 36, and 314 dominate the original layout's predicted output noise.

With clean hidden activations held fixed, changing range-group direction reduces predicted mean output-logit variance from 0.61444 to 0.002698, about 228-fold. This is an analytical conditional variance comparison, not a 228-fold accuracy or speed improvement. Total weight variance falls much less: the placement of noise matters, not just its amount.

An operator-level survey also finds that only 21 of 37 measured linear operators prefer the reoriented layout by local activation-weighted variance. A general implementation should select grouping direction by operator and activation statistics rather than blindly transpose everything.

## Native Extropic THRML check

The output-head mechanism was also tested with official THRML sampling: 6,144 logical p-bits representing eight actual GPT-2 output rows, projected against eight real hidden vectors. Both layouts used five independent seeds and nested sample budgets of 128, 512, 2,048, and 8,192.

All 40 weight-moment checks passed. Packed trajectories were independently decoded, and two full native chains were replayed bit for bit (100,663,296 states). At 8,192 samples, mean output MSE about the finite-control expectation was 0.03228 for the original grouping and 0.0002645 for the reoriented grouping. This supports the operator mechanism; it is not full-model native THRML validation.

[Native audit](results/thrml_spw/spark_depth_20261106/native_head_audit_v1/results.json)

![Native THRML output-head check](results/thrml_spw/spark_depth_20261106/native_head_audit_v1/native_head_variance.png)

## Direct stochastic operator check

Using the same saved native THRML trajectories, a direct sample-and-accumulate output head matched reconstruction-first output for all ten layout/seed conditions in FP64 CPU arithmetic. This establishes operator consistency, not a free reduction in sampling variance. A full nonlinear network or a different arithmetic order requires separate testing.

[Direct versus reconstructed operator results](results/thrml_spw/spark_depth_20261106/native_linear_identity_v1/results.json)

## Native matched-observation spatial comparison

Twenty native THRML operator cases now compare one p-bit at 32,768 samples with four positional p-bits at 8,192 samples each. Each case uses the same 201,326,592 total device observations. On the reoriented head bank, average output MSE is 0.00006518 for K1 and 0.003741 for K4. All retained count readouts and the original word-code definition were audited.

This compares the study's actual finite-field four-bit word against a 12-bit probability controller. Their information budgets differ. It is an operator result, not proof of full-model quality or physical cost superiority.

![Native matched-observation comparison](results/thrml_spw/spark_depth_20261106/native_spatial_temporal_audit_v1/equal_observation_head.png)

## Scope and costs

The causal and confirmation experiments draw exact ideal independent Bernoulli **counts** using a binomial generator, reconstruct weights, and run conventional BF16 operators (R1). They do not execute native THRML trajectories or demonstrate direct stochastic hardware execution. The earlier Gaussian trace is labeled separately and is not treated as a p-bit sampling result.

The baseline representation uses one logical p-bit per parameter plus a 12-bit probability-controller code and two FP32 range values per 128-weight group: approximately **13.5 ideal packed bits per parameter including the binary state**. Runtime BF16 tensors, sampling counters, addressing, calibration, and hardware circuits are additional costs. This result establishes neither a one-bit digital model file nor physical energy savings. Exact bookkeeping gives an ideal packed control/range/state/policy payload of 84.375% of BF16 weight storage; retaining controls as uint16 instead gives 109.375%, before working weights and framework overhead. [Resource ledger](results/thrml_spw/spark_depth_20261106/resource_ledger_v1/results.json).

Finite random observations cannot guarantee exact BF16 numerical equivalence for every input and parameter: distinct interior Bernoulli probabilities assign positive probability to the same finite observation sequences. The empirical target is broad task-quality equivalence with explicitly counted resources. An [executable finite-controller collision and exact binomial rounding calculation](results/thrml_spw/spark_depth_20261106/precision_limits_v1/README.md) and a [capacity bound](results/thrml_spw/spark_depth_20261106/precision_limits_v1/capacity_note.md) now document this distinction.

## Follow-up calibration (five seeds)

Four targeted policies were evaluated on training/calibration articles. The best point estimate used operator-selected grouping and 32,768 samples for every parameter: perplexity was 0.45% above BF16, with a 95% Monte Carlo interval of -1.37% to +2.29%. It therefore **failed** the frozen 2% upper-bound rule. The policy that increased samples only outside the tied embedding used 3.07 times the original observation budget and also failed that rule.

The ten-seed extension is complete. Operator-aware grouping with **32,768 samples for every parameter** passes the calibration rule: mean perplexity **0.906% above BF16**, with a 95% Monte Carlo interval from **-0.051% to +1.872%**. The cheaper policy sampling the non-embedding parameters more often still fails (upper bound +2.768%). The original five-seed failures are preserved.

The selected policy uses one p-bit per parameter and four times the original T=8,192 observation budget. Its complete T=32,768 benchmark is reported above, including the original grouping at exactly the same budget. The failed T=8,192 and initial five-seed calibration experiments remain preserved.

![Calibration policy selection](results/thrml_spw/spark_depth_20261106/selection_figures_v1/calibration_selection.png)

[Ten-seed calibration results](results/thrml_spw/spark_depth_20261106/profile_extension_combined_v1.json) · [Frozen full-benchmark protocol](results/thrml_spw/spark_depth_20261106/benchmark_v1/protocol.json)

A pinned DistilBERT checkpoint and 64 IMDb training reviews are prepared for a separate cross-architecture calibration check. The full 25,000-review evaluation and its replay audits have now completed, as reported above. A CPU calibration pass has now collected 53 module summaries and selected directions for 40 weight matrices: 24 prefer transposed grouping and 16 prefer original grouping. In one feed-forward matrix, blindly transposing predicts about 3.20 times the original output variance. These predictions use FP32 arithmetic with BF16-rounded weights, not a BF16 task-quality test.

![DistilBERT operator directions](results/thrml_spw/spark_depth_20261106/distilbert_diagnostic_summary_v1/operator_direction.png)

## Nonideal controls and calibration

An illustrative real-weight operator study now includes temporal correlation, cross-device correlation, gain/offset mismatch, and drift. These are explicit simulated assumptions, not measured Extropic device characteristics. Fixed mismatch produces a bias floor that more temporal samples cannot remove.

Calibration can also add noise. In the reoriented head bank, unregularized 4,096-shot fitting worsened expected output MSE under the assumed small mismatch. A binomial MAP fit using the explicitly assumed device-population prior reduced it from 0.0002803 to 0.0001608. The fit was checked against an independent optimizer. Its three probes cost 12,288 observations per device, or 37.5% of one 32,768-sample reconstruction, before amortization across reuse. The assumed prior must be measured or validated before making a hardware claim.

![Calibration cost](results/thrml_spw/spark_depth_20261106/calibration_prior_audit_v1/calibration_cost.png)

[Calibration audit](results/thrml_spw/spark_depth_20261106/calibration_prior_audit_v1/results.json) · [Comparison and resource contracts](results/thrml_spw/spark_depth_20261106/resource_comparison_contract.md)

## Selective sampling follow-up

Training-only gradients identified small normalization/bias tensors that need high precision. A five-seed CPU check of one derived schedule used about 40% fewer nominal observations with nearly unchanged loss, but its maximum sample count was four times higher. That can increase latency despite saving total work.

A second schedule uses groups of 16 for those small vectors and keeps groups of 128 elsewhere. It adds **53,088 bytes** of range metadata, retains one p-bit for every parameter, and keeps the maximum sample count at **32,768**. In five-seed CPU calibration it used **41.7% fewer observations** than uniform sampling. Its paired perplexity ratio to uniform sampling was 0.99924, with a Monte Carlo interval of 0.99567–1.00282. This is CPU FP32 arithmetic with sampled BF16 weights; it still needs prospective GPU benchmark validation.

[Audited CPU results](results/thrml_spw/spark_depth_20261106/cpu_vector_audit_v1.json) · [Frozen GPU follow-up plan](results/thrml_spw/spark_depth_20261106/reduced_work_GPU_plan_v1.json)

A separate CPU/meta inventory confirms the local LFM3 checkpoint has 1,728,801,536 parameters. Its BF16 weights exceed the current 2 GiB research GPU allocation cap, although its largest parameterized leaf is about 524 MB. An inference-only leaf offloader has passed CPU cleanup/alias tests and exact streamed-versus-resident BF16 CUDA comparisons on a tiny three-layer LFM3 model. The full original 1.7B checkpoint has now also passed execution preflight at 17, 65 and 1,024 tokens: all 287 parameterized modules were exercised, weights remained unchanged, and peak PyTorch GPU allocation was 824 MB. This execution path keeps BF16 weights on CPU; one-p-bit task-quality validation of the large model remains pending.

## Next work

1. Calibrate the LFM3 representation on training articles using the verified full-checkpoint execution path.
2. Freeze and run its original/mean/stochastic comparisons on the complete prepared language task.
3. Validate the reduced-work GPT-2 policy on the GPU with its frozen fresh-seed plan.
4. Extend full-model native/nonideal evidence and integrate verified results into the package without overclaiming hardware savings.

## Reproduction and evidence

Sources: [research/thrml_spw/spark_depth](research/thrml_spw/spark_depth).

- [Layer/head/neuron traces](results/thrml_spw/spark_depth_20261106/trace_analysis_v1/results.json)
- [Causal and sampling audit](results/thrml_spw/spark_depth_20261106/explanation_v1/results.json)
- [Activation-weighted operator preferences](results/thrml_spw/spark_depth_20261106/activation_variance_v1/results.json)
- [Complete five-seed confirmation audit](results/thrml_spw/spark_depth_20261106/confirmation_audit_v1/results.json)
- [Source snapshots](results/thrml_spw/spark_depth_20261106/source_manifest.json)

![Causal restoration and sensitive channels](results/thrml_spw/spark_depth_20261106/explanation_v1/causal_and_neurons.png)

![Five-seed confirmation](results/thrml_spw/spark_depth_20261106/confirmation_audit_v1/five_seed_confirmation.png)

## LFM3 training-only calibration and depth study

The original 1,728,801,536-parameter LFM3 checkpoint completed activation collection on eight training windows. The proposed policy assigns transposed G128 ranges to 42 linear tensors, original-layout G128 to 112, G7 to all 13 depthwise convolution tensors, and G16 to all 119 normalization tensors. All learned scalars remain represented; the tied embedding/head is counted once. This is a conditional-variance heuristic, not a measured end-to-end improvement.

An independent audit recomputed all 585 candidate scores from checkpoint weights and saved input moments with exact agreement. The first auditor incorrectly omitted attention-head multiplicity from normalization counts; its failed result is retained, and the corrected auditor accounts for 16 query heads and four KV heads. The calibration itself counted actual vectors correctly.

The running frozen training-cohort diagnostic compares original BF16, the finite-controller mean, and five independent K1 reconstructions at T=32768 on 32 training passages (8192 targets). It records each of the 24 block outputs and all 2048 hidden channels to identify error concentration and propagation. These measurements do not, by themselves, identify causal layer sensitivity or establish held-out quality. Full held-out LFM3 validation follows calibration and is still outstanding.

### LFM3 provisional first draw and package integration

The first complete stochastic draw on the 32-passage training cohort has perplexity **17.31328**, compared with original BF16 **17.32376** and finite-controller mean **17.31416**. It is only one seed and does not qualify the model. Its propagated block-output relative RMS error grows from about 2.3% to 5.0%, despite the similar aggregate task score. All five seeds have now completed; the complete result is reported below.

![One-seed LFM3 training diagnostics](results/thrml_spw/spark_depth_20261106/lfm3_first_seed_diagnostic_v1/provisional_layers.png)

A 55-condition causal plan is frozen: each of 24 blocks is independently made stochastic and independently restored within an otherwise stochastic model, followed by five component-family rescues. These diagnostic interventions deliberately mix deterministic and stochastic parameters and will not be reported as all-parameter K1 results. The complete 294,442-target LFM3 benchmark protocol is also frozen; its execution requires the complete five-seed training qualification and independent audit. Its selected-logit scoring path passed four tiny CPU BF16 semantic checks; full-model CUDA verification remains required.

The local SDK now exposes `FinitePBitProgram`: finite 12-bit control commands, FP32 group ranges, transposed matrix layout, aligned seven-weight convolution groups, and safe uint16/uint32 reconstruction counters. All five SDK/research fixtures match both count hashes and BF16-rounded weights, including three original LFM3 tensors. The development wheel and source distribution `0.1.0a2.dev0` built successfully; the installed wheel passed 18 package tests (one optional native module skipped). Artifacts are under `results/thrml_spw/spark_depth_20261106/sdk_distribution_v1/`. This development update is local and has not been published.

## Full-model Gaussian negative control

All 11 GPT-2 CPU training conditions completed: original BF16-rounded weights in FP32 arithmetic, five independent p-bit reconstructions, and five independent Gaussian perturbations. Both perturbations use the same finite-controller mean and coordinate-wise variance at T=32768. All 740 paired parameter variance totals match exactly, and three Gaussian reconstructions independently replay bitwise. The 1024-target training result is:

| Noise law | Perplexity ratio to CPU reference | 95% Monte Carlo interval |
|---|---:|---:|
| Bernoulli count reconstruction | 0.996880 | 0.984085–1.009840 |
| Variance-matched Gaussian | 1.004059 | 0.988554–1.019808 |

This supports testing generic neural noise tolerance as an explanation; it establishes neither statistical equivalence nor a Bernoulli-specific advantage. It is a small training diagnostic, not a new GPU or held-out benchmark. See `results/thrml_spw/spark_depth_20261106/gaussian_control_v3_audit.json`.

The serial local LFM3 continuation is now live. It waits for the current training process, audits the full cohort, performs 55 isolated/restored layer and family interventions with six full-model neural replays, and starts the complete validation benchmark only if the training criterion passes. The final benchmark uses the original full-sequence output head to keep its arithmetic consistent with training; an earlier unexecuted selected-head optimization protocol is retained but superseded. No duplicate GPU jobs should be started while this queue is live.

## Full-model calibration error and correction ceiling

A five-seed, 16-condition GPT-2 CPU experiment applies independent device gain SD 0.02 and logit-offset SD 0.02 to all programmed weights. On 1024 training targets, uncalibrated perplexity is 1.03743 times the reference (95% Monte Carlo interval 1.00595–1.06991). The paired ratio to ideal stochastic devices is 1.03229 (1.00038–1.06522). Oracle correction, using true device coefficients and requantizing commands, gives 0.99643 times the reference (0.98501–1.00798). These illustrative device parameters are not measurements from hardware, and the broad intervals do not establish a deployable quality guarantee.

Oracle correction reduces systematic weight-error SSE from about 1695.16 to 3.79, a factor of 447.3. Ordinary temporal averaging cannot remove this systematic bias. Retaining two FP32 calibration coefficients for every GPT-2 parameter would add 995,518,464 bytes; a real design must decide where and when such calibration information is stored. All 16 saved scores, matched device-characteristic arrays, and nine independently reconstructed parameter cases passed the audit.

![Calibration error and oracle ceiling](results/thrml_spw/spark_depth_20261106/gpt2_calibration_analysis_v1/device_error.png)

The lower-work GPU runner is implemented and prospectively frozen, but has not yet run. It will evaluate all 283225 targets using five fresh seeds plus a fresh BF16 reference, preserving the earlier 0.58324 observation-budget fraction and T_max=32768 policy. The full LFM3 benchmark auditor is also prepared; its synthetic fixtures reject duplicate, omitted, and invalid-context targets and reject a favorable mean whose uncertainty exceeds the quality margin. Passing these fixtures is not a substitute for auditing the future real outputs.

## Completed LFM3 training qualification

All seven full-model cases finished on 32 training passages (8192 targets). Original BF16 perplexity is **17.3237567**; the finite-controller mean is **17.3141595**; the stochastic cohort perplexity at mean NLL is **17.3550744**. The ratio to BF16 is **1.00180779**, with 95% Monte Carlo interval **[0.99984757, 1.00377184]**, passing the frozen training qualification. This is not yet a held-out LFM3 benchmark result.

Independent checks verified all 32 input windows in every case, every one of the 286 parameter policies, all sample budgets, and all aggregate weight-error moments. The standardized deviations from predicted aggregate SSE ranged from about -1.10 to +1.33 across the five draws. The layer/channel score audit also passed. The causal stage is now running: it reconstructs the first weight draw for independent replay, then performs the 55 interventions. Full-dataset validation follows automatically if that stage completes.

![Completed five-seed LFM3 training diagnostics](results/thrml_spw/spark_depth_20261106/lfm3_training_depth_analysis_v1/layer_channel_error.png)

## Completed LFM3 causal layer study

All 55 fixed-draw training interventions completed and their saved scores were independently recomputed. Every isolated-layer intervention left all earlier block outputs exactly unchanged. The first stochastic weight point reproduced all 286 count records, and six full-model neural replay windows matched the original saved losses bitwise.

The ranking depends on the intervention: block 3 caused the largest isolated NLL increase (0.002879), while restoring block 2 within the fully stochastic model gave the largest layer-rescue improvement (0.005668). These are observations from one fixed draw on 1024 training targets, not population-level sensitivity rankings. The mixed deterministic/stochastic rescue cases are diagnostic controls and do not count as all-parameter K1 results.

![Causal LFM3 interventions](results/thrml_spw/spark_depth_20261106/lfm3_causal_depth_analysis_v1/causal_layers.png)

![Downstream response to isolated perturbations](results/thrml_spw/spark_depth_20261106/lfm3_causal_propagation_v1/propagation_matrix.png)

The baseline activation survey covers 8224 positions at each of the 24 blocks. Signal energy is uneven across channels: its participation ratio ranges from roughly 9 to 654 out of 2048 channels. This statistic is not a neuron-count reduction or a pruning result. The correlation measurements describe baseline activations, not stochastic-error covariance.

![LFM3 neuron activation structure](results/thrml_spw/spark_depth_20261106/lfm3_neuron_structure_v1/neuron_structure.png)

Relative-error dips must be interpreted carefully. From block 11 to block 12, absolute error power increases by 20.8%, while signal power increases by 57.0%; relative RMS error therefore falls by 12.3%. This dip does not establish denoising.

![Absolute and relative LFM3 error](results/thrml_spw/spark_depth_20261106/lfm3_error_propagation_v1/absolute_and_relative.png)

The full 294442-target LFM3 benchmark is now running. Its final independent audit is prepared, and a separate SDK-based reconstruction and 21-window neural replay protocol has been frozen. No full-task LFM3 quality result is accepted until those stages complete.

## Accepted full LFM3 benchmark

All seven cases on the complete 62-article, 294442-target benchmark finished. Original BF16 perplexity is **18.9301548**; the selected five-seed cohort perplexity at mean NLL is **18.9791768**. Its ratio is **1.00258963**, with the frozen 95% Monte Carlo interval **[0.98751035, 1.01789916]**, passing the 2% criterion. The joint seed/article bootstrap interval is **[0.99188281, 1.01065363]**.

The complete saved-output/coverage/statistical audit passed. A separate SDK reconstruction reproduced all seven full weight points, all 1430 stochastic parameter count hashes, and all 21 prespecified neural replay windows bitwise. This accepts the scoped R1 ideal-independent-sampling result. It does not establish native full-model THRML execution or physical-device performance.

![Three accepted model/task checks](results/thrml_spw/spark_depth_20261106/cross_model_summary_v2/tested_model_quality.png)

The three checkpoint/task evaluations all pass their own frozen criteria. They cover two previously inspected benchmark tasks and do not constitute proof of universal numerical BF16 equivalence or unseen-task generalization.

## Stronger spatial control changes the cost comparison

The earlier K4 word baseline carries four fixed numerical bits, whereas K1 uses a 12-bit probability command. A new native THRML control combines three clamped coarse bits with one programmable fractional p-bit, also using a 12-bit probability command. Its normalized value is `(q + z)/8`, where the three-bit integer q sets the coarse level and z supplies the fractional probability. This is a stronger internal baseline, not a claim about Extropic's proprietary implementation.

For the reoriented real GPT-2 head bank, K1 at T32768 measured mean output MSE **7.0041e-5**. The stronger K4 control at T8192 measured **3.3115e-6** at the same 201326592 scheduled node observations, using four times as many physical elements. Only one of its four elements is free to fluctuate, so its free-node updates are 50331648; clamped-node reads must not be charged as random transitions. K4 at T32768 measured **5.6909e-7** with equal free-node updates and four times the scheduled reads.

All 30 native cases passed an independent decode/statistics audit, and three native trajectories replayed bitwise. Exact binomial moment analysis accounted for correlations between input vectors that share sampled weights; all weight and output-MSE moment checks passed. The MSE-variance formula was independently checked by enumerating all 16 outcomes of a small binomial fixture.

This comparison demonstrates a spatial/temporal tradeoff, not general K1 efficiency dominance. The richer K4 representation also carries three coarse command bits per parameter. Counting those commands, a 12-bit fractional command and four binary states separately gives 19 bits per parameter before ranges, versus 13 for K1. Actual hardware may store clamp information differently; energy, area and readout costs remain unmeasured. Earlier favorable comparisons with the four-bit-word reference remain valid for that reference and do not generalize to all multi-element encodings.

The lower-work GPT-2 full-task run is now active. A final five-draw LFM3 covariance study is queued afterward to measure correlated neural errors, separately from the baseline activation correlations already reported.

## Lower-work full-task result (replay pending)

The fresh five-seed lower-work GPT-2 cohort completed all 283225 targets. Its perplexity ratio to the fresh BF16 reference is **1.00779677**, with 95% Monte Carlo interval **[0.99906888, 1.01660092]**, passing the frozen 2% criterion. The output, target-coverage and resource-budget audit passed. It uses 58.324% of the uniform policy's nominal sampling observations, adds 53088 bytes of range metadata, and keeps maximum T=32768. This reduces logical sampling work by 41.676%; it does not establish the same reduction in inference latency or physical energy. Independent SDK/neural replay remains pending behind the covariance GPU job.

## Cumulative GPT-2 layer controls

A separate CPU training study completed 41 conditions: one deterministic baseline and eight cumulative prefix/suffix/all-block/all-parameter conditions across five seeds. All saved scores and conversion masks were audited, the first weight draw was reconstructed with the SDK, and six independent neural rows replayed bitwise. The curves show no simple monotonic loss increase with converted block count within their wide Monte Carlo intervals. Nonblock parameters stay deterministic in the prefix/suffix curves; these are partial-conversion diagnostics, not all-K1 held-out quality claims.

![Cumulative block substitutions](results/thrml_spw/spark_depth_20261106/gpt2_cumulative_summary_v1/cumulative_blocks.png)

The accepted-core archive is prepared locally under `publish/pbit1-experiments/updates/20261106_spark_core/`. Its 2421 files, including 1188 immutable source snapshots, passed hash verification. All 63 checked protocol source references resolve to archived bytes. Raw model/data arrays are omitted with hashes and explicit reproduction limitations. Later lower-work and covariance results are intentionally outside that frozen core snapshot.
