# Single-P-Bit Weight Encoding experiments

## LFM3 research: one p-bit per parameter

The target is **one logical programmable p-bit for each of LiquidAI/LFM3-1.7B's 1,728,801,536 unique learned parameters**, evaluated against BF16 task quality. More temporal samples do not add more p-bits per parameter. Four- and eight-p-bit encodings are comparison controls. Signed weights use shared local ranges; finite programming controls, range metadata, and accumulators are counted separately.

The initial frozen five-seed THRML study covered every parameter on 512 fixed language targets. Mean NLL was **4.9282 versus BF16's 4.6357**, or subset perplexity **138.13 versus 103.10**. That configuration misses BF16 quality. Its [independent statistical audit](results/thrml_spw/lfm3_1p7b/language_native_v2/statistics_audit.json) remains part of the record; this is a conditional subset evaluation, not a full WikiText benchmark.

Both 16-update adaptation arms and the unadapted control passed independent trajectory/export audits. The prior-selected mean-adapted one-p-bit cohorts are now complete: five seeds per sampling budget on all 2,048 validation targets. At **T1024**, mean perplexity is **83.9167 versus same-fit BF16's 75.3147** (11.42% worse). At **T4096**, it is **76.1024** (1.05% worse), with a conditional Monte Carlo 95% perplexity-ratio interval of **0.9988 to 1.0223**. The upper bound is 2.23% worse, narrowly exceeding the unchanged 2% criterion. Both gates therefore fail; neither a confidence interval crossing one nor a favorable mean proves equivalence. [Completed-cohort root review](results/thrml_spw/lfm3_1p7b/completed_onebit_cohorts_root_review.json). **BF16 task-quality equivalence remains unproven**, and no numerical BF16 equivalence is claimed.

| Track | Protocol or accepted evidence | Current state source |
| --- | --- | --- |
| Model and coverage | [Pinned inventory](results/thrml_spw/lfm3_1p7b/meta_inventory.json) | [Reviewed preparation](results/thrml_spw/lfm3_1p7b/preparation_status.json) |
| Primary mean-adapted one-p-bit study | [T1024 protocol](results/thrml_spw/lfm3_1p7b/counted_mean_validation_protocol.json) | [Completed cohort](results/thrml_spw/lfm3_1p7b/counted_mean_validation/pipeline.json) |
| Conditional higher sampling, still one p-bit | [T4096 protocol](results/thrml_spw/lfm3_1p7b/counted_mean_t4096_protocol.json) | [Completed extension](results/thrml_spw/lfm3_1p7b/counted_mean_t4096/pipeline.json) |
| New higher sampling, one p-bit per parameter | [T8192 protocol](results/thrml_spw/lfm3_1p7b/counted_mean_t8192_protocol.json) | [Five fresh seeds running](results/thrml_spw/lfm3_1p7b/counted_mean_t8192/pipeline.json) |
| Original three-arm native validation | [Protocol](results/thrml_spw/lfm3_1p7b/adaptation_native_v2_protocol.json) | [Pipeline](results/thrml_spw/lfm3_1p7b/adaptation_native_v2/pipeline.json) |
| Sampling allocation at the same learned point | [Protocol](results/thrml_spw/lfm3_1p7b/allocated_native_protocol.json) | [Pipeline](results/thrml_spw/lfm3_1p7b/allocated_native/pipeline.json) |
| Original frozen-checkpoint eight-p-bit comparison | [512-target protocol](results/thrml_spw/lfm3_1p7b/word8_language_protocol.json) | [Pipeline](results/thrml_spw/lfm3_1p7b/word8_language/pipeline.json) |
| Matched adapted-model K4/K8 controls | [Same-point, same-target, matched-work protocol](results/thrml_spw/lfm3_1p7b/adapted_word_validation_v2_protocol.json) | [Recovery pipeline](results/thrml_spw/lfm3_1p7b/adapted_word_validation_v2/pipeline.json) |
| Count-based software observer | [Strict whole-model replay](results/thrml_spw/lfm3_1p7b/counted_native_replay/initial_replay/independent_audit.json) | [All-stage bit-pattern review](results/thrml_spw/lfm3_1p7b/counted_native_replay/root_replay_crosscheck.json) |
| Spatial-control worker integration | [GPU trajectory/operator audit](results/thrml_spw/lfm3_1p7b/adapted_word_worker_gpu_preflight/independent_audit.json) | Full adapted-model quality pending |
| Untouched article-separated holdout | [Data audit](results/thrml_spw/lfm3_1p7b/holdout_data/independent_audit.json) | [Pre-score eligibility rule](results/thrml_spw/lfm3_1p7b/holdout_treatment_selection_v2_policy.json); all 10,240 targets unscored |

A new T8192 condition is now running five fresh seeds at the same point and the same 2% criterion. It uses twice T4096's retained sampling work and eight times T1024's; all previous results remain intact. The holdout eligibility rule was explicitly versioned to include it before holdout scoring, which still requires a complete execution protocol. The T4096 extension ran after the complete T1024 cohort missed the fixed quality criterion. It used four times the retained scheduled node evaluations; warmup, programming controls, and ranges are separately counted. The adapted K4/K8 comparison uses the same learned point and targets, with actual padding-inclusive work checked against the corresponding one-p-bit run; it is running under the original memory guards after recovery of its upstream statistics bookkeeping. The original 512-target frozen-checkpoint K8 study cannot serve as that adapted-model comparator. All original K8 native cases passed. Its final audit stopped because a live pipeline-file hash was incorrectly treated as immutable. The [corrected immutable-snapshot statistics audit](results/thrml_spw/lfm3_1p7b/word8_language_statistics_v2/statistics_audit.json) passes with identical numerical results and no neural sampling rerun; the original failures remain preserved.

The count-based observer passed a strict same-key full-model replay before its separate study began. It uses temporary per-weight counters; this does not establish a physical output-only datapath, reduced sample requirements, or hardware efficiency. Original studies retain their frozen workers and settings. [Detailed research log](research/thrml_spw/RESEARCH_LOG.md) and [LFM3 progress](results/thrml_spw/LFM3_ADAPTATION_PROGRESS.md) record accepted results, negative findings, and pending requirements. Executed sources and accepted evidence are immutable; each extension has its own namespace.

One logical p-bit per parameter describes device assignment. The simulator reuses bounded banks and retains programming controls, shared ranges, counters, and reference weights. [Resource accounting](results/thrml_spw/lfm3_1p7b/language_resource_model/independent_audit.json) distinguishes those costs. Physical energy, latency, full device residency, and a hardware systems advantage remain unmeasured.

Read current state without launching or restarting an experiment, from the repository root:

```bash
cat results/thrml_spw/lfm3_1p7b/counted_mean_validation/pipeline.json
cat results/thrml_spw/lfm3_1p7b/counted_mean_t4096/pipeline.json
cat results/thrml_spw/lfm3_1p7b/counted_mean_t8192/pipeline.json
cat results/thrml_spw/lfm3_1p7b/adapted_word_validation_v2/pipeline.json
cat results/thrml_spw/lfm3_1p7b/word8_language/pipeline.json
```

For reproduction, use source hashes and pinned inputs in each protocol, with archived bytes under `results/thrml_spw/source_snapshots/`. Torch drivers use `.venv`; official THRML GPU workers use `.venv-thrml-gpu` through `research.thrml_spw.gpu_environment`. Fixed output namespaces use exclusive creation, so reproduce in a separate workspace. Raw public-state replay is retained for component checks; complete LFM3 trajectories are not all saved or independently replayed. The reserved holdout requires a separate full execution/statistics protocol before any model scoring.

## Historical milestone entries

The entries below preserve earlier checkpoints. Use the current state sources above for the LFM3 study.


Current run state and reviewed findings: [research status](results/thrml_spw/STATUS.md). The milestone entries below are historical snapshots; use the linked live pipeline for current counts.

Full probability-training reproduction passed in a fresh process: all ten fits, complete epoch-history bytes, selection decisions and 240 exported arrays match exactly, including 2 million validation logits. Same-code reproducibility only; no new independent seeds or native quality result. [Reproduction report](results/thrml_spw/FASHION_CNN_PROBABILITY_TRAINING_REPRODUCTION_REPORT.md). Full 300-case THRML evaluation remains live; goal active/global index stale.

Matched CNN comparison fully audited: at equal stochastic-node evaluations, one-p-bit accuracy is 66.0010/78.5182/83.0638% versus four-p-bit 83.0896/83.1714/83.2036%. Largest-budget accuracy and CE difference intervals include zero; no equivalence or systems-win claim. [Reviewed comparison](results/thrml_spw/FASHION_CNN_MATCHED_WORD_REVIEW.md). All 30 learned-probability native validation cases also pass; the full 300-case fine-tuned native task evaluation is now running. [Learned-command validation](results/thrml_spw/FASHION_CNN_PROBABILITY_NATIVE_REPORT.md), [live task cohort](results/thrml_spw/fashion_cnn_native/probability_cohort_evidence/pipeline.json). Goal active/global index stale during live writes.

All 150 four-node positional CNN production cases completed successfully: 223,776,000,000 scheduled node updates. Every saved case hash, shape, finite logit array and clipping-metadata record has been reviewed without computing task metrics. Complete independent public-state replay is now running; accuracy comparisons remain pending replay/statistics. [Production review](results/thrml_spw/fashion_cnn_native/word_comparison_evidence/production_root_review.json), [live comparison](results/thrml_spw/fashion_cnn_native/word_comparison_evidence/pipeline.json). Both learned-probability follow-ups remain actually queued behind the required audits. Goal active/global index stale during live writers.

Full native probability-finetuned CNN evaluation is source-frozen and actually queued: 300 cases across five models/two training methods/three budgets/ten inference seeds, complete public-state recovery and all 33 descriptive contrasts. Native validation must pass first. Deterministic control and cross-environment bootstrap prerequisites pass; no new task results yet. [Full evaluation protocol](results/thrml_spw/FASHION_CNN_PROBABILITY_COHORT_PROTOCOL_REPORT.md). Capacity forecast includes all queued outputs; goal active/global index stale during live writes.

Learned-probability CNN native validation is now source-frozen and actually queued behind the matched four-node full audits: all ten checkpoints, T64/256/1024, two eight-image validation banks, complete independent public-state recovery. All 26,640 commands and fixed ranges/biases statically checked; no native validation or new quality scores yet. [Native validation protocol](results/thrml_spw/FASHION_CNN_PROBABILITY_NATIVE_PROTOCOL_REPORT.md). Goal active/global index stale during writes.

CNN probability-aware fine-tuning preparation complete: five models × two methods, fixed ranges/biases and learned finite-field commands. Selected validation outputs/exports independently replayed (2 million logits exact). This is Torch surrogate training; native confirmation and test quality remain open. [Preparation report](results/thrml_spw/FASHION_CNN_PROBABILITY_FINETUNE_PREPARATION_REPORT.md). Matched four-node CNN production continues unchanged. Goal active; global index stale during live writes.

Latest completed CNN result: all 150 native cases now pass full public-state replay and independent control/statistical audits. At 1024 samples/layer accuracy is **83.0638% versus 85.0860% FP32**, a **2.0222 percentage-point loss**; the approximate 1–2 point quality target is not declared passed. FP16/BF16 controls are also fully audited and remain near baseline. All 7,077,375,000 native counter entries match public recovery. The matched four-node CNN pipeline has started GPU validation; its full quality comparison remains pending. [Reviewed CNN result](results/thrml_spw/FASHION_CNN_NATIVE_REVIEW.md), [precision controls](results/thrml_spw/FASHION_CNN_PRECISION_REPORT.md), [live matched comparison](results/thrml_spw/fashion_cnn_native/word_comparison_evidence/pipeline.json). Goal active; global artifact index stale while writers remain live. Earlier entries below are historical checkpoints.

All150 direct integer-activation classifier cases now pass full causal/count/statistical audits: one free p-bit per each of2368matrix weights, no activation p-bits or per-weight operational reconstruction counters. AtT1568 accuracy is89.6378% versus89.8800%FP32 and89.6428%same-trajectory reconstruction. All70875000counters match exactly. [Reviewed direct result](results/thrml_spw/INTEGER_ACTIVATION_CLASSIFIER_REVIEW.md). The earlier300-case Boolean comparison and static-cache control remain independently audited. The full CNN has passed GPUvalidation and50deterministic controls and is actually producing cases(24/150at this checkpoint); FP16/BF16evaluation is complete and waits for full parent audits before comparison. [CNN GPU validation](results/thrml_spw/FASHION_CNN_GPU_VALIDATION_REPORT.md), [live CNN pipeline](results/thrml_spw/fashion_cnn_native/cohort_evidence/pipeline.json).

A bounded native convolution operator now passes all12 cases, including signed ranges, stride, dilation and exact masked padding:5904 counters agree with independent public states and separate spatial-loop outputs agree within4.9e-15. This is an operator prerequisite, not full CNN accuracy. [Convolution report](results/thrml_spw/CONVOLUTION_NATIVE_REPORT.md).

Five conventional Fashion-MNIST CNN checkpoints are now frozen and independently checked. A native three-layer causal validation witness passes all15 cases/566700 exact public-state counters, with accumulation before ReLU/maxpool. No test accuracy has been scored for this cohort; complete full-test native production/control/statistical protocols and a capacity-bounded replay format remain to implement. [CNN preparation report](results/thrml_spw/FASHION_CNN_PREPARATION_REPORT.md).

CNN capacity preparation now passes exact public recovery of all420bounded validation arrays and actual vectorized-native replay. Compact counter/output receipts plus final logits require204MB before supporting artifacts for the planned150-case cohort; hashes require complete sampler regeneration and cannot decode counters themselves. Full-budget GPU validation is source-sealed and actually queued behind the integer classifier; full CNN test scoring is still unstarted. [CNN recovery report](results/thrml_spw/FASHION_CNN_COMPACT_REPORT.md).

The full150-case native CNN cohort is now source-sealed before test scoring and its real runner waits behind the GPU prerequisite. All5models/10MCseeds/T64,256,1024, ten deterministic control conditions, complete public replay and independent statistics are fixed. Planned production223776000000updates and matching audit are not yet executed; full CNN quality is pending. [Full cohort protocol/report](results/thrml_spw/FASHION_CNN_COHORT_PROTOCOL_REPORT.md).

The CNN FP16/BF16 supplement is separately source-sealed and actually queued:20validation paths pass, distinguishing rounded weights/F64arithmetic from pinned TorchCPU low-precision execution. It adds no native draws and waits on parent controls/full audits before evaluation/comparison. [Precision protocol report](results/thrml_spw/FASHION_CNN_PRECISION_PROTOCOL_REPORT.md).

Full CNN conventional-control review passes all50paths/five millionlogit values with no prediction mismatches under the fixed numeric conditions. The four-node positional CNN reference also passes25native/public validation cases,944500integer counters and700byte-exact bank arrays. It is an internal reference; GPU/full matched-resource test comparison remains pending. [Four-node witness](results/thrml_spw/FASHION_CNN_WORD_REFERENCE_WITNESS_REPORT.md).

The full150-case four-node positional CNN comparison is now source-sealed and actually queued behind the one-node full audit. WordT16/64/256 match one-nodeT64/256/1024 in stochastic-node evaluations. GPU proof, complete native/public cases and paired-model/independent-MC statistics are fixed before word test scoring; no word full-test quality yet. [Matched comparison protocol](results/thrml_spw/FASHION_CNN_MATCHED_WORD_PROTOCOL_REPORT.md).

All150one-p-bit native CNN production cases completed with223776000000scheduled updates and exit0. Every case hash/schema is reviewed; complete public-state causal replay is actually running(21/150at this checkpoint). CNN task quality remains pending full replay/statistical audits. [Production review](results/thrml_spw/fashion_cnn_native/cohort_evidence/production_root_review.json), [live pipeline](results/thrml_spw/fashion_cnn_native/cohort_evidence/pipeline.json).

Fully audited matched-update neural comparison: one p-bit per normalized converted
weight at T256 gives PPL144.850 versus139.959 for four positional p-bits/T64 and
134.587 FP32. Accuracies are22.402%,22.305%,22.949%; the paired accuracy interval
includes zero. One node uses one quarter the dynamic weight nodes, four times the
sample rounds and a larger control/range payload. All10 model paths,960 native
bank outputs and60,000 resampling values are independently checked. This converts
51.84% of one checkpoint on consumed short validation contexts; no physical
systems advantage or Extropic architecture disproof follows. [Report](results/thrml_spw/POSITIONAL_STUDENT_REPORT.md).

The long-horizon goal remains active. The following matched-update Transformer cohort is closed; the full Boolean classifier study linked above is live. Broader coverage and measured device/controller costs remain open. Earlier entries below are retained as historical context.

New native Boolean learned-layer path independently verified:58.4million integer
AND/weight/popcounts match public raw-state replay exactly. An explicit activation
range restores standardized negative inputs; the first clipped-input attempt stays
failed. Across five learned first layers and fixed budgets, activation sampling
adds about4.60times the matched-control sampling MSE and25% extra activation
nodes for eight simultaneous inputs. This is learned-layer arithmetic, not new
classifier accuracy or physical efficiency. [Report](results/thrml_spw/BOOLEAN_AFFINE_LAYER_REPORT.md).

Finite-control information analysis now independently audited:112 conditions,
including a matched16-level scalar prior, quantify how sampling reveals a
programmed probability. This is analytical evidence, not new neural or hardware
quality. The source/audit failure and corrected derivative are retained.
[Information report](results/thrml_spw/INFORMATION_BUDGET_REPORT.md).

Finite-field spatial comparison now independently audited: 33,924,000 native
count elements match public THRML replay exactly; all44 scalar theory/error rows
pass. At equal updates, four positional p-bits give lower scalar error than one
p-bit throughout tested budgets. One p-bit overtakes two-bit quantization at
longer sampling budgets. This is an internal positional reference, not an asserted
Extropic architecture, neural comparison or hardware advantage. [Report](results/thrml_spw/FINITE_POSITIONAL_REPORT.md).

Conventional controls independently checked on the same consumed text: zero-preserving
INT4/group16 gives perplexity **132.298**, versus **134.816** for SPW/group16/T512
and **129.541** FP32. Nominal code/range/bias payload is 6.28125 versus 12.03125
bits per converted weight. All 21 deterministic variants, their token metrics,
891,813,888 reconstructed weight coordinates and 440,000 paired resampling values
were checked. The encoded-mean SPW reference gives PPL 129.492; finite sampling
remains consequential. This diagnostic uses no fresh confirmation text or native
samples and does not change the failed SPW strict gate. The encoding works in
THRML; a hardware systems advantage remains unproved. [Control report](results/thrml_spw/FINE_RANGE_QUANTIZATION_CONTROLS_REPORT.md).

Audited five-seed end-to-end comparison: finer ranges at T512 give perplexity
134.816 versus 136.834 coarse T1024 and 129.541 FP32. Fine next-token accuracy
loses 0.430 percentage points versus FP32. The fine point quality rule passes,
but its strict rule fails (perplexity upper sensitivity +5.96% versus the 5% limit).
The paired CE interval includes zero; a paired improvement is not confirmed.
Fine halves observed node updates with 41% more converted-weight control/range
payload. All 240 native operators, ten complete model paths and 60,000 resampling
values passed independent checks. This converts 51.84% of one model on short
teacher-forced contexts. [Full report](results/thrml_spw/FINE_RANGE_CONFIRMATION_REPORT.md). The broad goal remains active.
[Restore instructions](results/thrml_spw/lossless_archive/README.md) apply to archived historical arrays.

New local-range development study: group16/T512 gives raw geometric operator MSE
ratio 0.670558 versus group128/T1024, across all 24 projections,
five seeds and 360 native operators on consumed validation inputs. All 72 theory
checks pass the frozen 20% tolerance, and every independent frequency reference
satisfies its IEEE envelope. The strict byte-exact F32 replay gate **fails in
270/360 cases**; this is preserved, not relabeled as a fully passed audit.
The control/range payload rises from 8.53125 to 12.03125 bits per converted weight
while observed node updates halve. At that development checkpoint, full stochastic neural quality was untested
and storage unavailable. The later confirmation above supplies that missing test;
the original operator-only scope and failed numerical gate remain unchanged.
[Local-range operator report](results/thrml_spw/RANGE_LOCALIZATION_REPORT.md). The broad research goal remains active.

New independently audited validation-selected allocation matches uniform sampling's
weighted update budget exactly but gives no confirmed quality benefit. Held-out
perplexity is 169.960 allocated versus 169.775 uniform T1024 (FP32 159.794);
the paired CE sensitivity interval includes zero and both 5% quality gates fail.
All 120 validation/240 test native operators, ten full test cases and 60,000
bootstrap values passed independent native/model/score checks. Six layers change
budget; serialized layer sampling rounds fall 8.33% while maximum layer T doubles.
These are counters, not a physical latency/energy improvement. All six blocks use
one free p-bit per normalized weight (51.84% of one checkpoint's unique parameters).
[Validation-selected allocation report](results/thrml_spw/SAMPLE_ALLOCATION_REPORT.md). The broad goal remains active.

New independently audited fixed-regime scaling test converts all six Transformer
blocks' 24 affine matrices: 42,467,328 weights, 51.84% of unique parameters.
At ideal finite8/T1024, perplexity is 131.834 versus 127.513 for the shared
first-three-block prefix and 123.754 FP32. All24 point gate: False; strict gate:
False. Five seeds, 32 fresh short-context windows, 120 operators and both
paired neural paths were independently replayed and scored. One fixed frequency
tolerance failure has byte-exact publicF32/roundoff evidence; the original and tighter
compensation failures remain explicit. Embeddings/head
remain deterministic; this is broader conversion, not full-model or physical proof.
[All-six-block scaling report](results/thrml_spw/ALL_BLOCKS_REPORT.md). The broad goal remains active.

New independently audited matched-observation experiment reduces calibrated signed-weight
MSE by 25.24% with probe locations chosen only from nominal weights before sampling.
On 4,096 fresh text targets, Transformer perplexity is 139.133 versus 139.798 for fixed
probes (fresh FP32 133.043). The paired CE sensitivity interval includes zero, and
both methods fail the predeclared strict 5% quality gate. The candidate meets only
the point rule. All 240 probes, 120 operators, ten full model cases and regenerated
input hashes passed independent audits. This converts 25.92% of one checkpoint;
physical energy/latency advantages and a confirmed neural improvement remain unproved.
More central probes increase a post-hoc analytic recorded-state-change proxy despite
equal observation counts; that proxy is not a physical energy measurement.
[Weight-only probe placement report](results/thrml_spw/PROBE_PLACEMENT_REPORT.md). The broad goal remains active.

New audited diagnostic identifies calibration estimates as the main limitation in
the tested Transformer regime. Keeping eight-bit control range, T1024, devices and
sampling keys fixed, known-parameter controls score perplexity 131.676 versus
135.426 measured calibration and 131.597 ideal devices. All 60 new native operators,
five complete model forwards and 16 mean-weight diagnostics passed independent audits.
[Calibration-estimation diagnostic](results/thrml_spw/TRANSFORMER_CALIBRATION_FLOOR_REPORT.md).
This uses the same 25.92% converted subset and consumed text. True device parameters
are granted to an oracle, so it is not a deployable fix or new confirmation. The old
failed 5% measured-calibration regime remains unchanged. The broad goal stays active.

New audited pretrained-model result: measured calibration reduces perplexity from
1,137.355 to 135.426 under synthetic static gain/offset errors; FP32 is 127.140.
The calibrated result remains 6.52% above FP32 and fails the fixed 5% quality limit.
The paired ideal-device case scores 131.597 (+3.51%) and passes both frozen rules.
All 15 sequential quarter-model runs, 180 native operators and 120 probe windows
passed independent sampling, model-input/loss, score and bootstrap audits.
[Calibrated Transformer report](results/thrml_spw/CALIBRATED_TRANSFORMER_REPORT.md).
This converts 25.92% of one DistilGPT-2 checkpoint on 4,096 new short-context targets.
Calibration adds 50% observed updates for this stream; finite control/range/bias
payload is 8.53125 bits per converted weight. These are not physical hardware costs.
The failed calibrated regime is retained. The broad goal remains active.

New audited training result: five paired starting models now optimize probability
weights with native THRML sampling and finite eight-bit controls. Device-augmented
fine-tuning reaches 77.256% versus 75.531% for expected-weight fine-tuning on reserved
uncalibrated device profiles at T64: +1.726 points with approximate model-level normal
margin 0.528. All 10,750 native training updates, 450 validation/profile evaluations
and 600 full-test cases passed independent replay, gradient/update and score audits.
[Five-seed training report](results/thrml_spw/PROBABILITY_FINETUNE_REPORT.md).
The benefit is conditional: calibrated T256 accuracy is 88.428% for device-augmented
training versus 88.741% for the expected-weight control. Original FP32 is 89.880%.
Checkpoint selection used validation only; test device/MC seeds were reserved, while
MNIST images were already used in prior work. No hardware advantage is established.
The broad long-horizon goal remains active.

New audited drift result: complete networks now keep every matrix weight as a free
THRML p-bit while bias changes during calibration, inference and idle time. At the
slower synthetic drift rate, scheduled refresh scores 88.540% versus 81.463% with
initial calibration retained (89.880% FP32). At the faster rate it scores 78.223%
versus 49.043%, preserving a substantial quality failure. All 150 full-test runs,
750 calibration events and 1,550,400 field hashes passed independent audits and
public-factor replay. Refresh needs 11.356% more active observed node updates.
[Clocked full-network report](results/thrml_spw/CLOCKED_MNIST_REPORT.md).
Comparison clocks match by inserting idle into the retained policy; equal hardware
energy/latency is not established. The goal stays active for validation-selected
quality/cost policies, nonideal-aware training, larger conversion and device measurements.

New audited result without exact endpoint clamps: all 2,368 weight coordinates
are freely sampled native THRML p-bits. With measured calibration, finite eight-bit
bias controls and T1024 per layer, five learned MNIST models score 89.334% versus
89.880% FP32. The paired clamped control scores 89.363%; the endpoint contrast is
-0.029 points with approximate five-model normal margin 0.122. All 1,600 full-test
evaluations and 400 probe runs passed raw/control audits and all-example public
THRML replay. Finite biases approximate zero/one; they cannot represent them exactly.
[Finite-endpoint report](results/thrml_spw/FINITE_ENDPOINT_MNIST_REPORT.md).
Removing clamps costs 18.92% more active updates at equal sampling depth. This is
synthetic static-device evidence on an existing test set, not hardware performance.
The long-horizon goal remains active for clocked networks, training and broader conversion.

New audited full-network result: five learned direct-p=w MNIST models retain
89.371% accuracy after measured finite-control calibration under synthetic static
gain/offset errors, versus 89.880% FP32 (0.509-point loss) at T1024 per layer.
All 750 complete test evaluations, 200 probe runs and every example in both layers
passed independent score/control and public-factor path audits. UINT4 scores
89.336%; the small native point difference is not a hardware or quantization win.
[Full-network calibration report](results/thrml_spw/CALIBRATED_MNIST_REPORT.md).
The existing test set and exact endpoint modes are explicit; the goal stays active.

Earlier audited drift result: bias changes during both calibration and inference.
In the faster synthetic drift case at the late query, refresh reduced operator MSE
from 0.3475 to 0.0521, but did not recover the zero-drift baseline near 0.0187.
It requires another 4,096 probe samples per free device. All 90 calibration events,
150 query windows and 6,960 active field hashes passed independent clock/path audits.
These are fixed counterfactual queries, not a hardware refresh interval in seconds.
[Clocked drift report](results/thrml_spw/DRIFT_RETENTION_OPERATOR_REPORT.md).

Earlier audited comparison: calibrated one- and four-p-bit operators use equal total
calibration and inference draws. At the largest tested probe budget, output MSE is
0.018626 versus 0.018672. The paired uncertainty interval includes zero; this is
not formal equivalence. One p-bit uses four times fewer logical nodes and stored
commands, with four times the sampling depth. All 100 inference and 120 probe runs
passed independent audits. These are synthetic static device errors on one old
operator input tensor, not full-model or physical performance.
[Calibrated matched-resource report](results/thrml_spw/CALIBRATED_MATCHED_OPERATOR_REPORT.md).

Earlier audited result: two-probe calibration handles synthetic gain plus offset error
on the same 2.36-million-weight single-p-bit operator. At 4,096 total probe reads,
MSE falls from 0.05064 with an offset-only fit to 0.01903 with a gain/offset fit
(62.4% lower). It hurts when gain is already known, and at the low 256-read budget.
All 160 native inference and 180 probe runs passed independent audits.
[Gain/offset report](results/thrml_spw/GAIN_CALIBRATED_OPERATOR_REPORT.md).
These are synthetic static device errors, not physical hardware measurements.

Earlier robustness result: sample-based calibration reduced a synthetic fixed-offset
large-operator MSE from 0.2540 to 0.0160, retaining one p-bit per normalized weight,
finite eight-bit controls and T512 inference. It required 4,096 native probe samples
per free device. All 30 calibration and 50 inference runs passed independent audits.
This is an isolated operator under synthetic additive offsets, not measured hardware
or new full-model quality. [Calibration report](results/thrml_spw/CALIBRATED_OPERATOR_REPORT.md).

Latest verified result: 25.92% of DistilGPT-2 parameters execute as native THRML
probability-encoded weights. At 1,024 samples, independent confirmation perplexity
is 133.395 versus 129.868 FP32 (+2.72%), with token accuracy down 0.347 percentage
points. All predeclared quality and resampling-margin constraints passed on 8,192
new targets across five sampling seeds. Ten calibration/confirmation runs and 120
operator checkpoints passed independent audits.
[Conservative confirmation report](results/thrml_spw/QUARTER_CONSERVATIVE_REPORT.md).

This takes twice the sampling of the earlier failed T512 confirmation; that negative
result remains preserved. Only 25.92% of parameters are converted, using local affine
ranges and finite 8-bit controls. Physical memory, latency and energy advantages
remain unproven; conventional affine8 inference scores better here. The long-horizon
research goal remains active.

The complete research protocol is in [master_design.tex](master_design.tex). The
execution record and caveats are in [results/execution_report.md](results/execution_report.md);
gate decisions are in [results/decision_gates.md](results/decision_gates.md).

The new active research goal uses Extropic's official THRML simulator to implement
the direct representation w=p for weights in [0,1]. See the
[native THRML track](research/thrml_spw/README.md) and its
[long-horizon roadmap](research/thrml_spw/ROADMAP.md). Native simulator evidence is
under `results/thrml_spw/`; the earlier generic pilots remain separate background.

New native results include a larger, five-seed Fashion-MNIST study with explicit
direct versus affine encodings ([report](results/thrml_spw/FASHION_REPORT.md)),
and one actual THRML projection inside pretrained DistilGPT-2
([selective conversion report](results/thrml_spw/TRANSFORMER_PROJECTION_REPORT.md)).
The subsequent [complete first-block projection experiment](results/thrml_spw/TRANSFORMER_BLOCK_REPORT.md)
converts 7.08 million weights (8.64% of parameters), with regenerated intermediate
activations. At 1,024 samples, test-subset perplexity is 114.99 versus FP32 112.21.
A stronger conventional 4-bit control scores 112.81. Full-model and physical-system
advantages remain unproven. The current goal is active.

## Setup

The scalar experiments require NumPy. Vision and Transformer experiments use CPU
PyTorch/torchvision, and the Transformer pilot additionally uses Hugging Face
Transformers and datasets. On CPU-only ARM64, use:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-vision.txt
.venv/bin/python -m pip install -r requirements-transformer.txt
```

## Core reproduction commands

```bash
.venv/bin/python -m experiments.e00_reproduce_baseline
.venv/bin/python -m experiments.e01_scalar
.venv/bin/python -m experiments.e02_local_range
.venv/bin/python -m experiments.e03_nonidealities
.venv/bin/python -m experiments.e04_dot_product
.venv/bin/python -m experiments.e05_matmul
.venv/bin/python -m experiments.e06_vision --datasets mnist fashion fashion_mlp --seeds 0 1 2 3 4
```

Vision runs download MNIST/Fashion-MNIST automatically and save checkpoints under
`data/checkpoints/`. CIFAR-10 and WikiText-2 require network access. All measurements
are software simulations; no physical p-bit hardware was available.

## Earlier generic pilot highlights

- Scalar empirical MSE agrees with theory within about 0.4% over the specified 12
  sample counts, 1,000 targets, and 1,000 trials.
- Linear sample-and-accumulate is algebraically identical to reconstruct-then-multiply
  for matched sampled weights; the measured maximum difference was below `1e-15`.
- On five MNIST checkpoints evaluated on a shared 2,000-example subset, per-output-channel
  SPW weight reconstruction followed by deterministic inference reached 96.8% at 64
  samples and 97.0% at 256, near 97.2% FP32. Averaging complete stochastic-network
  outputs reached only 71.1% and 78.8%.
- Fashion-MNIST CNN showed the same split: reconstructed inference reached 88.4%/89.4%
  at 64/256 samples versus 89.8% FP32; direct stochastic-network averaging remained
  near 40%.
- The five-epoch CIFAR-10 pilot reached 71.9% FP32 and 68.1% R1 at 256 samples on
  separate 2,000-example evaluation subsets; this model did not meet the small-model
  1–2 pp target at the tested budget.
- The R1 quality comes with repeated reads per weight. It does not demonstrate lower
  total transitions, energy, or latency than deterministic low-bit inference.
- Expected-weight, straight-through, Concrete, local-reparameterization, and REINFORCE
  MNIST training pilots ran. In these short single-seed runs, no sampled estimator clearly
  beat expected-weight training; REINFORCE showed a high gradient norm.

## Layout

`weight_encodings/` contains the FP, INT4, multi-p-bit, SPW, and adaptive residual
representations. `stochastic/` contains ideal and nonideal samplers. `experiments/`
contains runnable phases. `analysis/` produces resource comparisons and figures.
Raw CSVs, caveats, and the phase-by-phase report are under `results/`.

The subsequent [native range-localization study](results/thrml_spw/GROUPED_BLOCK_REPORT.md)
uses 128-weight groups to reduce the selected schedule from 1,024 to 256 samples
per projection. Test-subset perplexity is 116.35 (+3.70% versus FP32), with 75.34%
fewer free-node evaluations and eight times as much range metadata. This remains
a software quality/cost study; controller precision and physical advantages are open.

The [finite-controller experiment](results/thrml_spw/TRANSFORMER_CONTROL_REPORT.md)
selects 8-bit programmed biases at 256 samples: test-subset perplexity 116.63 versus
112.21 FP32. Coarser uniform controls fail the declared calibration gate. The study
separates persistent programming error from sampling noise and counts controller
storage explicitly. A disjoint, unscored confirmation set is now reserved.

Packed controller artifacts now cover all 21.23 million converted weights using
27.64 MB total (10.415 bits/weight including metadata), with exact native roundtrip
checks and deterministic mean replay. This is control serialization, not one total
storage bit or runtime/hardware compression. [Details](results/thrml_spw/PACKED_CONTROLS_REPORT.md).
The conservative driver remains live in confirmation; final results are pending.
