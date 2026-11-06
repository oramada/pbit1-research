# Active research track: direct w=p encoding in Extropic THRML

## LFM3 research: one p-bit per parameter

The target is **one logical programmable p-bit for each of LiquidAI/LFM3-1.7B's 1,728,801,536 unique learned parameters**, evaluated against BF16 task quality. More temporal samples do not add more p-bits per parameter. Four- and eight-p-bit encodings are comparison controls. Signed weights use shared local ranges; finite programming controls, range metadata, and accumulators are counted separately.

The initial frozen five-seed THRML study covered every parameter on 512 fixed language targets. Mean NLL was **4.9282 versus BF16's 4.6357**, or subset perplexity **138.13 versus 103.10**. That configuration misses BF16 quality. Its [independent statistical audit](../../results/thrml_spw/lfm3_1p7b/language_native_v2/statistics_audit.json) remains part of the record; this is a conditional subset evaluation, not a full WikiText benchmark.

Both 16-update adaptation arms and the unadapted control passed independent trajectory/export audits. The prior-selected mean-adapted one-p-bit cohorts are now complete: five seeds per sampling budget on all 2,048 validation targets. At **T1024**, mean perplexity is **83.9167 versus same-fit BF16's 75.3147** (11.42% worse). At **T4096**, it is **76.1024** (1.05% worse), with a conditional Monte Carlo 95% perplexity-ratio interval of **0.9988 to 1.0223**. The upper bound is 2.23% worse, narrowly exceeding the unchanged 2% criterion. Both gates therefore fail; neither a confidence interval crossing one nor a favorable mean proves equivalence. [Completed-cohort root review](../../results/thrml_spw/lfm3_1p7b/completed_onebit_cohorts_root_review.json). **BF16 task-quality equivalence remains unproven**, and no numerical BF16 equivalence is claimed.

| Track | Protocol or accepted evidence | Current state source |
| --- | --- | --- |
| Model and coverage | [Pinned inventory](../../results/thrml_spw/lfm3_1p7b/meta_inventory.json) | [Reviewed preparation](../../results/thrml_spw/lfm3_1p7b/preparation_status.json) |
| Primary mean-adapted one-p-bit study | [T1024 protocol](../../results/thrml_spw/lfm3_1p7b/counted_mean_validation_protocol.json) | [Completed cohort](../../results/thrml_spw/lfm3_1p7b/counted_mean_validation/pipeline.json) |
| Conditional higher sampling, still one p-bit | [T4096 protocol](../../results/thrml_spw/lfm3_1p7b/counted_mean_t4096_protocol.json) | [Completed extension](../../results/thrml_spw/lfm3_1p7b/counted_mean_t4096/pipeline.json) |
| New higher sampling, one p-bit per parameter | [T8192 protocol](../../results/thrml_spw/lfm3_1p7b/counted_mean_t8192_protocol.json) | [Five fresh seeds running](../../results/thrml_spw/lfm3_1p7b/counted_mean_t8192/pipeline.json) |
| Original three-arm native validation | [Protocol](../../results/thrml_spw/lfm3_1p7b/adaptation_native_v2_protocol.json) | [Pipeline](../../results/thrml_spw/lfm3_1p7b/adaptation_native_v2/pipeline.json) |
| Sampling allocation at the same learned point | [Protocol](../../results/thrml_spw/lfm3_1p7b/allocated_native_protocol.json) | [Pipeline](../../results/thrml_spw/lfm3_1p7b/allocated_native/pipeline.json) |
| Original frozen-checkpoint eight-p-bit comparison | [512-target protocol](../../results/thrml_spw/lfm3_1p7b/word8_language_protocol.json) | [Pipeline](../../results/thrml_spw/lfm3_1p7b/word8_language/pipeline.json) |
| Matched adapted-model K4/K8 controls | [Same-point, same-target, matched-work protocol](../../results/thrml_spw/lfm3_1p7b/adapted_word_validation_v2_protocol.json) | [Recovery pipeline](../../results/thrml_spw/lfm3_1p7b/adapted_word_validation_v2/pipeline.json) |
| Count-based software observer | [Strict whole-model replay](../../results/thrml_spw/lfm3_1p7b/counted_native_replay/initial_replay/independent_audit.json) | [All-stage bit-pattern review](../../results/thrml_spw/lfm3_1p7b/counted_native_replay/root_replay_crosscheck.json) |
| Spatial-control worker integration | [GPU trajectory/operator audit](../../results/thrml_spw/lfm3_1p7b/adapted_word_worker_gpu_preflight/independent_audit.json) | Full adapted-model quality pending |
| Untouched article-separated holdout | [Data audit](../../results/thrml_spw/lfm3_1p7b/holdout_data/independent_audit.json) | [Pre-score eligibility rule](../../results/thrml_spw/lfm3_1p7b/holdout_treatment_selection_v2_policy.json); all 10,240 targets unscored |

A new T8192 condition is now running five fresh seeds at the same point and the same 2% criterion. It uses twice T4096's retained sampling work and eight times T1024's; all previous results remain intact. The holdout eligibility rule was explicitly versioned to include it before holdout scoring, which still requires a complete execution protocol. The T4096 extension ran after the complete T1024 cohort missed the fixed quality criterion. It used four times the retained scheduled node evaluations; warmup, programming controls, and ranges are separately counted. The adapted K4/K8 comparison uses the same learned point and targets, with actual padding-inclusive work checked against the corresponding one-p-bit run; it is running under the original memory guards after recovery of its upstream statistics bookkeeping. The original 512-target frozen-checkpoint K8 study cannot serve as that adapted-model comparator. All original K8 native cases passed. Its final audit stopped because a live pipeline-file hash was incorrectly treated as immutable. The [corrected immutable-snapshot statistics audit](../../results/thrml_spw/lfm3_1p7b/word8_language_statistics_v2/statistics_audit.json) passes with identical numerical results and no neural sampling rerun; the original failures remain preserved.

The count-based observer passed a strict same-key full-model replay before its separate study began. It uses temporary per-weight counters; this does not establish a physical output-only datapath, reduced sample requirements, or hardware efficiency. Original studies retain their frozen workers and settings. [Detailed research log](RESEARCH_LOG.md) and [LFM3 progress](../../results/thrml_spw/LFM3_ADAPTATION_PROGRESS.md) record accepted results, negative findings, and pending requirements. Executed sources and accepted evidence are immutable; each extension has its own namespace.

One logical p-bit per parameter describes device assignment. The simulator reuses bounded banks and retains programming controls, shared ranges, counters, and reference weights. [Resource accounting](../../results/thrml_spw/lfm3_1p7b/language_resource_model/independent_audit.json) distinguishes those costs. Physical energy, latency, full device residency, and a hardware systems advantage remain unmeasured.

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


Current run state and reviewed findings: [research status](../../results/thrml_spw/STATUS.md). The milestone entries below are historical snapshots; use the linked live pipeline for current counts.

Full probability-training reproduction passed in a fresh process: all ten fits, complete epoch-history bytes, selection decisions and 240 exported arrays match exactly, including 2 million validation logits. Same-code reproducibility only; no new independent seeds or native quality result. [Reproduction report](../../results/thrml_spw/FASHION_CNN_PROBABILITY_TRAINING_REPRODUCTION_REPORT.md). Full 300-case THRML evaluation remains live; goal active/global index stale.

Matched CNN comparison fully audited: at equal stochastic-node evaluations, one-p-bit accuracy is 66.0010/78.5182/83.0638% versus four-p-bit 83.0896/83.1714/83.2036%. Largest-budget accuracy and CE difference intervals include zero; no equivalence or systems-win claim. [Reviewed comparison](../../results/thrml_spw/FASHION_CNN_MATCHED_WORD_REVIEW.md). All 30 learned-probability native validation cases also pass; the full 300-case fine-tuned native task evaluation is now running. [Learned-command validation](../../results/thrml_spw/FASHION_CNN_PROBABILITY_NATIVE_REPORT.md), [live task cohort](../../results/thrml_spw/fashion_cnn_native/probability_cohort_evidence/pipeline.json). Goal active/global index stale during live writes.

All 150 four-node positional CNN production cases completed successfully: 223,776,000,000 scheduled node updates. Every saved case hash, shape, finite logit array and clipping-metadata record has been reviewed without computing task metrics. Complete independent public-state replay is now running; accuracy comparisons remain pending replay/statistics. [Production review](../../results/thrml_spw/fashion_cnn_native/word_comparison_evidence/production_root_review.json), [live comparison](../../results/thrml_spw/fashion_cnn_native/word_comparison_evidence/pipeline.json). Both learned-probability follow-ups remain actually queued behind the required audits. Goal active/global index stale during live writers.

Full native probability-finetuned CNN evaluation is source-frozen and actually queued: 300 cases across five models/two training methods/three budgets/ten inference seeds, complete public-state recovery and all 33 descriptive contrasts. Native validation must pass first. Deterministic control and cross-environment bootstrap prerequisites pass; no new task results yet. [Full evaluation protocol](../../results/thrml_spw/FASHION_CNN_PROBABILITY_COHORT_PROTOCOL_REPORT.md). Capacity forecast includes all queued outputs; goal active/global index stale during live writes.

Learned-probability CNN native validation is now source-frozen and actually queued behind the matched four-node full audits: all ten checkpoints, T64/256/1024, two eight-image validation banks, complete independent public-state recovery. All 26,640 commands and fixed ranges/biases statically checked; no native validation or new quality scores yet. [Native validation protocol](../../results/thrml_spw/FASHION_CNN_PROBABILITY_NATIVE_PROTOCOL_REPORT.md). Goal active/global index stale during writes.

CNN probability-aware fine-tuning preparation complete: five models × two methods, fixed ranges/biases and learned finite-field commands. Selected validation outputs/exports independently replayed (2 million logits exact). This is Torch surrogate training; native confirmation and test quality remain open. [Preparation report](../../results/thrml_spw/FASHION_CNN_PROBABILITY_FINETUNE_PREPARATION_REPORT.md). Matched four-node CNN production continues unchanged. Goal active; global index stale during live writes.

Latest completed CNN result: all 150 native cases now pass full public-state replay and independent control/statistical audits. At 1024 samples/layer accuracy is **83.0638% versus 85.0860% FP32**, a **2.0222 percentage-point loss**; the approximate 1–2 point quality target is not declared passed. FP16/BF16 controls are also fully audited and remain near baseline. All 7,077,375,000 native counter entries match public recovery. The matched four-node CNN pipeline has started GPU validation; its full quality comparison remains pending. [Reviewed CNN result](../../results/thrml_spw/FASHION_CNN_NATIVE_REVIEW.md), [precision controls](../../results/thrml_spw/FASHION_CNN_PRECISION_REPORT.md), [live matched comparison](../../results/thrml_spw/fashion_cnn_native/word_comparison_evidence/pipeline.json). Goal active; global artifact index stale while writers remain live. Earlier entries below are historical checkpoints.

All150 direct integer-activation classifier cases now pass full causal/count/statistical audits: one free p-bit per each of2368matrix weights, no activation p-bits or per-weight operational reconstruction counters. AtT1568 accuracy is89.6378% versus89.8800%FP32 and89.6428%same-trajectory reconstruction. All70875000counters match exactly. [Reviewed direct result](../../results/thrml_spw/INTEGER_ACTIVATION_CLASSIFIER_REVIEW.md). The earlier300-case Boolean comparison and static-cache control remain independently audited. The full CNN has passed GPUvalidation and50deterministic controls and is actually producing cases(24/150at this checkpoint); FP16/BF16evaluation is complete and waits for full parent audits before comparison. [CNN GPU validation](../../results/thrml_spw/FASHION_CNN_GPU_VALIDATION_REPORT.md), [live CNN pipeline](../../results/thrml_spw/fashion_cnn_native/cohort_evidence/pipeline.json).

A bounded native convolution operator now passes all12 cases, including signed ranges, stride, dilation and exact masked padding:5904 counters agree with independent public states and separate spatial-loop outputs agree within4.9e-15. This is an operator prerequisite, not full CNN accuracy. [Convolution report](../../results/thrml_spw/CONVOLUTION_NATIVE_REPORT.md).

Five conventional Fashion-MNIST CNN checkpoints are now frozen and independently checked. A native three-layer causal validation witness passes all15 cases/566700 exact public-state counters, with accumulation before ReLU/maxpool. No test accuracy has been scored for this cohort; complete full-test native production/control/statistical protocols and a capacity-bounded replay format remain to implement. [CNN preparation report](../../results/thrml_spw/FASHION_CNN_PREPARATION_REPORT.md).

CNN capacity preparation now passes exact public recovery of all420bounded validation arrays and actual vectorized-native replay. Compact counter/output receipts plus final logits require204MB before supporting artifacts for the planned150-case cohort; hashes require complete sampler regeneration and cannot decode counters themselves. Full-budget GPU validation is source-sealed and actually queued behind the integer classifier; full CNN test scoring is still unstarted. [CNN recovery report](../../results/thrml_spw/FASHION_CNN_COMPACT_REPORT.md).

The full150-case native CNN cohort is now source-sealed before test scoring and its real runner waits behind the GPU prerequisite. All5models/10MCseeds/T64,256,1024, ten deterministic control conditions, complete public replay and independent statistics are fixed. Planned production223776000000updates and matching audit are not yet executed; full CNN quality is pending. [Full cohort protocol/report](../../results/thrml_spw/FASHION_CNN_COHORT_PROTOCOL_REPORT.md).

The CNN FP16/BF16 supplement is separately source-sealed and actually queued:20validation paths pass, distinguishing rounded weights/F64arithmetic from pinned TorchCPU low-precision execution. It adds no native draws and waits on parent controls/full audits before evaluation/comparison. [Precision protocol report](../../results/thrml_spw/FASHION_CNN_PRECISION_PROTOCOL_REPORT.md).

Full CNN conventional-control review passes all50paths/five millionlogit values with no prediction mismatches under the fixed numeric conditions. The four-node positional CNN reference also passes25native/public validation cases,944500integer counters and700byte-exact bank arrays. It is an internal reference; GPU/full matched-resource test comparison remains pending. [Four-node witness](../../results/thrml_spw/FASHION_CNN_WORD_REFERENCE_WITNESS_REPORT.md).

The full150-case four-node positional CNN comparison is now source-sealed and actually queued behind the one-node full audit. WordT16/64/256 match one-nodeT64/256/1024 in stochastic-node evaluations. GPU proof, complete native/public cases and paired-model/independent-MC statistics are fixed before word test scoring; no word full-test quality yet. [Matched comparison protocol](../../results/thrml_spw/FASHION_CNN_MATCHED_WORD_PROTOCOL_REPORT.md).

All150one-p-bit native CNN production cases completed with223776000000scheduled updates and exit0. Every case hash/schema is reviewed; complete public-state causal replay is actually running(21/150at this checkpoint). CNN task quality remains pending full replay/statistical audits. [Production review](../../results/thrml_spw/fashion_cnn_native/cohort_evidence/production_root_review.json), [live pipeline](../../results/thrml_spw/fashion_cnn_native/cohort_evidence/pipeline.json).

Fully audited matched-update neural comparison: one p-bit per normalized converted
weight at T256 gives PPL144.850 versus139.959 for four positional p-bits/T64 and
134.587 FP32. Accuracies are22.402%,22.305%,22.949%; the paired accuracy interval
includes zero. One node uses one quarter the dynamic weight nodes, four times the
sample rounds and a larger control/range payload. All10 model paths,960 native
bank outputs and60,000 resampling values are independently checked. This converts
51.84% of one checkpoint on consumed short validation contexts; no physical
systems advantage or Extropic architecture disproof follows. [Report](../../results/thrml_spw/POSITIONAL_STUDENT_REPORT.md).

The long-horizon goal remains active. The following matched-update Transformer cohort is closed; the full Boolean classifier study linked above is live. Broader coverage and measured device/controller costs remain open. Earlier entries below are retained as historical context.

New native Boolean learned-layer path independently verified:58.4million integer
AND/weight/popcounts match public raw-state replay exactly. An explicit activation
range restores standardized negative inputs; the first clipped-input attempt stays
failed. Across five learned first layers and fixed budgets, activation sampling
adds about4.60times the matched-control sampling MSE and25% extra activation
nodes for eight simultaneous inputs. This is learned-layer arithmetic, not new
classifier accuracy or physical efficiency. [Report](../../results/thrml_spw/BOOLEAN_AFFINE_LAYER_REPORT.md).

Finite-control information analysis now independently audited:112 conditions,
including a matched16-level scalar prior, quantify how sampling reveals a
programmed probability. This is analytical evidence, not new neural or hardware
quality. The source/audit failure and corrected derivative are retained.
[Information report](../../results/thrml_spw/INFORMATION_BUDGET_REPORT.md).

Finite-field spatial comparison now independently audited: 33,924,000 native
count elements match public THRML replay exactly; all44 scalar theory/error rows
pass. At equal updates, four positional p-bits give lower scalar error than one
p-bit throughout tested budgets. One p-bit overtakes two-bit quantization at
longer sampling budgets. This is an internal positional reference, not an asserted
Extropic architecture, neural comparison or hardware advantage. [Report](../../results/thrml_spw/FINITE_POSITIONAL_REPORT.md).

Conventional controls independently checked on the same consumed text: zero-preserving
INT4/group16 gives perplexity **132.298**, versus **134.816** for SPW/group16/T512
and **129.541** FP32. Nominal code/range/bias payload is 6.28125 versus 12.03125
bits per converted weight. All 21 deterministic variants, their token metrics,
891,813,888 reconstructed weight coordinates and 440,000 paired resampling values
were checked. The encoded-mean SPW reference gives PPL 129.492; finite sampling
remains consequential. This diagnostic uses no fresh confirmation text or native
samples and does not change the failed SPW strict gate. The encoding works in
THRML; a hardware systems advantage remains unproved. [Control report](../../results/thrml_spw/FINE_RANGE_QUANTIZATION_CONTROLS_REPORT.md).

Audited five-seed end-to-end comparison: finer ranges at T512 give perplexity
134.816 versus 136.834 coarse T1024 and 129.541 FP32. Fine next-token accuracy
loses 0.430 percentage points versus FP32. The fine point quality rule passes,
but its strict rule fails (perplexity upper sensitivity +5.96% versus the 5% limit).
The paired CE interval includes zero; a paired improvement is not confirmed.
Fine halves observed node updates with 41% more converted-weight control/range
payload. All 240 native operators, ten complete model paths and 60,000 resampling
values passed independent checks. This converts 51.84% of one model on short
teacher-forced contexts. [Full report](../../results/thrml_spw/FINE_RANGE_CONFIRMATION_REPORT.md). The broad goal remains active.
[Restore instructions](../../results/thrml_spw/lossless_archive/README.md) apply to archived historical arrays.

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
[Local-range operator report](../../results/thrml_spw/RANGE_LOCALIZATION_REPORT.md). The broad research goal remains active.

New independently audited validation-selected allocation matches uniform sampling's
weighted update budget exactly but gives no confirmed quality benefit. Held-out
perplexity is 169.960 allocated versus 169.775 uniform T1024 (FP32 159.794);
the paired CE sensitivity interval includes zero and both 5% quality gates fail.
All 120 validation/240 test native operators, ten full test cases and 60,000
bootstrap values passed independent native/model/score checks. Six layers change
budget; serialized layer sampling rounds fall 8.33% while maximum layer T doubles.
These are counters, not a physical latency/energy improvement. All six blocks use
one free p-bit per normalized weight (51.84% of one checkpoint's unique parameters).
[Validation-selected allocation report](../../results/thrml_spw/SAMPLE_ALLOCATION_REPORT.md). The broad goal remains active.

New independently audited fixed-regime scaling test converts all six Transformer
blocks' 24 affine matrices: 42,467,328 weights, 51.84% of unique parameters.
At ideal finite8/T1024, perplexity is 131.834 versus 127.513 for the shared
first-three-block prefix and 123.754 FP32. All24 point gate: False; strict gate:
False. Five seeds, 32 fresh short-context windows, 120 operators and both
paired neural paths were independently replayed and scored. One fixed frequency
tolerance failure has byte-exact publicF32/roundoff evidence; the original and tighter
compensation failures remain explicit. Embeddings/head
remain deterministic; this is broader conversion, not full-model or physical proof.
[All-six-block scaling report](../../results/thrml_spw/ALL_BLOCKS_REPORT.md). The broad goal remains active.

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
[Weight-only probe placement report](../../results/thrml_spw/PROBE_PLACEMENT_REPORT.md). The broad goal remains active.

New audited diagnostic identifies calibration estimates as the main limitation in
the tested Transformer regime. Keeping eight-bit control range, T1024, devices and
sampling keys fixed, known-parameter controls score perplexity 131.676 versus
135.426 measured calibration and 131.597 ideal devices. All 60 new native operators,
five complete model forwards and 16 mean-weight diagnostics passed independent audits.
[Calibration-estimation diagnostic](../../results/thrml_spw/TRANSFORMER_CALIBRATION_FLOOR_REPORT.md).
This uses the same 25.92% converted subset and consumed text. True device parameters
are granted to an oracle, so it is not a deployable fix or new confirmation. The old
failed 5% measured-calibration regime remains unchanged. The broad goal stays active.

New audited pretrained-model result: measured calibration reduces perplexity from
1,137.355 to 135.426 under synthetic static gain/offset errors; FP32 is 127.140.
The calibrated result remains 6.52% above FP32 and fails the fixed 5% quality limit.
The paired ideal-device case scores 131.597 (+3.51%) and passes both frozen rules.
All 15 sequential quarter-model runs, 180 native operators and 120 probe windows
passed independent sampling, model-input/loss, score and bootstrap audits.
[Calibrated Transformer report](../../results/thrml_spw/CALIBRATED_TRANSFORMER_REPORT.md).
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
[Five-seed training report](../../results/thrml_spw/PROBABILITY_FINETUNE_REPORT.md).
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
[Clocked full-network report](../../results/thrml_spw/CLOCKED_MNIST_REPORT.md).
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
[Finite-endpoint report](../../results/thrml_spw/FINITE_ENDPOINT_MNIST_REPORT.md).
Removing clamps costs 18.92% more active updates at equal sampling depth. This is
synthetic static-device evidence on an existing test set, not hardware performance.
The long-horizon goal remains active for clocked networks, training and broader conversion.

Latest audited network milestone: measured calibration reaches 89.371% versus 89.880%
FP32 on full-test direct-p=w MNIST networks under static synthetic device errors.
[Report](../../results/thrml_spw/CALIBRATED_MNIST_REPORT.md). All 750 runs replayed;
[scope audit](SCOPE_AUDIT.md) keeps endpoints, drift and physical questions open.

Earlier audited operator milestone: clocked drift during probes and inference, with
retained/refreshed controls at fixed query ages and all costs explicit.
[Report](../../results/thrml_spw/DRIFT_RETENTION_OPERATOR_REPORT.md). Goal active;
[scope audit](SCOPE_AUDIT.md) preserves full-network and physical requirements.

Earlier audited operator milestone: calibrated K1/K4 has matched total probe and
inference draws, with high-budget MSE 0.018626/0.018672 and explicit storage/depth
tradeoffs. [Report](../../results/thrml_spw/CALIBRATED_MATCHED_OPERATOR_REPORT.md).
Goal active; [scope audit](SCOPE_AUDIT.md) preserves the remaining requirements.

Earlier audited operator milestone: equal-budget two-probe gain/offset calibration
reduces varying-gain MSE 62.4% at 4096 reads, while preserving low-budget/unit-gain
failures. [Report](../../results/thrml_spw/GAIN_CALIBRATED_OPERATOR_REPORT.md).
[Full scope audit](SCOPE_AUDIT.md): goal active; physical and broader requirements open.

Earlier audited milestone: measured finite-control calibration reduces synthetic large-
operator MSE from 0.2540 to 0.0160 without replica p-bits; calibration cost is counted.
[Latest robustness report](../../results/thrml_spw/CALIBRATED_OPERATOR_REPORT.md).

Current milestone: the T1024 conservative quarter-model confirmation passed (+2.72%
perplexity on 8,192 new targets); all ten native runs and 120 operators were audited.
[Latest report](../../results/thrml_spw/QUARTER_CONSERVATIVE_REPORT.md). Goal active.

This track implements the user's stated representation: every scalar weight w in
[0,1] is the stationary probability of one logical p-bit. It uses Extropic's official
THRML source at commit `2d54cd8404df85bb65703d7e8b4e84e48c1b72a9`, installed in
`.venv-thrml`. Earlier NumPy-only pilots remain separate evidence under `results/`.

## Exact representation

For a THRML spin s in {-1,+1}, a unary energy is E(s)=-beta*b*s. Therefore

    P(s=+1) = sigmoid(2*beta*b).
    b = (log(w) - log(1-w)) / (2*beta).

The Boolean readout z=(s+1)/2 then has E[z]=w. The estimator mean(z) is unbiased
up to numerical control rounding, with variance w*(1-w)/T for independent draws.
The weights are represented in unary bias fields, not in pairwise Ising couplings.
All weight-bank couplings are zero. Coupling these nodes would generally change
their marginal probabilities and needs a separate analysis.

Finite fields implement 0<w<1. Exact endpoints 0 and 1 use THRML's explicit clamped
node API, rather than silently clipping requested weights. Both count as logical
nodes; endpoint modes and their control metadata are reported.

The pinned `IsingEBM` convenience class attempts to build an empty edge factor for
an uncoupled model, which raises IndexError. `native.py` instead uses official
`BlockGibbsSpec`, `SpinEBMFactor`, `SpinGibbsConditional`, `FactorSamplingProgram`, and
`sample_states`. Upstream code is unchanged. Every simulated p-bit update is carried
out by THRML's sampler; no NumPy binomial substitute is used.

## Reproduce

```bash
git clone https://github.com/extropic-ai/thrml.git vendor/thrml
git -C vendor/thrml checkout 2d54cd8404df85bb65703d7e8b4e84e48c1b72a9
python3 -m venv .venv-thrml
.venv-thrml/bin/python -m pip install -r requirements-thrml.lock.txt
.venv-thrml/bin/python -m unittest research.thrml_spw.test_native -v
.venv-thrml/bin/python -m research.thrml_spw.official_example
.venv-thrml/bin/python -m research.thrml_spw.scalar
.venv-thrml/bin/python -m research.thrml_spw.preactivation
# Train using the existing vision environment, then execute native THRML inference:
.venv/bin/python -m research.thrml_spw.train_positive
.venv-thrml/bin/python -m research.thrml_spw.positive_inference --examples 10000 --out results/thrml_spw/positive_full_test
.venv-thrml/bin/python -m research.thrml_spw.control_sweep
.venv-thrml/bin/python -m research.thrml_spw.control_neural
.venv-thrml/bin/python -m research.thrml_spw.correlations
.venv-thrml/bin/python -m research.thrml_spw.coupling_calibration
.venv-thrml/bin/python -m research.thrml_spw.matched_resources
.venv-thrml/bin/python -m unittest research.thrml_spw.test_accumulator -v
.venv-thrml/bin/python -m research.thrml_spw.layer_budgets
.venv-thrml/bin/python -m research.thrml_spw.layer_budget_matched
.venv-thrml/bin/python -m research.thrml_spw.layer_update_matched
.venv-thrml/bin/python -m research.thrml_spw.accumulator_study
.venv-thrml/bin/python -m research.thrml_spw.accumulator_bias
.venv-thrml/bin/python -m research.thrml_spw.joint_efficiency
.venv-thrml/bin/python -m research.thrml_spw.verify_efficiency
.venv-thrml/bin/python -m research.thrml_spw.efficiency_report
.venv-thrml/bin/python -m unittest research.thrml_spw.test_adaptive_native research.thrml_spw.test_adaptive_shared -v
.venv-thrml/bin/python -m research.thrml_spw.adaptive_study
.venv-thrml/bin/python -m research.thrml_spw.adaptive_bias
.venv-thrml/bin/python -m research.thrml_spw.adaptive_shared_study
.venv-thrml/bin/python -m research.thrml_spw.verify_adaptive
.venv-thrml/bin/python -m research.thrml_spw.verify_adaptive_shared
.venv-thrml/bin/python -m research.thrml_spw.adaptive_report
# Second dataset and larger models; conventional training stays in .venv:
.venv/bin/python -m research.thrml_spw.train_fashion_positive
.venv-thrml/bin/python -m unittest research.thrml_spw.test_range_native -v
.venv-thrml/bin/python -m research.thrml_spw.fashion_native
.venv-thrml/bin/python -m research.thrml_spw.fashion_report
.venv-thrml/bin/python -m research.thrml_spw.verify_fashion
# Pretrained selective projection; keep Torch and native sampling separate:
.venv/bin/python -m research.thrml_spw.transformer_export
.venv-thrml/bin/python -m research.thrml_spw.transformer_projection_native
.venv/bin/python -m research.thrml_spw.transformer_replay
.venv-thrml/bin/python -m research.thrml_spw.verify_transformer
.venv-thrml/bin/python -m research.thrml_spw.transformer_report
```

The native sampler is CPU-configured for this initial reproducible track. Each run
stores source revision, package versions, seeds, configurations, code hashes, raw
evidence, and statistical checks in `results/thrml_spw/`.

## Established first milestone

- The official README's 5-node Ising-chain example reproduced its exact enumerated
  distribution across 10 seeds (mean total-variation distance 0.0131).
- The scalar sweep used 1,001 weights, 1,000 independent trials per weight pooled
  across five seeds, and T=1,2,4,8,16,32,64,128,256,512,1024,4096.
- For w=0.73, mean observed frequency was 0.730224365 at T=4096 across 1,000 trials.
- No target/sample condition failed a conservative exact-binomial check at a 1%
  familywise threshold. Empirical MSE matched its analytic value within 2.28
  estimated standard errors across all sample budgets.
- A native THRML observer accumulated linear outputs before ReLU without storing a
  reconstructed float-weight matrix. Against a separate reconstructed-weight
  observer on the identical Gibbs paths, maximum linear-output error was 5.61e-6
  (float32 accumulation). Its arithmetic remains modeled CPU/JAX computation.
- Five reproducibly initialized/trained MNIST models used 2,368 learned matrix
  weights each, all in [0,1], plus 42 deterministic neural bias values. On the entire
  10,000-image test set, deterministic accuracy averaged 89.88%, and native THRML
  preactivation accumulation averaged 88.75% at T=256 (a 1.13 percentage-point drop).
  UINT4 inference averaged 89.34%; there is no demonstrated systems advantage over
  it. Of the 2,368 logical nodes per model, 1,955–2,024 were interior probabilities;
  the remaining zero/one weights used explicit clamps. See `positive_full_test/`.

## Scope of the proof

This establishes the mathematical encoding and its realization through THRML's
actual sampling API. THRML still stores float32 control fields in software. These
experiments do not establish that the probability controller stores precision for
free, that Python memory is compressed, or that a physical device is more efficient.
Control precision, controller area, calibration, readout, accumulation, sample rate,
and independence must all enter a hardware comparison. See `ROADMAP.md` for the
remaining active research milestones.

## Robustness and comparison evidence

Six-bit finite field controls selected on validation data retained native full-test
accuracy: 88.65% at T=256 versus 88.75% with float32 controls (five checkpoints).
The scalar sweep measures the range/precision error floor separately from sampling
variance. Small coupled-bank tests establish that the independent logit formula
needs calibration when nodes interact; exact moment-matched fields restore marginals
without eliminating covariance. Actual native K-replica and temporal comparisons
follow the same estimator law at equal reads, while their node counts and parallel
depths differ. Results and manifests are under `results/thrml_spw/`.

See `REFERENCE_MAPPING.md` before interpreting the K-bit baselines as comparisons
against Extropic's hardware. The available reference distinguishes sampled model
variables from programmed weights; our direct neural-weight bank is a separate
construction. Hardware density and energy remain unmeasured.

## Reduced sampling and accumulator precision

A 25-pair native validation sweep selected 64 first-layer and 256 second-layer
samples. Full-test confirmation across five models and five sampling seeds per
model reached 88.48%, versus 88.74% for uniform256 on the same keys. Logical reads
fell by 64.86% and free-node updates by 63.77%; endpoint clamps explain the difference.
Both near-equal read and near-equal free-update uniform comparisons are preserved.

A separate validation sweep selected a 16-bit integer sum register. At uniform256,
test accuracy was 88.62% versus 88.63% for float32 sums on its paired keys, without
saturation. Combining the selected nonuniform budgets, six-bit field controls and
16-bit integer sums reached 88.30%. This is a joint confirmation, not an assumption
that individually tested changes compose without loss.

Rounding increments can leave a bias floor even when sums do not saturate. The
two-p-bit negative control preserves a concrete example. More fractional bits at
fixed sum width can also eliminate required headroom as T grows. See
`results/thrml_spw/EFFICIENCY_REPORT.md`, raw logits and manifests for the complete
results. Nominal register/control widths do not imply reduced JAX container storage
or measured physical costs. Larger native models remain open.

## Actual online confidence stopping

Native output-layer stopping now removes completed cases from later THRML requests.
The first layer remains fixed at 64 samples, and the output layer checks margin/SE
at 16/32/64/128/256. Validation-only calibration selected gamma=3. The batch-shared
route preserves weight-state reuse across inputs and was evaluated on the full
10,000-image test set for five trained models by five sampling seeds.

Accuracy was 88.4084% versus 88.4304% for its fixed-output256 reference, with 16.94%
fewer stochastic free-node updates including padding. At a bracketing fixed update
budget, accuracy was 87.6416%; the paired adaptive gain was 0.7668 +/-0.0590 pp.
These are comparisons within the declared batch128/chunk/key protocol. They are
not percentage savings to multiply with results from different execution protocols.

An independent-case route checks bit-for-bit online/replay identity; the shared
route records its actual post-compaction decision traces because regrouping changes
future random paths. Exact/native negative controls show that stopping can bias
returned numerical estimates. Confidence excludes first-layer uncertainty and is
not an anytime coverage guarantee. Variance storage and arithmetic are extra costs.
See `results/thrml_spw/ADAPTIVE_REPORT.md` and `adaptive_evidence.png` for evidence
and limits. Twelve native/control/accumulator/adaptive tests pass. Joint adaptive
execution with the previously selected finite formats is still untested.

## Second dataset and pretrained selective execution

Five larger 196->64->10 Fashion-MNIST models contain 13,184 matrix weights each.
Positive projected FP32 accuracy is 82.806%; the matched signed control reaches
87.704%. Direct p=w at 1024 samples/layer reaches 81.526% and fails its predeclared
one-percentage-point validation gate. This is preserved as a diagnostic result.

An explicitly separate per-output affine mapping reaches 82.300% on those positive
models using (64,1024) samples. The signed-control affine route reaches 86.724%
using (256,1024), versus its FP32 87.704%. Both affine routes passed validation
selection; full-test results average five sampling seeds per trained model. They
require 592 bytes of float32 range metadata per model, plus scaling arithmetic.
The output layer's 1024 samples remain a latency concern. See the
[Fashion report](../../results/thrml_spw/FASHION_REPORT.md),
raw outputs, paired uncertainty and the representation-specific storage ledger.

An actual pretrained DistilGPT-2 projection also runs through THRML: 589,824 weights
in transformer.h.0.attn.c_proj, 0.72% of model parameters. Native linear outputs are
inserted before all downstream nonlinearities, and the remaining model runs normally.
At calibration-selected T=64, held-out short-context subset perplexity is 115.326
versus FP32 112.207 (+2.78%). At T=256 it is 112.600. Zeroing the matrix makes
perplexity about five times worse. This is a useful selective-operator result,
not whole-Transformer p-bit inference or a full-corpus benchmark.

See the [Transformer projection report](../../results/thrml_spw/TRANSFORMER_PROJECTION_REPORT.md)
for pinned model/dataset sources, exact FP32
replay checks, raw token evidence and the 0.72% scope. Expanding conversion requires
regenerating later activations after earlier stochastic operators. Thirteen native
tests pass; finite formats and nonideal devices are not yet validated for these
larger operators. Torch/model package versions are recorded in model_environment.json
and requirements-models.snapshot.txt; the latter is an inventory, not a portable
wheel-resolution guarantee. The export now pins both recorded model and dataset
revisions explicitly. Its original discovery-time source is archived under its
recorded hash, and preserved exports support exact native/replay reproduction.

## Complete first-block projection experiment

All four matrix projections in DistilGPT-2 block 0 now execute through official
THRML: 7,077,888 weights, 8.6408% of model parameters. Later inputs are regenerated
after earlier stochastic substitutions; biases and LayerNorm stay deterministic.
Calibration rejected T=64 and T=256 and selected T=1024 per operator. On the fixed
2,048-token test subset, perplexity is 114.987 versus FP32 112.207 (+2.48%). Five
sampling seeds measure conditional Monte Carlo variation, not corpus uncertainty.

A stronger zero-preserving affine 4-bit control with 128-weight groups scores
112.806. This supports useful native quality, without demonstrating superiority
over conventional quantization. Float32 fields/ranges/biases total 28,339,200 bytes,
equal to the original selected matrices/biases before additional sampler overhead.
The selected schedule executes 28,934,406,144 free-node evaluations per test seed;
these are simulator counts, not measured hardware transitions or energy.

Fifteen native tests pass. Raw evidence audits cover 25 completed native cases,
100 operator checkpoints, 25 independent hook replays and 12 quantized-control
conditions. CPU/GPU checks support using CUDA to accelerate the official sampler;
GPU execution is not physical p-bit validation. Partial CPU checkpoints are retained.
See [the complete block report](../../results/thrml_spw/TRANSFORMER_BLOCK_REPORT.md).
The long-horizon goal remains active. Next: actual native finer-range experiments,
nonuniform operator budgets, finite controllers and larger converted fractions.

## Native range localization across the full first block

The actual input-block128 experiment is complete: all four native projections,
7,077,888 logical weight nodes, regenerated student inputs, five sampling seeds.
Calibration rejects T64 and selects T256 under the same 5% perplexity/one-point
accuracy tolerance. Test-subset perplexity is 116.354 versus FP32 112.207 (+3.70%);
token accuracy is 23.701% versus 24.121%. Previous per-output ranges selected T1024
and scored 114.987. Quality is tolerance-matched, not numerically identical.
Selected free-node evaluations fall from 28,934,406,144 to 7,134,511,104 per test
seed (75.34% reduction); dependent sampling rounds fall from 4,096 to 1,024.
Range metadata rises from 0.0625 to 0.5 bits/weight. All clamps, fields, expanded
range arrays and metadata must be counted. Float32 controls prevent a software
compression claim. The zero-preserving conventional affine4 block128 control
still scores 112.806. This is evidence that localization reduces temporal cost,
not that native sampling beats conventional arithmetic or physical hardware.

These corpus windows were inspected in earlier research, so this is a development
comparison. T selection uses calibration only; a fresh confirmatory text subset
remains necessary before broader generalization claims. See
[range-localization report](../../results/thrml_spw/GROUPED_BLOCK_REPORT.md).
The goal stays active for finite-controller experiments, nonuniform sampling,
larger converted fractions and fresh evaluation data.

Reproduce the grouped block experiment (model cache and prepared exports required):

```bash
.venv/bin/python -m research.thrml_spw.transformer_block_grouped
.venv/bin/python -m research.thrml_spw.block_hook_audit --root results/thrml_spw/transformer_block0_group128
OPENBLAS_NUM_THREADS=4 .venv-thrml/bin/python -m research.thrml_spw.verify_transformer_block --root results/thrml_spw/transformer_block0_group128
.venv-thrml/bin/python -m research.thrml_spw.grouped_block_report
.venv-thrml/bin/python -m research.thrml_spw.snapshot_evidence
```

The current native suite has seventeen passing checks, including two grouped-range
checks also verified on CUDA. This milestone audits 20 native cases, 80 operator
checkpoints and exact independent token replay for all 20 cases.

## Finite controllers in the Transformer block

A predeclared native 4/6/8/10/12-bit bias search at fixed T256 and dimensionless
field range [-4,4] selects 8 bits. Four bits fail calibration (+100.88% perplexity),
six fail (+6.92%), and eight pass (+2.63%). Eight-bit test-subset perplexity is
116.627 versus FP32 112.207 (+3.94%); its conditional Monte Carlo relative interval
is [1.0281,1.0508], so the upper end extends slightly beyond 5%. No test scores
select precision. The result is for 7,077,888 normalized weights in all four first-
block projections, with local128 affine ranges and regenerated student inputs.

Programmed-mean controls separate persistent quantization error from sampling:
eight-bit mean-weight test perplexity is 112.386, while four-bit is 228.491. More
samples cannot erase the latter error floor. This is evidence for this declared
uniform Ising-field lattice, not a claim that every possible four-bit controller fails.
The range is dimensionless, not a measured Extropic voltage specification.

The selected hypothetical packed control/mode/range/bias ledger is 10.4063 bits
per matrix weight before stochastic state and other costs; the conventional
zero-preserving affine4 ledger is 4.3125. Actual software retains float32 fields
and uint32 code arrays, so it is not compressed to the packed ledger. Logical
one-p-bit-per-weight remains distinct from total control information and physical
system costs. No conventional-quality or physical-efficiency win is established.

Eighteen native checks pass; the new finite-grouped check also passes on CUDA.
Audits cover 25 completed native runs, 100 operator checkpoints, exact independent
hook replay of all 25 cases, and ten independent programmed-mean model replays.
See [finite-controller report](../../results/thrml_spw/TRANSFORMER_CONTROL_REPORT.md).

Next: fixed confirmation on 128 reserved WikiText-2 test windows (8,192 targets),
with zero positional overlap with prior native tiles and the documented older
four-window generic pilot. Token provenance and the old pilot score were audited;
confirmation model scores have not been computed. The plan compares FP32,
conventional affine4 block128, floating-field T256 and selected eight-bit T256,
five sampling seeds for native conditions, without retuning on confirmation scores.
This remains the same corpus, not a new-dataset or pretraining-clean guarantee.
The long-horizon goal is active; broader conversion and hardware costs remain open.

Reproduce finite-controller evidence:

```bash
.venv/bin/python -m research.thrml_spw.transformer_control
.venv/bin/python -m research.thrml_spw.block_hook_audit --root results/thrml_spw/transformer_control
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_transformer_control
.venv-thrml/bin/python -m research.thrml_spw.transformer_control_report
.venv/bin/python -m research.thrml_spw.verify_confirmation_reservation
```

The preparation command refuses to overwrite reserved confirmation tokens. Reuse
those immutable tokens and their manifest instead of drawing a more favorable set.

## Frozen confirmation completed on 8,192 new targets

The predeclared 128-window confirmation is complete, with no precision or sampling
selection using its scores. All four first-block matrix projections run natively
(7,077,888 weights, 8.64% of model parameters), using per-output input-block128
ranges and T256. Later student inputs are regenerated after earlier substitutions.

| Condition | Confirmation perplexity | Increase versus same-text FP32 |
|---|---:|---:|
| FP32 | 140.846 | — |
| Conventional affine4 block128 | 143.746 | 2.06% |
| Native float controls, T256 | 145.506 | 3.31% |
| Native eight-bit controls, T256 | 145.554 | 3.34% |

Both native conditions pass the frozen point-mean 5% perplexity/one-point accuracy
loss criterion. Native token accuracy is 22.888% versus FP32 23.096%, a 0.208-point
loss. The selected finite-control paired seed/bank-bootstrap relative interval is
[1.0222,1.0427] under exchangeable bank resampling. Conditional MC interval on fixed
text is [1.0271,1.0398]. These are distinct uncertainty statements; neither certifies
unseen-corpus or physical-device performance. Conventional 4-bit still has better
perplexity; no native quality or systems superiority is established.

No token positions overlap the earlier native tiles or the documented older
four-window generic pilot. The confirmation set is now consumed and cannot be
called fresh for a tuned successor. This is the same WikiText-2 corpus, not a new
dataset. Absolute scores differ from older subsets because the text differs.

Audits verify all ten native cases, 40 operator checkpoints, independent hook
replay of every case, both deterministic model baselines, frozen choices and token
provenance. A separate statistical audit independently recovers 100,000 saved
bootstrap CE/accuracy draw values. Two uncertainty-function invariants pass, in
addition to the earlier eighteen native checks. Each native test seed uses
28,538,044,416 free-node evaluations across sixteen eight-context bank batches;
these are logical counts, not physical transitions, latency or energy.

See [confirmation report](../../results/thrml_spw/TRANSFORMER_CONFIRMATION_REPORT.md).
The goal remains active. Next is broader conversion on development data, including
the natural quarter-model target (three blocks, about 25.9% of parameters), followed
by another independently reserved confirmation set. Grouped partial-sum execution,
large-operator nonidealities, training and physical controller/readout costs remain
open. One logical p-bit per normalized weight is established in software; total
programming precision and physical benefit remain separate questions.

Reproduce the fixed confirmation and its independent checks:

```bash
.venv/bin/python -m research.thrml_spw.transformer_confirmation
.venv/bin/python -m research.thrml_spw.block_hook_audit --root results/thrml_spw/transformer_confirmation --tokens data/thrml_spw/transformer_confirmation/tokens.npz
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_transformer_confirmation
.venv-thrml/bin/python -m research.thrml_spw.confirmation_report
.venv-thrml/bin/python -m research.thrml_spw.verify_confirmation_statistics
.venv-thrml/bin/python -m research.thrml_spw.snapshot_evidence
```

Reproduction reuses the same reserved text and frozen configuration; it does not
constitute a new independent confirmation. The driver refuses protocol changes in
its existing output directory. The bootstrap code has two exact analytic checks
(identical paired losses and a constant paired effect), plus raw-draw verification.

## Three-block development study: 25.92% converted, tolerance missed on test

All twelve projections in the first three DistilGPT-2 blocks now run through
actual THRML: 21,233,664 normalized weights, each with one logical p-bit, eight-bit
bias levels and local input-block128 affine metadata. Student inputs are regenerated
across all block boundaries. Biases, LayerNorm, embeddings/head and the last three
blocks stay deterministic.

Calibration rejects T256 (+11.68% perplexity) and narrowly selects T512 (+4.96%).
The reused development test scores **118.338 versus 112.207 FP32 (+5.46%)**, missing
the same 5% tolerance. Conditional MC interval [1.0450,1.0644] straddles the boundary;
this is a point-mean miss, not proof the true loss exceeds 5%. Accuracy loses 0.508
percentage points. Conventional affine4 scores 118.677 and affine8 scores 112.297;
the programmed-mean8 diagnostic scores 112.655. No quarter-model confirmation or
native superiority is established.

Independent audits verify 15 native cases, 180 operator checkpoints, all final token
arrays via separate hooks, eight independently reproduced baseline models, exact
causal lineage, finite-field MSE decomposition and resource counts. Selected test
cost is 42,807,066,624 free-node evaluations per seed; actual software control arrays
remain 32-bit. Digital code subtotals and logical nodes do not measure physical area
or energy. [Development report](../../results/thrml_spw/TRANSFORMER_QUARTER_REPORT.md).

Another 128-window confirmation set (8,192 targets) is reserved, audited and unscored.
It excludes both consumed native text sets and the old four-window pilot. T512 and
all controls were frozen before quarter development-test outcomes; no confirmation
tuning is allowed. The next step is that fixed confirmation, retaining pass or fail.

A separate actual native partial-sum operator pilot preserves keys/draws and matches
the original 2.36M-weight projection closely. Its ledger reduces range-scaling
multiplications 63x, at 24x accumulator size; binary products/draws are unchanged.
It is not end-to-end quality or hardware-performance proof.
[Operator pilot](../../results/thrml_spw/PARTIAL_OPERATOR_REPORT.md).
A separate cached worker fixes the verified eight-entry shared-LRU eviction for
12 operators; CPU/CUDA tests preserve outputs/counts exactly. Canonical development
artifacts were not changed. Goal active: confirmation, broader/nonideal operators,
training and physical programming/readout/accumulator costs remain open.

Reproduce the three-block development study and independent evidence checks:

```bash
.venv/bin/python -m research.thrml_spw.transformer_quarter
.venv/bin/python -m research.thrml_spw.sequential_hook_audit
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_transformer_quarter
.venv-thrml/bin/python -m research.thrml_spw.quarter_report
.venv/bin/python -m research.thrml_spw.verify_quarter_reservation
.venv-thrml/bin/python -m research.thrml_spw.freeze_quarter_confirmation
```

The prepared next entry point is `.venv/bin/python -m
research.thrml_spw.quarter_confirmation`. It has not yet executed or computed
confirmation scores. It requires audited development artifacts and uses the
already frozen settings, including the same T512 despite the development-test
miss. Its native observer and sampler math are unchanged; a separately validated
worker cache retains compiled functions across twelve operators. The old shared
LRU held only eight, causing eviction/recompilation. This is a software cache fix,
not a reduction in stochastic samples or physical inference work.

For the separate group-partial operator pilot, run
`.venv/bin/python -m research.thrml_spw.partial_operator_pilot`, followed by
`OPENBLAS_NUM_THREADS=4 .venv-thrml/bin/python -m
research.thrml_spw.verify_partial_operator`. Its raw inputs refer to the completed
canonical quarter-model development case; it does not replace the neural results.

## Frozen quarter-model confirmation: T512 misses the 5% criterion

The fixed three-block configuration completed confirmation on 128 new 65-token
windows (8,192 targets), with no retuning after the development-test miss.
All 12 projection matrices use actual THRML sampling, one logical p-bit per
normalized weight, eight-bit bias levels, local input-block128 affine ranges
and 512 samples. Converted scope remains 21,233,664 weights, 25.9223% of the model.

| Condition | Confirmation perplexity | Relative to FP32 |
|---|---:|---:|
| FP32 | 133.203 | 1.0000 |
| Conventional affine4 | 142.002 | 1.0661 |
| Conventional affine8 | 133.193 | 0.9999 |
| Programmed-mean8 | 133.625 | 1.0032 |
| Native eight-bit/T512 | 141.186 | 1.0599 |

The native 5.99% perplexity increase misses the frozen 5% tolerance. Accuracy
loss 0.637 percentage points remains within the one-point bound. Conditional MC
relative interval [1.0529,1.0670] excludes 5%; paired seed/bank-bootstrap sensitivity
interval [1.0483,1.0700] straddles it. These are different uncertainty statements;
no unseen-corpus coverage guarantee is claimed. The programmed-mean floor is only
0.32%, indicating finite sampling dominates this outcome. Native/affine4 paired
perplexity interval [0.9814,1.0061] includes equality; native superiority is unproven.

Independent audits cover all five native runs, 60 operator checkpoints, exact
hook replay of every case, four independent deterministic model replays and 140,000
bootstrap draw values. Each native seed evaluates 171,228,266,496 free nodes across
16 eight-context banks. These are simulator counts, not physical transitions or
chip latency/energy. Float32 fields, uint32 codes and expanded range arrays remain
in software; physical or total-memory gains remain unproven.
[Confirmation report](../../results/thrml_spw/QUARTER_CONFIRMATION_REPORT.md).
The earlier first-block 8.64% confirmation remains valid within its smaller scope.

The confirmation text is consumed and will not be reused as fresh evidence for a
tuned successor. A new prospective study has reserved 128 validation windows and 128
test windows, 8,192 targets per split, excluding all 16 earlier validation windows,
all 288 consumed native test/confirmation windows and four legacy windows.
Independent source/token/offset/selection audit passed; no new-study scores exist.
It will test T1024/2048 with the same quarter-model/control/range configuration and
select on a conservative crossed-bootstrap upper-loss/lower-accuracy criterion,
using validation only. Higher T doubles/quadruples prior stochastic cost. This is
new development motivated by the failure, not a revision of the failed experiment.
The long-horizon goal stays active for that study, broader/nonideal execution,
training and physical-system evidence.

Reproduce the completed frozen quarter confirmation:

```bash
.venv/bin/python -m research.thrml_spw.quarter_confirmation
.venv/bin/python -m research.thrml_spw.sequential_hook_audit --root results/thrml_spw/quarter_confirmation --tokens data/thrml_spw/quarter_confirmation/tokens.npz
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.verify_quarter_confirmation
.venv-thrml/bin/python -m research.thrml_spw.quarter_confirmation_report
.venv-thrml/bin/python -m research.thrml_spw.verify_quarter_confirmation_statistics
```

The conservative successor is specified in
`results/thrml_spw/quarter_conservative_preparation/protocol.json`; its immutable
new tokens are under `data/thrml_spw/quarter_conservative/`. The preparation refuses
to overwrite them. Independently verify the reservation with
`.venv/bin/python -m research.thrml_spw.verify_conservative_reservation`.
Its model experiments have not run yet. Implementation must use its frozen
validation-only rule and new test positions, retain the failed T512 result, and
count the doubled/quadrupled sampling cost.

## Actual packed controller artifacts; conservative study still running

Packed controls now cover all 12 quarter-model matrices (21,233,664 weights),
with eight-bit finite-bias codes, two-bit endpoint modes, float32 shared ranges
and retained neural biases. Artifacts store no original converted target matrices.
Actual size is 27,643,445 bytes, or 10.41495bits per converted weight including headers
and archive overhead. Numeric payload is 10.40625 bits/weight. Nominal FP32 matrix/bias
arrays are 85,017,600 bytes (3.0755 times the artifact size). This is quantization and
packing of control information, not one total storage bit per learned weight.
Conventional affine8 and affine4 ledgers are smaller at 8.34375 and 4.3125 bits/weight.

Every code, mode, range and reconstructed field was checked against the original
mapping across all 21.23M weights. The real model's previously scored 16-window
programmed-mean replay matches every token loss exactly. Two tiny CPU/CUDA native
checks produce identical sampled outputs under matching keys, including ragged,
constant and singleton groups. This is not full stochastic packed-model inference;
the decoder still expands float32 fields, probabilities, node/program objects and
other buffers. Runtime compression, physical area and energy wins remain unproven.
[Packed-control report](../../results/thrml_spw/PACKED_CONTROLS_REPORT.md).

The conservative native study continues unchanged under exec session 15504. Three
T1024 calibration seeds are complete, each about 2.25%-3.08% above same-text FP32;
two more seeds and the predeclared uncertainty rule are needed before selection.
Confirmation remains unscored. Poll the actual existing session/process, not just
the checkpoint file, before resuming. Do not launch a duplicate or relax the rule.
[Live checkpoint](../../results/thrml_spw/quarter_conservative/LIVE_CHECKPOINT.json).
The goal remains active. The full artifact index must be refreshed after the active
writer completes; current new source versions are content-addressed separately.

## Conservative calibration selected T1024; confirmation running

All five new calibration seeds completed. At T1024, mean perplexity is 135.1167
versus 131.8593 FP32 (+2.4704%). The predeclared crossed-bootstrap upper relative
bound is 1.03065, and lower accuracy difference is -0.3906 percentage points; both
conservative constraints pass. Selection is frozen at 1024; the predeclared 2048
condition is not run because the first candidate passed. An independent arithmetic
checker recovers the decision and every 20,000 saved selection bootstrap values.
Full raw-stage/model verification still awaits the completed confirmation.

The same driver is now running its five selected-budget seeds on separate test
text, after freezing selection. Confirmation baselines are FP32: 129.8675,
affine4:136.6173, affine8:129.8228 and programmed-mean8:130.2906. No final native
confirmation outcome is available yet. Poll exec session 15504 or the actual
processes recorded in LIVE_CHECKPOINT.json before continuing; do not duplicate it.

Separately, a real 2,359,296-weight projection executed directly from its packed
controller artifact through actual THRML/CUDA. It loaded no original target FP32
weight matrix. At T512 with identical inputs/keys, every output value exactly
matches the canonical execution, with 2,378,170,368 free-node evaluations per path.
An independent audit decodes mode bits and checks variance, inputs, hashes and
counters without target tensors. This is a large single-operator result, not full
stochastic packed-model quality or runtime/physical compression.
[Packed evidence](../../results/thrml_spw/PACKED_CONTROLS_REPORT.md).
The goal remains active for confirmation, full audits, broader/nonideal execution
and physical-system questions. The full artifact index must wait for the live
writer to finish; the closed packed experiment has its own checksum index.

## Verified conservative confirmation: T1024 passes on new text

The prospective conservative successor is complete and independently audited.
The predeclared validation-only search selected T1024: mean relative perplexity
1.024704, crossed seed/bank upper ratio 1.030649. The T2048 candidate was not run,
as prescribed after the first passing budget. Selection was frozen before test.
On 128 separately reserved confirmation windows (8,192 targets), five native seeds
score perplexity 133.395 versus 129.868 FP32 (+2.7164%), accuracy difference -0.3467
percentage points. The paired seed/bank relative-perplexity sensitivity interval
is [1.0192,1.0353], with accuracy-difference lower bound -0.6397 points. Both point
and conservative resampling constraints pass. These engineering resampling limits
are not formal guarantees for a whole corpus or pretraining-independent dataset.

Scope: all twelve projections in blocks 0/1/2, 21,233,664 weight nodes (25.9223%
of parameters), input-block128 affine ranges, finite eight-bit fields in [-4,4].
The remaining model stays deterministic. Five calibration plus five confirmation
runs passed independent hook replay; all 120 operators, eight deterministic
baselines, raw counts/errors, source/data lineage and 40,000 selection/confirmation
resampling values passed a separate audit. Conventional affine4/8 test perplexities
are 136.617/129.823; these numerical controls do not establish physical efficiency.

Sampling costs double relative to the failed T512 confirmation at the same text
count: 342,456,532,992 free-node evaluations per seed for all 128 windows. The
complete calibration/confirmation study used 3,424,565,329,920 evaluations. The
old failed confirmation remains unchanged. No full-model, one-bit total storage,
physical energy or latency improvement follows. Actual packed artifacts retain
controller/mode/range/bias costs; runtime arrays are larger still.

[Full report](../../results/thrml_spw/QUARTER_CONSERVATIVE_REPORT.md),
[raw audit](../../results/thrml_spw/quarter_conservative/evidence_audit.json),
[figure](../../results/thrml_spw/quarter_conservative_evidence.png).

The goal stays active. Next priorities are fixed per-device versus shared
calibration error in large matched-resource operators, with persistent bias floors
separated from reducible sampling variance; broader converted fractions; and
probability-aware training. New device-error sweeps must be labeled dimensionless
simulator scenarios unless supported by measured device data. Any model tuning
requires validation-only choices and a fresh confirmation reservation. Existing
scored positions are consumed. Preserve the original w=p positive-weight results
separately from affine encoding of signed pretrained weights. 

## Native fixed-device error and measured finite-control calibration

Previous goal turn classification: **progress**. It completed the prospective
quarter-model confirmation and independent audits. This continuation also makes
verified progress; it is not a wait or status-only turn.

A fixed prospective stress study executed 160 native large-operator conditions:
2,359,296 unique weights, K1/K4 replicas, total observations 128/512, synthetic
additive unary-field offset SD 0/0.05/0.2, ten joint device/sampling realizations.
Offsets are fixed across sampling times and budgets. K4 shares offsets across
replicas or has independent device offsets; distinct weights stay independent.
Endpoint clamps remain exact. At each observation count, free updates and binary
products match. Independent raw audits recomputed every field hash, conditional
mean/error, count and saved output. Native dynamic fields were checked against
separately rebuilt public factors on CPU/CUDA.

The exact conditional bias floor does not shrink with temporal samples. Independent
replicas reduce device-specific errors in these scenarios, while shared offsets
leave exactly the same conditional mean/floor as K1 for every device seed. This is
an important condition on a viable single-device representation, not a measured
Extropic imperfection or an argument against the representation itself.
[Stress report](../../results/thrml_spw/CALIBRATION_OPERATOR_REPORT.md).

The zero-offset primary ten-seed sample had a small K4-minus-K1 difference whose
unadjusted normal interval excluded zero. We retained it and froze exactly 40 fresh
seeds 10..49, without significance-driven stopping. All 80 additional native outputs
passed raw audits. Fresh K1/K4 sampling-MSE/theory ratios are 1.00123/1.00480; fresh
MSE contrast is +0.00005088 +/-0.00016934, including zero. The combined 50 result is
reported separately as diagnostic, not independent confirmation or equivalence
proof. [Fixed follow-up](../../results/thrml_spw/ZERO_OFFSET_RECHECK_REPORT.md).

We then executed a count-based calibration procedure on the same large K1 operator
with fixed synthetic offset SD 0.2. A real eight-bit probe code 128 has field +0.0156863
(zero is not representable). Official THRML counts at Ncal 256/1024/4096 estimate each
offset using a smoothed logit. The measured estimator receives no true offsets or
task labels. Correction is rounded/clipped to actual eight-bit codes in [-4,4];
fresh inference keys execute T512. A known-offset oracle is diagnostic only. All
50 native inference runs and 30 calibration-count runs passed independent raw audits;
every measured code was independently reconstructed from saved counts.

Mean output MSE: uncalibrated 0.254024; measured calibration 256:0.039827,
1024: 0.020998,4096: 0.016026; known-offset finite-control oracle 0.014266.
The 4096 probe reduces the synthetic operator-error metric 93.69%, not task error or
hardware energy. Mean estimated-offset RMSE falls 0.063813/0.031901/0.015950.
Probe frequency MSE/theory ratios are 1.00015/1.00012/1.00020. All budgets are retained;
there was no outcome-based selection. Controller estimates are temporary; the
corrected command is one uint8 per free node, plus modes/ranges/bias metadata and
larger actual runtime arrays.

Executed calibration draws total 124,853,944,320 free-node evaluations; the 50
inference runs total 59,454,259,200. One selected calibration budget reused over
Nreuse eight-context bank invocations has draw ratio 1+Ncal/(512*Nreuse), giving
9x/1.08x/1.008x for Ncal 4096 at reuse 1/100/1000. This assumes offsets remain fixed;
probe/readout/programming/accumulation costs differ, so these are not energy or
latency estimates. [Measured calibration report](../../results/thrml_spw/CALIBRATED_OPERATOR_REPORT.md).

All inputs were already consumed old development activations. No fresh neural
confirmation text was used. These scenarios do not characterize real Extropic
error distributions, gain error, drift, physical control stability, area or energy.
The goal remains active. Next software work should relax the unit-gain/fixed-offset
assumptions with identifiable multi-probe calibration and drift/recalibration cost,
then assess broader model conversion with validation-only choices and fresh reserved
confirmation. Physical hardware characterization and system cost remain outstanding.

## Equal-budget gain/offset identification in official THRML

Previous goal turn classification: progress. It added independently audited fixed-
offset calibration, native matched-resource controls and a fixed-seed diagnostic.
This continuation also makes verified progress, rather than a status-only update.

The same 2,359,296-weight operator now has a frozen prospective test of one- versus
two-probe calibration. Device field is float32(a*b_command+d). Offsets have synthetic
SD 0.2; gains are either exactly 1 or positive lognormal mean 1, log-SD 0.2. Gain/offset
realizations remain fixed. These dimensionless laws are stress scenarios, not measured
Extropic specifications. Endpoint clamps remain exact and uncalibrated.

One probe uses all 256/1024/4096 reads at finite code 128 and assumes a=1. Two probes
split the same total budget equally across actual codes 96/159. A smoothed-logit line
fit identifies gain and offset without true parameters or task labels. Inferred gain
uses a predeclared [0.25,4] guard (13 low activations, zero high activations total).
Both methods produce nearest eight-bit bounded commands, keep one logical p-bit per
normalized weight and execute T512 with independent inference keys.

At varying gain and 4096 total reads, one-probe mean output MSE 0.050640 becomes
0.019033 with two probes: 62.42% lower. Paired MSE difference is -0.031607 +/-0.001012
under the stated approximate ten-realization normal summary. Gain-estimation RMSE
falls from the unit-gain assumption error 0.20203 to 0.02588. Offset RMSE is larger
for two probes because each measurement gets fewer reads.

Negative controls are preserved: with actual unit gain, two 4096 MSE 0.018296 is 13.65%
worse than single 4096 MSE 0.016098. Under varying gain at 256 reads, two-probe MSE
0.079983 is 8.26% worse than one-probe 0.073880; at 1024 reads two probes improve to
0.031262 versus 0.055484. No budget or method is selected from these outcomes. The
six displayed normal contrasts are not multiplicity-adjusted formal coverage or
equivalence tests; ten realizations vary both fixed devices and sampling streams.

All 160 native inference and 180 calibration-probe runs passed separate raw audits:
actual packed bits, true-simulation parameter hashes, counts, independently fitted
commands, every conditional expectation/error and resource counter. Unit tests
recover known gain/offset pairs, finite inverse commands and guard behavior. The
unchanged native field/count path retains its separate CPU/CUDA public-factor
trajectory checks. Every recorded native manifest source version has an exact
current or archived copy.

Executed calibration draws: 499,415,777,280 free-node evaluations; inference draws:
190,253,629,440. Equal probe draws do not imply equal programming cost: two probes
need two command settings. Corrected codes still use one uint8 per free node plus
modes/ranges/biases; two count arrays and gain/offset estimates add temporary memory.
Readout, programming, drift and physical accumulator costs remain unmeasured.

[Full gain/offset report](../../results/thrml_spw/GAIN_CALIBRATED_OPERATOR_REPORT.md),
[independent audit](../../results/thrml_spw/gain_calibrated_operator/evidence_audit.json),
[scope and remaining requirements](SCOPE_AUDIT.md).

The full goal remains active. Next comparisons must give calibrated K4 replicas the
same opportunities and match total probe and inference draws; do not compare a
calibrated K1 design against an uncalibrated K4 design as a systems win. Drift and
recalibration remain open, as do stronger multi-checkpoint native training, broader
model conversion with fresh confirmation, and physical-system measurements. The
scope audit preserves these requirements rather than treating scalar proof or this
isolated operator as full completion.

## Calibrated K1/K4 comparison with matched total probe and inference draws

Previous goal turn classification: progress. It established native gain/offset
identification with independently audited finite controls and preserved the cases
where extra parameter fitting hurts. This continuation also changes authoritative
state: the calibration-aware spatial/temporal comparison is now executed and audited.

The frozen study has one 2,359,296-weight normalized-weight operator, eight old
consumed development contexts, ten paired device/sampling realizations, and K1/K4
explicit replica banks. Gains are synthetic positive lognormal mean 1/log-SD 0.2 and
field offsets have SD 0.2. Device parameters stay fixed. K1 shares the first K4
physical-simulation device realization; other replica errors are independent.
Endpoint clamps are exact. This is not a measured hardware distribution.

Both representations use two actual finite probes (codes 96/159), the same count-
based gain/offset fit and predeclared gain guard, and corrected eight-bit codes.
No true parameters enter the measured estimator. For P=256/1024/4096 total probe
draws per unique weight, K1 gets P/2 samples per setting and K4 gets P/8 on every
replica. Inference uses T512 versus T128. Total calibration updates, inference
updates, logical reads and nominal binary-product counts match at each budget.

Mean MSE K1/K4: P=256,0.080493/0.073615; P=1024,0.030970/0.030633;
P=4096,0.018626/0.018672. At P=4096 the paired K4-minus-K1 difference is
+0.00004587 +/-0.00061351 under an approximate ten-realization normal margin.
Its interval includes zero but does not establish equivalence. At P=256 the K4
advantage (-0.006878 +/-0.001707) remains preserved; no favorable-budget selection
or significance-based extension occurs. Oracle and uncalibrated controls are also
reported, without pretending oracle access to true parameters is implementable.

Resource costs are not collapsed into a systems claim. Logical nodes are 2,359,296
versus 9,437,184; corrected command bytes 2,322,432 versus 9,289,728. The two calibration
count arrays occupy 18,579,456 versus 74,317,824 bytes; replica output accumulators
occupy 1,597,440 versus 6,389,760 bytes. Shared unique group-range metadata is 1,179,648
bits in both. At P=4096, calibration sampling depth is 4096 versus 1024 and inference
depth 512 versus 128. Each representation uses 9,512,681,472 calibration free updates
and 1,189,085,184 inference updates per case. Actual runtime includes additional
arrays; logical nodes and these byte subtotals are not physical area measurements.

All 100 inference and 120 native probe runs passed independent audits of actual
packed modes, device/field hashes, saved counts, independently fitted codes,
conditional means/variance/MSE, output archives and every resource counter. New
CPU/CUDA trajectory tests verify distinct K1/K4 states and counts against separately
constructed official factors, same-path inference reconstruction and K1 identity
with the earlier counter. Source history includes exact copies of every version
listed by the native manifest. The controls-scope note distinguishes inherited
nominal target labels from actual differing realized marginals.

[Full report](../../results/thrml_spw/CALIBRATED_MATCHED_OPERATOR_REPORT.md),
[raw audit](../../results/thrml_spw/calibrated_matched_operator/evidence_audit.json),
[resource ledger](../../results/thrml_spw/calibrated_matched_operator/resource_ledger.json).

The goal remains active. This closes one bounded calibration-aware matched-budget
comparison, not physical-system superiority or full-neural quality. Other fitting
procedures, probe settings, correlated device errors and allocation policies remain
possible. Next work should test calibration retention and drift/recalibration cost,
then strengthen calibration-aware full-network inference, native training across
independent checkpoints and broader conversion with fresh reserved confirmation.
The scope audit keeps physical hardware and relevant official comparator gaps open.

## Clocked native drift during probes and inference

Previous goal turn classification: progress. It executed the calibrated K1/K4
matched-resource comparison and independently verified its counts, errors and
source history. This continuation also makes verified progress through a new
native retention study and independent streamed-path audit.

The same 2,359,296-weight K1 operator uses fixed synthetic positive gains (mean 1,
log-SD 0.2), initial field-offset SD 0.2 and Brownian offset drift. An interval means
512 modeled Gibbs-sweep ticks, not seconds. Offsets change after each 64-sweep
segment with SD sigma/sqrt(8), sigma in [0,0.01,0.04]. The full path advances during
calibration, inference and idle gaps. Endpoints remain exact clamps. This is a
specified discrete synthetic model, with no physical clock or drift measurement.

A calibration has two finite eight-bit probes, 2048 samples each (4096 total), so it
lasts eight clock intervals. It pools 32 native count segments per probe, fits a
smoothed-logit gain/offset line without true parameters/path/labels, guards estimated
gain and emits bounded eight-bit commands. Since fields move within those pooled
counts, a stationary-binomial interpretation would be incorrect.

Initial calibration spans segments 0..63. Queries start at 64/128/320 (ages 0/8/32
intervals after that calibration). Retained controls use the initial fit. Refresh
uses a new eight-interval probe ending at the same query start. These are separate
counterfactual schedules sharing an exogenous absolute-time path; refresh occupies
a gap in which retention idles. Age 0 has one shared baseline, not two independent
runs. Programming/fitting/settling time and idle physical transition energy are not
modeled, so this is not a throughput or steady periodic-policy result.

At drift SD 0.01 and age 32, MSE retained/refreshed is 0.040187/0.020549 (ratio 0.51134).
At SD 0.04 and age 32, it is 0.347454/0.052109 (85.00% lower after refresh), paired
difference -0.295344 +/-0.004128 under the approximate ten-realization normal summary.
Zero-drift refresh contrasts include zero at both later ages; retained conditional
means agree exactly across ages. All ages/rates are preserved with no threshold,
interval or policy selection. Six normal contrasts are not multiplicity-adjusted
formal coverage and concern one old input tensor rather than model accuracy.

Refresh has residual error: zero-drift query MSE is about 0.0187, versus about 0.052
under the faster drift even immediately after a probe. Probe-end offset-fit RMSE
rises from about 0.02558 at zero drift to 0.07134 for the late faster-drift probe;
gain-fit RMSE rises from about 0.02589 to 0.04274. Drift between low/high probes can
be misidentified as gain error, and old counts lag the ending field state.

Conditional query moments use the average probabilities across all eight segments.
Sampling variance is scaled by average q_j*(1-q_j)/512, not qbar*(1-qbar)/512 or
an endpoint probability. An exact 16-trajectory enumeration separately verifies
this decomposition and rejects the stationary substitution. It does not imply
unconditional 1/T convergence with drift; random path mean error is retained.

All 90 calibration events (180 probe windows, 5760 native count segments) and 150
native queries (1200 linear segments) passed independent audits. A separately
streamed Brownian process reproduces every one of 6960 active field hashes, every
count-derived code, conditional query mean/error/variance and active-plus-idle
clock identity. Actual calibration/query free-node evaluations total 856,141,332,480
and 178,362,777,600. The common initial calibration is reused as a source artifact;
per-counterfactual costs still charge it. Retain uses 4608 active sweeps including
initial calibration/query; refresh uses 8704. At age 32 both end at 20992 clock sweeps,
with retention 16384 and refresh 12288 idle sweeps. Probe and device programming
costs are not assumed equal or free.

CPU/CUDA trajectory tests validate changing-field segments against separately built
official factors and preactivation accumulation. Full truth paths are transient
simulation buffers, not implemented device/controller memory; source/data hashes,
counts, output archives and exact source history remain reproducible.

[Full drift report](../../results/thrml_spw/DRIFT_RETENTION_OPERATOR_REPORT.md),
[path audit](../../results/thrml_spw/drift_retention_operator/evidence_audit.json),
[clock/cost ledger](../../results/thrml_spw/drift_retention_operator/clock_cost_ledger.json).

The goal stays active. Next work should strengthen calibration-aware full-network
quality using validation-only quality/cost targets, then independent-checkpoint
native training and broader conversion with fresh confirmation. Drift-aware
tracking/refresh policies and physical sampling/stability measurements remain
open; this retention curve alone does not choose a deployable hardware interval.

## Complete direct-p=w networks with measured device calibration

Previous goal turn classification: progress. It established clocked operator drift,
explicit probe/idle costs, independent field-path reconstruction and nonstationary
moment checks. This continuation makes verified progress from isolated operators
to full learned networks under static nonidealities.

The frozen protocol covers five existing 64->32->10 positive-weight MNIST exports,
ten joint device/sampling realizations each, every 10000 test image and T64/256/1024.
Each matrix weight is directly its intended p-bit probability; no affine ranges are
used. Each model has 2368 weight coordinates, 42 deterministic neural biases, and
344--413 exact endpoint clamps. Free fields are finite eight-bit codes in [-4,4].
Endpoint flags are counted; physical exact clamps and finite-endpoint leakage are
not validated by this result.

Static synthetic gains are lognormal mean 1/log-SD 0.2, and offsets have SD 0.2.
Ideal8, uncalibrated, measured P1024/P4096 and known-parameter diagnostics all share
inference random numbers for paired comparisons, but not identical bit states.
Measured calibration pools actual THRML counts at two finite probes and receives no
true parameters or task labels. The same device properties persist across batches.
The standard test images were held out from training but used in earlier studies;
this is not fresh-data confirmation or a clocked full-network experiment. All
budgets/conditions are fixed and retained without outcome-based selection.

At T256, mean accuracy is 88.461% calibrated P4096, 78.329% uncalibrated and 88.597%
ideal8. FP32 is 89.880%. The predeclared primary paired recovery is +10.132 percentage
points +/-1.282 points under the descriptive five-model normal margin. At T1024,
P4096 accuracy 89.371% differs from FP32 by -0.509 +/-0.152 points. P1024/T1024 is
88.621%, so the lower probe budget is not silently promoted to the stronger result.

UINT4/UINT8 numerical controls score 89.336%/89.922%. Calibrated P4096/T1024 differs
from UINT4 by +0.035 +/-0.272 points; this is not superiority. At T256 it remains
0.875 points below UINT4. Integer controls have no inference-time stochastic draws
in this benchmark, and their arithmetic/hardware costs are not matched by accuracy.
Normal margins use 1.96*SD/sqrt(5) across trained-model means after averaging ten
joint device/sampling realizations; they do not separately identify hardware and
Monte Carlo variance, or provide formal five-model coverage/equivalence guarantees.

All 750 native full-test evaluations and 200 native probe runs passed independent
raw audits of source/model/data hashes, counts, fitted codes/fields, expected-weight
controls, every score and resource counter. A second CUDA audit independently
constructs public THRML graphs, counts exact sampled state frequencies and reconstructs
both layers for every example in every run, independent of the production output-
sum observer. Largest scaled logit error is 5.14e-6, below 2e-5 FP32 summation tolerance.
CPU/CUDA unit tests additionally detect activating before averaging and verify
calibration/inference device-coordinate correspondence.

Each batch shares a weight realization across 32 images; 313 independent batch chains
cover 10016 padded rows and discard 16 rows. Vectorization and separate calibration/
layer graph objects are software representations of the same device coordinates,
not a claim to extra or duplicated physical capacity. Actual simulation RAM includes
those graph/state copies, fields, truth arrays and all proof artifacts.

Production calibration/inference free-evaluation counts are 509,747,200 and
209,410,521,600. Independent replay adds another 209,410,521,600 verification updates,
not new accuracy replicates or candidate inference cost. One P4096 calibration reused
across 313 banks adds 5.11% active free evaluations at T256 or 1.28% at T1024 relative
to the corresponding inference evaluation. This assumes static devices; programming,
readout, inactive-device background switching and hardware timing/energy remain
unmeasured. One controller code per free weight plus endpoint modes/neural biases
is not one total storage bit per weight.

[Full-network report](../../results/thrml_spw/CALIBRATED_MNIST_REPORT.md),
[raw audit](../../results/thrml_spw/calibrated_mnist/evidence_audit.json),
[all-example public replay](../../results/thrml_spw/calibrated_mnist/public_path_audit.json).

The full goal remains active for finite-endpoint leakage, clocked nonideal network
quality and validated quality/cost policies, stronger independent-checkpoint native
training, broader pretrained conversion with new confirmation, and real device/
system characterization. Existing scalar, operator, Transformer and negative-control
results are unchanged. The scope audit preserves these remaining requirements.

## Entirely free finite-endpoint full-network continuation

The previous static-calibration result explicitly held 344--413 endpoint weights.
This continuation predeclared four variants: clamped B4/b8 and entirely free B4/b8,
B8/b8 and B8/b10. No retraining, range normalization or test-based configuration
selection occurs. Every free variant has 2,368 native free SpinNodes and zero
clamp blocks. Finite endpoint targets are clipped to sigmoid(-2B) and its complement:
0.0003353501 for B4 and 1.1253516e-7 for B8. They approximate endpoints, not exact
p=0/1. Learned original FP32 weights and deterministic 42 neural biases remain unchanged.

Five trained models x ten full-coordinate joint device/sampling realizations x
four encodings x four conditions x two budgets produced 1,600 full-test evaluations.
Each uses all 10,000 existing test images, T256/T1024 per layer, accumulation before
ReLU, static gain log-SD0.2/mean1 and field-offset SD0.2. Full-coordinate errors align
by original weight index across graph geometries; identical random state paths across
free/clamped geometries are not assumed. Measured calibration uses 4,096 native probes
per free device, split over two finite codes near -1/+1; the true device parameters
and task labels never enter the count-based fit. All 400 probe windows are retained.

Primary calibrated T1024 accuracy is 89.334% for free B4/b8 versus 89.363% paired
clamped B4/b8; paired difference -0.0292 points +/-0.1225 approximate normal margin
across five model means after averaging ten joint realizations. FP32 is 89.880%.
Predeclared secondary B8/b8 and B8/b10 variants score 89.256% and 89.414%. Conventional
UINT4 scores 89.336%; these observations do not establish stochastic superiority.
All configurations/conditions/budgets, mean-network diagnostics and paired contrasts
are retained. The test set was already consumed in prior studies; no fresh-data claim
or formal equivalence test is made.

All raw fields/codes, finite endpoint transforms, active indices, paired device
parameters, calibration count fits, programmed means, full scores and resource
counters passed independent audit. Separately constructed public THRML factors
replayed every example in every one of 1,600 evaluations, with exact integer state
frequencies and independent linear reconstruction. Maximum scaled preactivation/
logit errors 1.82e-6/6.00e-6 are below the predeclared 2e-5 F32 summation tolerance.
CPU and CUDA golden-path tests also verify finite code extrema, uint8/uint16 storage,
both layers and the absence of all clamp blocks in free variants. Production source
versions are preserved and independently resolved to current or archived content.

Free variants have 18.923% more active sampled nodes/updates than the clamped control
at equal T and P. This comparison is not matched active-evaluation cost. Eight-bit
free command arrays use 2,368 bytes; ten-bit logical commands use actual 4,736-byte
uint16 arrays, with no achieved ten-bit packing claimed. No endpoint mode flags are
needed for entirely free variants. Other field/state/calibration arrays, neural biases
and programming information remain additional costs. Production executes 1,862,696,960
calibration and 728,780,185,600 inference free-node evaluations. Independent replay
repeats inference work for verification, not independent statistical replication.

[Report](../../results/thrml_spw/FINITE_ENDPOINT_MNIST_REPORT.md),
[raw audit](../../results/thrml_spw/finite_endpoint_mnist/evidence_audit.json),
[public-path replay](../../results/thrml_spw/finite_endpoint_mnist/public_path_audit.json).

The full goal stays active. Finite-endpoint feasibility now has bounded full-network
support under synthetic static conditions. Clocked nonideal full-network quality,
independent training checkpoints, larger calibrated pretrained conversion, actual
controller span/precision and physical energy/latency/device measurements remain open.

## Full-network drift and scheduled-refresh continuation

Previous goal turn classification: progress. The entire all-free finite-endpoint
full-network study closed with 1,600 complete native evaluations and independently
replayed examples. This turn advances a previously open requirement: bias drift
throughout complete network inference, including actual measurement-based calibration.
No blocking condition or outstanding sampler remains at closure.

New fixed experiment uses all 2,368 weight coordinates as free p-bits, finite eight-bit
fields [-4,4], endpoint epsilon=sigmoid(-8), 42 deterministic neural biases, five frozen
positive-weight MNIST models and five joint device/path/sampling seeds each. All
10,000 old test examples are evaluated in one fixed shuffled order, batch 32, with 16
padding rows discarded. T1024 per layer is split into sixteen 64-sweep segments;
segment linear outputs are averaged before ReLU. No premature nonlinear averaging
or neural weight reconstruction is used in production. CPU/CUDA trajectory tests
reject the incorrect ordering and verify public state counts and clock partition.

Static gain is positive lognormal mean 1/log-SD 0.2 and initial field-offset SD 0.2.
Independent Brownian field-offset increments have SD 0,0.01,0.04 per 1024 sweeps;
the process advances for both weight banks during probes, serial layers and idle.
A 4096-sweep two-probe calibration precedes batches 0,32,...288. Drift changes within
both 2048-sample probes. Counts alone fit a smoothed-logit gain/offset line with
predeclared gain guard [0.25,4] and emit inverse finite codes. The model never supplies
true gain, offset, drift or task labels to the measured estimator.

Retain-initial and refresh-every32 compare on the same latent path and absolute clock.
Retain idles in the nine later probe windows; refresh samples them. Both finish at
681,984 native sweeps: 641,024 serial layer sweeps plus 40,960 calibration/idle sweeps.
Resetting segment states is explicit: independent unary conditionals mix in their
first observed sweep, with exactly 64 actual updates per segment. No temporal or
cross-device persistence, readout noise, autonomous background switching energy,
programming/fitting time or real-time device calibration is modeled. Dataset/segment
vmap evaluates the fixed histories in software, not physical parallel capacity.

All 150 cases are retained: at drift SD 0, retain/refresh score 89.374/89.368%; at 0.01,
81.463/88.540%; at 0.04,49.043/78.223%. FP32 scores 89.880%. Primary fast-rate paired
refresh gain is 29.1804 percentage points +/-1.9524 approximate normal margin over five
trained-model means after averaging five joint realizations. Slow-rate gain 7.0772
+/-1.1372 points; zero-drift contrast -0.0064 +/-0.0967 points includes zero. These are
small-sample descriptive normal margins, not formal coverage or independence claims
for the twenty-five joint seeds. No test-based tuning or fresh-data claim occurs.

Public replay independently rebuilds free-node THRML factors, absolute clocks, paths,
keys and all 1,550,400 active field hashes. It exactly matches every segment count
from 750 calibration events/1,500 probe windows and reconstructs both neural layers
for every example in 150 full-test cases. Maximum scaled preactivation/logit errors
2.67e-7/2.86e-7 are below 2e-5 tolerance. Separate raw audit recomputes count fits,
finite codes, every full/window task score and clock/resource counters. Production
source versions resolve exactly to retained current/content-addressed copies; model,
input and case archives are hashed. Derived model-level means/margins are also checked.

One complete trajectory executes 758,972,416 observed inference free-node updates.
Retain uses 9,699,328 probe updates; refresh 96,993,280, for 11.356% more total active
observed updates. If artificial comparison idle is removed, refresh needs 5.714% more
active serial sweeps. No hardware latency/energy gain follows from matched clocks.
Production totals 7,274,496,000 calibration and 113,845,862,400 inference updates;
independent replay repeats both only for verification. Commands use 2368 uint8 bytes,
plus neural biases, measured-fit buffers and much larger runtime/history arrays.

The fast-drift quality failure remains explicit: refresh helps substantially but
still trails FP32 by 11.657 points. Probe drift, intra-query drift and finite controller
limits coexist; no causal decomposition or optimum policy is claimed. A slower
synthetic rate permits useful task quality under the fixed policy. Physical rates
and voltage limits remain unmeasured.

[Report](../../results/thrml_spw/CLOCKED_MNIST_REPORT.md),
[raw audit](../../results/thrml_spw/clocked_mnist/evidence_audit.json),
[public replay](../../results/thrml_spw/clocked_mnist/public_path_audit.json).

The full goal remains active. Next software work can use existing training-held-out
validation inputs for prospective quality/cost choices with independently reserved
paths, strengthen native training across independent model checkpoints, and extend
calibrated finite-controller conversion to larger pretrained architectures. The
clocked MNIST experiment establishes a bounded simulated regime, not these broader
requirements or physical-system superiority.

## Five paired seeds of finite-control native probability fine-tuning

Previous goal turn classification: progress. The clocked all-free full-network
experiment closed with complete public replay and retained its fast-drift quality
failure. This turn addresses an earlier weak requirement: probability training had
only one checkpoint. Five independently pretrained starting networks now each run
three paired fine-tuning methods, with independently checked native forward paths,
explicit surrogate gradients, optimizer updates and validation-only selections.

Methods are expected-weight training, native ideal-device training and native
synthetic-device-augmented training. Each runs five epochs over the identical
seed/epoch-specific shuffled 55,000 training rows, batch 256 with 40 repeated padding
examples per epoch. Only 2,368 probability logits are optimized; 42 neural biases are
frozen. Adam lr0.001 clips theta into [-8,8]. Native programs use one free SpinNode per
matrix weight, zero exact clamps, nearest eight-bit bias codes in [-4,4] and T64 linear
accumulation before ReLU. Device augmentation redraws lognormal gain mean 1/log-SD 0.2
and offset SD 0.2 once per minibatch, held throughout that forward. No measured device
model, drift or temporal/cross-device persistence is assumed.

Backward is an explicitly biased straight-through estimator through unquantized
sigmoid expected linears, using the expected second-layer matrix. It does not
differentiate Gibbs samples, quantization or device parameters. The forward uses
actual official THRML samples, not a replacement Bernoulli sampler or noisy-activation
surrogate. This is five independent starting-model seeds with paired methods, not
fifteen independent initializations or new training from scratch.

Each method/model selects minimum native uncalibrated validation CE on all 5,000
validation images averaged over five fixed synthetic device/sampling profiles, among
epochs 0--5 including the initial finite-probability checkpoint. Shared validation
keys/profiles reduce comparison noise but can be overfit by selection. All 15 choices
are frozen before the evaluation program loads test arrays or scores. Reserved test
profile-domain seeds and MC key roots are disjoint from training and selection.
The old 10,000 MNIST test images are not fresh data; no test-driven tuning occurs.

All 600 predeclared full-test cases compare initial/expected/nativeideal/nativedevice
checkpoints on five models x five reserved device/sampling realizations x three
conditions(ideal, uncalibrated, measuredcal4096) x T64/T256. The primary uncalibrated
T64 comparison scores 77.256% after device augmentation versus 75.531% expected control,
with paired model-level gain 1.7256 points +/-0.5278 approximate normal margin. Individual
paired model gains are 1.422,1.902,1.638,1.028,2.638 points. Native ideal training scores
76.269% in that condition. This is a bounded positive robustness result, not recovery
of original deterministic 89.880% quality.

The negative tradeoff is retained: at measured-calibrated T256, expected training
scores 88.741%, native ideal training 88.756%, and device augmentation 88.428%. At ideal
T256, the corresponding scores are 88.964/88.943/88.590%. All deterministic expected,
UINT4 and UINT8 per-checkpoint numerical controls are preserved. Measured calibration
uses 50 native probe windows, 2,048 draws each, counts-only smoothed-logit fitting and
finite inverse commands. Probe evidence is shared across targets on the same static
devices; candidate deployment must still count its own calibration. No true device
parameters or labels enter the measured estimator.

Every one of 10,750 native training updates and 450 complete validation/profile
forwards was independently replayed with separately constructed public THRML
state-frequency programs. All 5,375 expected-training forwards, all 16,125 explicit
surrogate gradients and all 16,125 Adam updates were checked. Maximum scaled forward
error 5.44e-7; worst gradient difference is 0.02247 of the declared 2e-6+2e-4*abs(reference)
tolerance; maximum scaled one-step Adam error 5.30e-8. Validation metrics and every
selected epoch were independently recomputed, including two epoch 0 selections.
All 600 test cases and 50 probe windows passed independent finite-control/calibration,
classical-control, full-score/resource and public-layer path audits; maximum scaled
test-layer error 1.46e-6 is below 2e-5. CPU/CUDA golden tests passed. All production source
versions resolve to exact retained current/content-addressed copies, and derived
model-level means/paired margins are independently checked.

Production native free-node counts are 1,629,184,000 training; 10,707,148,800 validation;
71,153,664,000 reserved-test inference; 242,483,200 shared test calibration. Replay
repeats these for verification, not additional independent model seeds. Per-checkpoint
eight-bit commands occupy 2,368 bytes; float32 theta plus two Adam states alone require
28,416 bytes before gradients, activations, graph/state arrays and histories. The
logical one-p-bit representation is not one total storage bit or a physical training
energy/area/latency improvement. Native arithmetic is external floating simulation.

[Report](../../results/thrml_spw/PROBABILITY_FINETUNE_REPORT.md),
[training/validation/gradient audit](../../results/thrml_spw/probability_finetune/training_path_audit.json),
[reserved-test audit](../../results/thrml_spw/probability_finetune/test_path_audit.json),
[combined evidence audit](../../results/thrml_spw/probability_finetune/evidence_audit.json).

The full goal stays active. Next work should advance calibrated larger pretrained
conversion and validation-selected execution cost/quality, and distinguish measured
or clocked training from this synthetic distribution augmentation. Other gradient
estimators, broader architecture robustness, device/controller measurements and a
relevant official application comparator remain open. Existing negative results
and every unselected checkpoint remain preserved.

## Calibrated finite-control pretrained quarter-model continuation

Previous goal turn classification: progress. Five paired starting-model seeds of
native probability training closed with full training/validation/test audits.
This turn tests a different previously open requirement: measured finite-control
calibration in the actual sequential pretrained Transformer forward, beyond isolated
operators and positive-weight image networks.

The frozen study reserves 64 new WikiText-2 test windows (4,096 next-token targets),
strictly excluding all previously scored positions and legacy overlapping windows.
It fixes T1024, P4096 and three conditions(ideal, uncalibrated, measuredcal4096),
five joint device/sampling seeds and the same first-three-block target subset.
All 12 affine projections contain 21,233,664 of 81,912,576 unique parameters (25.922%).
There is no search, test-driven extension or threshold adjustment. The rest of the
model remains deterministic FP32; this is not full-model or long-context conversion.

Signed weights use local lower/span metadata per 128 input weights/output channel.
Each normalized probability gets one free THRML SpinNode, with zero exact clamps.
Targets are clipped to sigmoid(-8),1-sigmoid(-8), then finite eight-bit fields [-4,4].
This approximates endpoints and is not direct p=raw signed weight. The native worker
loads only codes, ranges and neural biases, and accumulates actual sampled linear
outputs before original nonlinearities. Later student inputs are regenerated after
all earlier native replacements and checked exactly. Sampling states are shared
across banks of eight contexts; software batching is not physical capacity.

Each weight has synthetic static positive gain lognormal mean 1/log-SD 0.2 and offset
SD 0.2, independent across coordinates and fixed across probes/conditions/input banks.
Two 2048-draw native probes at finite codes 96/159 fit a counts-only smoothed-logit
gain/offset line with guard [0.25,4]. Corrected inverse commands use the same finite
grid. True device parameters, text, labels and task scores never enter the fit.
No drift, temporal/cross-device persistence or measured device transfer law is used.

All 15 native quarter-model cases finished. Ideal perplexity 131.597 versus 127.140
FP32 is +3.5057%; token accuracy drops 0.6201 points. Its crossed upper PPL ratio 1.04239
and lower accuracy difference -0.9326 points pass the frozen 5%/1-point limits. With
uncalibrated devices PPL is 1,137.355(8.94570xFP32), accuracy 7.661%. Measuredcal4096
recovers PPL 135.426 and accuracy 22.534%, but PPL remains +6.5174% and fails the fixed
5% point gate. Token loss 0.9033 points passes its point bound, but crossed PPL upper
1.07902 and accuracy lower -1.3086 points also fail the strict margin rule. All failures
are retained; there is no claim that this calibration budget meets the target.

Primary calibration-versus-uncalibrated CE difference is -2.128034, PPL ratio 0.11907
with crossed sensitivity interval [0.108723,0.130357]. Five joint device/MC seeds are
crossed with eight whole 8-context banks in 10,000 predeclared resamples. This is a
heuristic exchangeable-cluster sensitivity analysis, not formal coverage or five
independently pretrained models. Independent affine4/8 numerical controls score
134.728/127.189 PPL; the ideal programmed-mean diagnostic scores 127.645. These
controls remain visible; no conventional-quantization or Extropic-architecture
superiority is established.

Every node in 120 native probe windows and 180 operator executions is independently
replayed using public THRML factor/state-frequency programs. Every field, measured
fit/code, conditional bias/variance statistic, update counter and native output is
recomputed. Maximum scaled operator error 7.85e-6 is below 2e-5 tolerance. Separate
pre/post-hooks on the unchanged original Transformer verify all student inputs
and every final token loss/prediction for all 15 cases. Deterministic baselines,
all prefix links and all 60,000 condition-quality bootstrap values plus both gates
are independently recomputed. CPU/CUDA native tests and a 64-window statistics test
passed. An earlier million-node preflight also passed; it is verification, not an
additional quality replicate.

A reporting-only bug was caught during the native phase: the old summary helper
requires 128 windows, while this prospectively fixed study has 64. The original
immutable driver finished all 15 native cases, closed its worker, and then failed
that guard. Its stderr and exact source are retained. A separate64-window finalizer
applies the unchanged predeclared formula to all completed raw cases, with no new
native sampling or altered budget/token/threshold. A strict reproduction wrapper
catches only that guard after verifying all 15 cases complete; its cached full
execution passes, reproduces the audited manifest exactly and performs no new
native sampling. All recorded source versions across 180 operator manifests resolve
to exact current/content-addressed copies, including historical auxiliary versions.

One 64-context profile executes 173,946,175,488 observed inference updates. Measured
calibration adds 86,973,087,744 (50%); static reuse over R streams gives count multiplier
1+0.5/R only under retained, unchanged devices. This is not hardware energy/latency.
The control/range/bias numeric payload is 22,643,712 bytes (8.53125 bits per converted
weight), actual archives 22,655,544 bytes. Expanded fields/ranges, graph/state objects,
calibration buffers, programming, readout, accumulation and background switching
remain additional. Logical one-p-bit encoding does not mean one total storage bit.

Production native evaluations total 434,865,438,720 calibration and 2,609,192,632,320
inference. Independent full replay repeats both; the one-large-operator preflight
adds 14,495,514,624 verification updates, and small unit tests are separate. None is
extra independent statistical replication or a candidate deployment cost.

[Report](../../results/thrml_spw/CALIBRATED_TRANSFORMER_REPORT.md),
[native replay](../../results/thrml_spw/calibrated_transformer/native_path_audit.json),
[model/score/gate audit](../../results/thrml_spw/calibrated_transformer/evidence_audit.json),
[reporting correction](../../results/thrml_spw/calibrated_transformer/postprocessing_correction.json).

The full goal stays active. The next experiment should distinguish finite-probe
estimation error from the finite inverse-control-range floor before choosing a new
prospective validation-only calibration/headroom policy. Higher budgets cannot be
assumed to fix bounded control saturation. Larger/full-model conversion, broader
contexts, measured/clocked training and physical controller/system measurements
remain open. The failed regime is not relabeled a successful quality demonstration.

## Known-parameter controls isolate the failed calibration regime

Previous goal turn classification: progress. The full measured-calibration
quarter-Transformer study closed with complete native/model/statistical audits,
retaining its 6.52% perplexity gap and failed 5% gate. This turn investigates the
remaining mechanism without changing that result or treating new oracle knowledge
as an implementable calibration method.

The frozen counterfactual study reuses the same 64 consumed WikiText-2 windows
(4,096 targets), five synthetic device/MC profiles, eight-bit bias grid [-4,4], T1024,
21,233,664 all-free normalized-weight nodes and first-three-block subset (25.92%).
Native keys exactly match the old 1140730 seed/stage/bank schedule. It predeclares
all 16 mean-model forwards and five complete native known-parameter oracle8 runs.
No new probes, fresh-data claim, quality gate, training, parameter search or
outcome-based extension occurs. True gains/offsets enter only explicit diagnostics.

The four field counterfactuals are nominal ideal8, old measuredP4096 codes, exact
known inverse clipped/rounded to the same 8-bit[-4,4] grid, and exact known inverse
clipped to the same range without grid rounding. Fields are emitted F32 after F64
arithmetic; analytic sigmoid marginals give ordinary F32 mean-weight matrices.
A mean-weight nonlinear network is not the expectation of the sampled network.
Original measured codes are independently refit from immutable native counts;
no true parameters or labels enter the measured estimator.

All five native oracle forwards complete with PPL 131.676 versus 135.426 measured
and 131.597 ideal. Original FP32 remains 127.140. The oracle-measured paired CE
contrast is -0.028079, PPL ratio 0.972311, with crossed seed/bank CE sensitivity
interval [-0.038188,-0.019237]. Token accuracy improves 0.249 pp, but its interval
[-0.054,+0.615]pp includes zero. Oracle-ideal CE contrast +0.000603 has interval
[-0.002150,+0.003069], which includes zero and does not establish formal equivalence.
The old failed measured-calibration quality gate is not upgraded by an oracle
that knows unobservable true device parameters. This is diagnostic evidence.

Mean-model PPL is 127.645 ideal, 131.227 measured, 127.713 known-grid oracle and 127.635
known-continuous bounded oracle. The exact weight-error telescope is
Wmeasured-Wideal=(Wmeasured-Woracle8)+(Woracle8-WcontinuousB4)+(WcontinuousB4-Wideal).
Across 106,168,320 coordinate/profile pairs, squared measured-to-known replacement
effect is 8.71470e-5, grid-to-continuous 8.85956e-6, bounded-continuous-to-ideal 3.25444e-8.
The signed estimate/grid cross contribution is -1.77096e-5; all cross terms are retained,
closing total measured-to-ideal MSE 7.83295e-5. The known inverse clips for 0.8552% of
coordinates. Squared replacement effect is 2677.8 times the squared range effect,
but these are not independent additive task-loss attributions or percentages.
The estimator replacement includes finite-data estimator choices and rounding
interactions, not pure unbiased calibration variance alone.

Independent analytic replay regenerates all 240 fields and mean matrices, verifies
all 60 moment identities, refits actual old counts-to-codes, and reproduces every token
loss/prediction in all 16 mean forwards. Public THRML frequency programs replay every
node of all 60 new native oracle operators; independent original-model pre/post-hooks
verify all inputs and final token losses/predictions of the five complete native
cases. Independent task audit recomputes all 40,000 paired bootstrap values, task
scores and update counters. Resampling inherits the old exchangeable 8-context-bank
heuristic, 10,000 draws, fixed seed 2026114701; no formal coverage or new gate is claimed.

New native sampling executes 869,730,877,440 free-node updates, with independent
native verification repeating that count. Zero new probe runs reflects oracle
knowledge and reuse of old measured evidence, not zero-cost implementable calibration.
The original control/range/bias payload remains 8.53125 bits per converted weight.
No physical energy/latency, memory density or full-model advantage is established.
Original target range metadata and external arithmetic remain explicit resources.

All recorded source versions across 60 new operator manifests resolve to exact
current/content-addressed copies. A completed-cache reproduction wrapper verifies
all five model cases, 60 operators and 16 mean archives without rewriting elapsed-time
metadata, changing the audited manifest or drawing new native samples. The known
inverse/rounding/clipping/telescope unit check passes. No old result is overwritten.

[Report](../../results/thrml_spw/TRANSFORMER_CALIBRATION_FLOOR_REPORT.md),
[mean/field/moment audit](../../results/thrml_spw/transformer_calibration_floor/mean_evidence_audit.json),
[native replay](../../results/thrml_spw/transformer_calibration_floor/native_path_audit.json),
[combined audit](../../results/thrml_spw/transformer_calibration_floor/evidence_audit.json).

The goal remains active. This bounded synthetic regime prioritizes improving probe
placement, count allocation or estimation over widening controller range. A practical
new method still needs resource-matched validation-only selection and separate
confirmation; no particular probe scheme or larger budget is proved successful.
Broader/full-model conversion, measured/clocked transfer laws, correlated hardware,
training and physical controller/readout/system costs remain open.
