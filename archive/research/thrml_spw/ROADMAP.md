# Long-horizon research goal — active

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

Objective: demonstrate direct w=p encoding on Extropic's official simulator, then
determine which useful computations and resource regimes support it. Evidence must
come from runnable THRML programs and preserve negative as well as positive findings.
This goal is not complete after the scalar demonstration.

## Milestones and acceptance criteria

1. **Official baseline — established.** Pin THRML source and dependencies; reproduce
   the official Ising example against exact enumeration. Preserve manifests.
2. **Representation — established.** Use one logical SpinNode per w in [0,1], with
   analytically derived fields and explicit endpoint handling. Verify target means,
   MSE law, initial-state independence, and control rounding over the prescribed
   scalar sweep. Preserve raw prefix counts and representative trajectories.
3. **Operator schedule — initial demonstration established.** Accumulate native
   p-bit linear outputs before nonlinearities. Compare the same Gibbs paths against
   reconstructed weights, and separately measure the bias of activating each sample
   before averaging. State which arithmetic runs outside the probabilistic substrate.
4. **Learned positive-weight network — initial demonstration established.** Train reproducible small networks with
   learned matrix weights constrained to [0,1]. Export exact weights and test examples;
   execute each layer with native THRML probability banks and preactivation accumulation.
   Report deterministic baseline, accuracy/CE, T, seeds, clamps, and node/controller counts.
5. **Robustness and efficient execution.** Vary field precision/range, temporal
   schedules, coupled-node effects, seeds, and dynamic range. Test confidence stopping
   and nonuniform sample budgets on the successful execution schedule, with independent
   validation data. Preserve configurations and account for stopping-policy bias.
   Finite-control scalar/native-neural sweeps and exact/native small coupled-bank
   characterization plus marginal calibration are now established. Validation-selected
   per-layer budgets and fixed-point accumulation now have five-model/five-sampling-seed
   full-test confirmation, including their joint use with finite controls. Online
   confidence stopping now has independent-case path checks and five-model/five-seed
   batch-shared confirmation with matched-budget references. Joint finite-format
   adaptive execution, broader calibration/drift and generalization remain open.
6. **Fair baselines and resource comparisons.** Reproduce an official Extropic
   application/baseline relevant to the claimed comparison. Distinguish neural weights
   from EBM coupling/control parameters. Compare equal quality, transitions, physical
   parallelism, accumulator precision, controller storage, and programming reuse.
7. **Larger networks and training.** Expand only from a verified small execution path.
   Compare signed affine mappings separately from direct w=p, retrain for native noise,
   repeat central findings on full test sets and independent seeds, and quantify where
   sampling costs overwhelm useful quality.
8. **Physical validation — requires access.** Use measured probability/bias transfer,
   retained control precision, switching/correlation, readout and accumulation costs.
   A software result alone cannot certify physical density, energy, or latency.

## Reporting rules

- A positive result means its stated criterion passed; do not silently upgrade a
  representation proof into an end-to-end hardware advantage.
- A negative result changes claims and experiment design without canceling unrelated
  milestones. Fix invalid implementations before treating them as evidence.
- Existing NumPy/PyTorch pilots are background results, not THRML proof.
- Use exact source revisions, seeded initialization, shared evaluation data, raw
  counts/output records, and uncertainty intervals for central comparisons.
- Finite-control endpoint approximations, clamping, learned probabilities, stored bias
  precision, and conventional arithmetic must always be explicit.

## First run ledger

See `results/thrml_spw/official_example/manifest.json`,
`results/thrml_spw/scalar_full/manifest.json`, and
`results/thrml_spw/preactivation/manifest.json` for completed native runs.
Five seeded positive-weight MNIST models and full-test native inference are recorded
in `results/thrml_spw/positive_training/` and `results/thrml_spw/positive_full_test/`.
The immediate next work is finite-control precision, matched-resource comparisons,
and lower sample budgets on this verified execution schedule.

That initial queue has now produced bounded evidence; subsequent ledgers below
record its results and the remaining work rather than restarting it.

## Second run ledger

- `results/thrml_spw/control_sweep/`: 16 actual-THRML field-control settings; measured
  MSE matches programming-bias plus sampling-variance predictions.
- `results/thrml_spw/control_neural/`: six-bit range [-4,4] chosen using validation
  only, then measured over all 10,000 test images and five checkpoints.
- `results/thrml_spw/correlations/` and `coupling_calibration/`: exact small Ising
  kernels, native coupled trajectories, and fields recovering requested marginals.
- `results/thrml_spw/matched_resources/`: actual native spatial/temporal sample
  budgets, explicit controller/read/update/depth counts, deterministic codes separate.
- `REFERENCE_MAPPING.md`: checked roles of sampled variables versus programmed model
  parameters; no official four-p-bits-per-learned-weight baseline established.
- Corrected native sampler-aware training passes its schedule-equivalence test. A
  full-test evaluation across five sampling seeds found modest low-T gains on one
  checkpoint but no advantage over matched UINT4. Wrong-schedule runs are excluded
  from target evidence and preserved in a marked archive.

## Third run ledger

- `layer_budgets/`: 25 layer allocations chosen on validation only, then all
  10,000 test images, five trained models and five sampling seeds each.
- `layer_budget_matched/` and `layer_update_matched/`: uniform integer budgets
  bracketing selected logical reads and actual free-node updates respectively.
- `accumulator_precision/`: native fixed-point observers, explicit rounding and
  saturation, validation-only format choice and full-test confirmation.
- `accumulator_bias/`: exact four-state expectation versus native trials showing
  a persistent rounding floor; extra samples do not remove arithmetic bias.
- `joint_efficiency/`: factorial confirmation of independently selected finite
  controls, accumulator precision and layer budgets, with no further selection.
- `efficiency_audit.json` recomputes raw scores, choices and accounting;
  `EFFICIENCY_REPORT.md` and standalone figures present the results and limits.

Those priorities have advanced in the fourth ledger below. Controller/readout
implementation and hardware data remain necessary for physical claims; their
absence does not block continued software research.

## Fourth run ledger

- `adaptive_native/`: actual online independent-case stopping and compaction;
  exact retained-path replay identity, first-layer recomputation, raw moments and
  padding-inclusive matched-budget references across five models.
- `adaptive_shared/`: actual online calibration with shared weight states, followed
  by full-test five-model/five-sampling-seed confirmation. Case compaction changes
  later shared paths; actual decision traces and bank counters are preserved.
- `adaptive_bias/`: 10,000 native trials versus an exact finite-horizon law showing
  that margin/SE stopping can bias a returned mean even for one p-bit.
- `ADAPTIVE_REPORT.md` and `adaptive_evidence.png/.pdf`: resource/quality evidence,
  independent-case versus shared-bank distinctions, conditional-uncertainty limits,
  confidence arithmetic and storage costs. Twelve native tests and both raw audits
  pass; physical energy or coverage guarantees are not established.

Those next priorities advanced in the fifth ledger below. Adaptive finite-control/
accumulator composition still needs its own validation. Hardware costs remain a
separate empirical requirement.

## Fifth run ledger

- `fashion_training/`: five larger positive/signed paired models, identical
  initialization and minibatch order, checkpoint selection on validation only.
- `fashion_native/`: 240 validation conditions and 300 full-test conditions; direct
  p=w versus explicit per-output affine mappings; gates, failed direct candidate,
  matched updates, metadata and representation-specific storage accounting.
- `range_native.py` and its same-path test: original affine design implemented with
  actual THRML states, including signed weights and safe constant groups.
- `transformer_preparation/`: pinned real DistilGPT-2 and WikiText-2 assets,
  short-context nonoverlapping validation/test windows, exact FP32 output replay.
- `transformer_projection_native/` and `transformer_replay/`: one actual native
  589,824-weight projection, linear variance law, downstream language-quality
  evaluation, calibration-only gate and raw token evidence. Only 0.72% is converted.
- FASHION_REPORT.md and TRANSFORMER_PROJECTION_REPORT.md present positive and
  negative results with standalone figures. Raw audits pass; thirteen native tests
  pass. Field controls/external arithmetic remain float32 on these larger routes.

Next: convert an entire Transformer block in the real forward schedule, regenerating
each later operator's inputs after earlier native replacements. Then evaluate larger
converted fractions, probability-aware training, and realistic finite controls.
The current single-operator success does not satisfy a full-model viability gate.
Physical controller/readout costs and a relevant Extropic application baseline
remain separate requirements; no hardware/system advantage is established.

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

## Nested local-range operator study (2026-10-27)

New local-range development study: group16/T512 gives raw geometric operator MSE
ratio 0.670558 versus group128/T1024, across all 24 projections,
five seeds and 360 native operators on consumed validation inputs. All 72 theory
checks pass the frozen 20% tolerance, and every independent frequency reference
satisfies its IEEE envelope. The strict byte-exact F32 replay gate **fails in
270/360 cases**; this is preserved, not relabeled as a fully passed audit.
The control/range payload rises from 8.53125 to 12.03125 bits per converted weight
while observed node updates halve. No full stochastic neural quality confirmation
or physical advantage follows. Persistent storage has less than 0.5 GB free;
at least 10 GB or another persistent path is requested for full-model confirmation.
[Local-range operator report](../../results/thrml_spw/RANGE_LOCALIZATION_REPORT.md). The broad research goal remains active.

Original weights, all controls, ideal coordinate inequalities, original-model teacher inputs, deterministic mean-model scores and all raw MSE/resource statistics were independently checked. Ideal nested-interval variance decreases at every coordinate; finite eight-bit fields have explicit exceptions (7,267 group16 coordinates increase versus group128). Group64 mean-model perplexity is worse than group128, so variance monotonicity is not task-quality monotonicity. No new test text or full stochastic student network was evaluated.

Two failed replay prefixes and a fresh production/public diagnostic are retained. All 360 final public replays complete; exact F32 gate fails in 270 cases (max absolute difference 6.103515625e-5), while all frequency/IEEE references pass. Fourteen descriptive 2e-5 flags fail and remain visible. Even fresh production execution differs from the preserved output; the numerical cause is unresolved. The combined audit correctly reports passed=false. No criterion was relaxed.

This goal turn made concrete progress: full-weight variance theorem checks, prospective full-size native operator evidence, explicit metadata tradeoffs, and a bounded numerical reproducibility failure. The prior allocation turn also made concrete progress, with exact budget matching but no confirmed quality improvement; its negative outcomes stay unchanged.

Next: isolate numerical execution reproducibility; after additional persistent storage is actually available, freeze a fresh unused-text end-to-end comparison of finer ranges and coarse control. Preserve failed quality gates. Nonideal/correlated/clocked transfer, broader models and physical system costs remain open. The pending storage question is a resource request, not an approval inferred from silence. No physical hardware or Extropic-superiority claim is established.

## Completed fixed-compiler check and confirmation implementation (2026-10-27)

The earlier in-progress numerical follow-up is now complete: all 72 cases (all 24 affine matrices, three control/sample configurations, seed0) match exactly across two independently started processes with XLA autotuning disabled. Public-factor/SumOnly replay also matches every output and count hashes match the previous audit. The original 360-case default-compiler gate remains failed; there is no retroactive quality or numerical criterion change. Maximum relative target-MSE difference from original seed0 outputs is about 1.99e-8. Historical compiled executables were not saved, so the original cause remains incompletely identified. This is scoped numerical reproducibility, not a full model or hardware result.

The prospective end-to-end fine-range confirmation now has CPU driver, separate native worker, independent original-model/public-factor auditors, paired-statistics audit and atomic result writes. Two isolated gate tests pass; the real preflight refuses execution for insufficient persistent storage before Torch/JAX model/runtime imports. No model has been scored on the reserved 32 windows, no execution-source seal or native confirmation manifest exists, and full integration remains unverified. Source snapshots and implementation_readiness.json state that limit explicitly.

The previous goal turn made concrete pilot/data-reservation progress. This goal turn adds completed broad compiler replication, actual source implementation and executed gate tests. It also identifies a possible resource remedy: a read-only probe of 66 historical single-link F32/F64 arrays restores every original byte, including headers, and estimates about 10.23 GB savings over 68.55 GB of eligible files (sample-max-ratio estimate 9.90 GB). These are extrapolations; no original file was changed. A durable lossless codec, independent disk restore verification, complete migration/restore manifest and explicit historical replay instructions are required before archiving anything. Capacity must actually satisfy the confirmation preflight before scoring.

[Compiler report](../../results/thrml_spw/RANGE_REPRODUCIBILITY_REPORT.md), [independent array audit](../../results/thrml_spw/range_localization/fixed_compilation/evidence_audit.json), [confirmation implementation](../../results/thrml_spw/fine_range_confirmation/README.md), [archival probe](../../results/thrml_spw/archival_feasibility/probe.json). Goal active; no live sampler remains after this checkpoint.

## Storage recovery verified; neural confirmation can start (2026-10-27)

The previous goal turn made concrete progress by testing reversible archival and launching migration. This turn finishes it: 2033 completed-run arrays were losslessly archived, recovering 9,782,951,378 file bytes. An independent decoder verified all 2,970 planned original checksums, including 937 unchanged raw files, against the frozen pre-migration index (68,546,530,560 original bytes). Headers, floating bit patterns, modes and modification times are retained. No scientific value, quality rule or earlier failure changed. Restore archived paths before historical commands using the documented utility.

The real capacity condition now passes. The reserved 32-context fine-range confirmation remains unscored at this closure, with candidate/group/sample/quality rules unchanged. Its report generator has been prepared, but refuses to report before the actual model/native/statistical audits. Next execute the frozen source/sealed comparison, then independent full student-input, public-native, score and paired-bootstrap audits. The broad research goal remains active; storage success is not neural quality or physical hardware proof.

[Integrity audit](../../results/thrml_spw/lossless_archive/evidence_audit.json), [restore instructions](../../results/thrml_spw/lossless_archive/README.md), [next frozen study](../../results/thrml_spw/fine_range_confirmation/README.md).

## Finer-range neural confirmation closed and audited (2026-10-27)

All ten cases, 240 native operators and actual stochastic student inputs are independently replayed exactly. All 60,000 paired seed/bank resampling values, labels, scores, source versions and update counters are checked. Fine group16/T512 gives PPL 134.816 versus coarse group128/T1024 136.834 and fresh FP32 129.541. Fine next-token accuracy is 24.424% versus 24.854% FP32 (loss 0.430 pp). Fine's mean satisfies the original point rule, but strict PPL upper sensitivity +5.957% exceeds 5%; its strict gate remains failed. Coarse fails point and strict. The paired CE interval [-.030653,+.003894] includes zero; no confirmed paired gain is inferred.

Fine halves observed node updates with 41.03% more converted-weight control/range payload (12.03125 versus 8.53125 bits plus 48 configuration bytes). Resource ledger includes the remaining deterministic FP32 parameters, per-bank versus per 32-context counters and normalized scalar effective-bit caveats. Teacher-forced 65-token contexts share each weight draw across 520 input positions per eight-context bank; these counters are not generation latency. One checkpoint, 51.84% conversion, ideal independent devices. Five descriptive 2e-5 flags remain, with all exact-output and IEEE gates passing. No physical or Extropic-superiority result follows.

The original launcher failed only during 60s child shutdown after all inference completed. Its frozen source/log/exit 1 remain preserved. A separately hashed finalizer verified every completed record and rebuilt statistics/manifest with zero new inference or native sampling. Full independent audits then passed. The production manifest stays an immutable pre-audit record; completion.json links the later audited status. All recorded source versions resolve to retained bytes. Only a planned reconstructible temporary input was removed after its original SHA matched the independently regenerated input.

This goal turn made concrete progress: completed inference, diagnosed/preserved launcher failure, recovered without resampling, completed all independent audits and reported the unchanged failed strict quality criterion. Previous storage recovery remains exactly reversible with old byte hashes. The broad goal stays active. Next choices require validation-only selection and new confirmation. Device/controller/clock/nonideal generalization, broader models/full parameter conversion and fair measured systems costs remain open. Conditional finite-capacity weight residency/programming costs may explain a spatial-temporal regime; node-count reductions alone do not establish it.

[Report](../../results/thrml_spw/FINE_RANGE_CONFIRMATION_REPORT.md), [combined audit](../../results/thrml_spw/fine_range_confirmation/evidence_audit.json), [completion](../../results/thrml_spw/fine_range_confirmation/completion.json), [resource ledger](../../results/thrml_spw/fine_range_confirmation/resource_ledger.json).

## Conventional numerical controls audited; storage accounting correction (2026-10-27)

All 21 predeclared deterministic controls on the same consumed 32 contexts are complete: FP32; minmax, zero-preserving affine and symmetric INT2/4/8 at groups16/128; two decoders of existing SPW control codes. Independent original-model replay reproduces every token loss/prediction and all 504 weight hashes (891,813,888 reconstructed coordinates). All 440,000 resampling values across 22 diagnostic comparisons pass a second implementation. Zero new native samples, fresh held-out contexts or trained/selected candidates.

Zero-preserving INT4/group16 gives PPL132.298 versus SPWfine134.816 and FP32129.541, with nominal converted code/range/bias payload6.28125 versus12.03125bits/weight. The consumed-text SPW-minus-INT4 CE interval [.003298,.034558] favors this conventional control descriptively. All18 conventional variants are reported, not only that reference. The deterministic model decoded from SPW programmed mean weights gives PPL129.492, consistent with consequential finite-sampling/nonlinear effects. It is not E[stochastic network], an additive error decomposition or physical performance. Original SPW point-pass/strict-fail and inconclusive fine-versus-coarse comparison stay unchanged.

The supplementary prior mixed-model resource ledger had counted 41,472 affine biases twice. The retained original ledger/source and correction audit document a165,888-byte reduction per method: coarse202,902,528bytes, fine221,482,032bytes. Converted payloads, every inference result and sampling count remain unchanged. Logical payloads exclude dynamic binary state and runtime/device overhead. All controls use dequantized F32 kernels; no packed-integer or chip comparison is measured.

Concrete progress this turn: close conventional controls with independent audits, preserve full results and accounting correction, and identify finite-bias positional p-bit comparison as the next missing mechanism. Existing matched-resource native studies compare same-probability replicas; historical positional encoding is deterministic. A new finite-field positional baseline must be explicitly internal, not attributed to Extropic. Broader goal active; no hardware advantage or Extropic disproof.

[Control report](../../results/thrml_spw/FINE_RANGE_QUANTIZATION_CONTROLS_REPORT.md), [completion](../../results/thrml_spw/fine_range_quantization_controls/completion.json).

## Finite-bias positional spatial baseline independently replayed (2026-10-27)

This continuation closes the conventional numerical controls and executes a previously missing mechanism: positional binary words represented by actual finite-bias free THRML nodes, distinct from same-probability replicas or deterministic bits. The pre-sampling source-sealed protocol uses 257 scalar targets including endpoints, five seeds with200 trials each, field bounds4(primary)/2(secondary), K2/4/8 spatial nodes and T1/4/16/64; one-node8bit control fields include every matched K*T budget. No clamps or infinite fields. All raw per-bit prefix counts retained.

723,712,000 production and723,712,000 independently replayed native node updates complete with terminal exit0. A separately constructed public THRML program matches all33,924,000 saved counts exactly and verifies160 unique key batches. All44 empirical/analytical MSE rows pass independent moment checks. The frozen six-sigma scalar gate passes (maximum absolutez2.41034). Finite controller and positional quantization bias are included in theoretical error.

At primaryB4, four positional nodes T1 have scalarRMSE.022444 versus one nodeT4 .203517 at equalfour nodeupdates. At256 updates, four nodesT64 .019246 versus one nodeT256 .025687. Eight-node positional references also win in tested equal-update budgets. One node overtakes the coarse two-node quantization at32/128updates: T32 .071976 versus two-nodeT16 .096007, and T128 .036086 versus two-nodeT64 .095955. These examples do not imply uniform superiority or neural quality. All24 equal-update comparisons and every44condition are retained.

The programmable-probability encoding remains demonstrated. Its physical density benefit must be weighed against temporal sampling and controller precision. Positional uses K binary selectors plus shared +/-B levels, SPW8-bit control code plus shared mapping; dynamic states, range metadata and physical circuits remain separate. Counts assume parallel device updates; no measured hardware energy/latency/area. Internal positional encoding is not attributed to Extropic's architecture. No new neural text, training, candidate selection or quality-gate rescue occurred.

Concrete progress this turn: independently audit conventional controls and correct retained supplementary bias accounting; execute and independently replay finite-field positional comparison. The broad research goal stays active. Next meaningful work is a validation-only native neural positional/operator comparison and explicit matched-quality controller/device/sampling accounting, preserving prior consumed-test boundaries and failures. Physical-hardware transfer, full parameter coverage and a systems crossover remain open.

[Finite positional report](../../results/thrml_spw/FINITE_POSITIONAL_REPORT.md), [completion](../../results/thrml_spw/finite_positional/completion.json), [conventional controls](../../results/thrml_spw/FINE_RANGE_QUANTIZATION_CONTROLS_REPORT.md).

## Full-matrix finite positional operator comparison launched (2026-10-27)

Previous goal turn classification: progress. It completed conventional quantization/encoded-mean controls and independently replayed the finite-bias positional scalar comparison. Current turn prepares and starts the missing all24-matrix neural-operator comparison: identical group16 ranges/FP32 teacher activations, single-node8-bit fields versus four-node positional words at +/-4, five seeds, four matched-update budgets4/16/64/256. Total960 operator cases are fixed before sampling. All endpoints remain free. This is consumed validation, not held-out or full stochastic student quality.

Independent pinned-checkpoint/original-model hooks already pass: all24 inputs and F64 targets match exactly, and all42,467,328 source weights plus both control formats/ranges/biases are checked. Production is actually live; source seals and content-addressed copies are retained. The live checkpoint records actual PID, execsession75658 and the original log. It must be polled as-is: an observation timeout does not mean stopped. No frozen source may change while the writer runs. The global checksum index intentionally awaits all writers.

No completed comparative native result is claimed. After production terminates, run independent public sum-only and separate frequency replay for every case, then the report. The full stochastic-network follow-up was recorded prospectively before comparative operator summaries: singleT256 versus positionalT64, five seeds, actual student inputs, four8-context banks, matched256updates/weight/bank. It cannot start scoring until operator/control audits and persistent-capacity checks pass. Prior failures remain unchanged; no hardware or Extropic superiority claim. The broad research goal remains active.

[Live checkpoint](../../results/thrml_spw/positional_operators/LIVE_CHECKPOINT.json), [frozen protocol](../../results/thrml_spw/positional_operators/protocol.json), [independent controls audit](../../results/thrml_spw/positional_operators/control_audit.json), [prospective next neural stage](../../results/thrml_spw/positional_operators/NEXT_NEURAL_STUDY.md).

## Operator production verified live; automatic replay queued; causal student sources ready (2026-10-27)

Previous goal turn classification: progress (independent all-source/checkpoint/input audit and real960-case production launch). This turn verifies the same original production PID/session still live; it is not restarted for slow compilation/log intervals. Completed per-case outputs continue accumulating. The live checkpoint and automatic audit pipeline checkpoint carry authoritative process identities and counts.

A watcher now waits for this exact production PID/start-tick/command to become terminal and requires completed manifests before starting independent public output/count replay. The frozen scientific producer/auditor remain unchanged. A separately recorded audit launcher retains the original pure frequency factories to avoid redundant retracing and verifies all960 unique condition/stage/seed identities and output hashes before replay. The original sampling keys, factors, observers, F32/exact/IEEE gates and all outcomes remain unchanged. No passed native audit or comparative result is claimed while waiting. The initial watcher launch failed before starting because a GPU helper imported JAX in the model-only environment; original log/source/exit1 are preserved in execution_incidents/audit_launcher_environment. Minimal CPU-only file helpers corrected the launcher; the actual experiment was unaffected and no native samples repeated.

The next full stochastic student validation sources now exist and parse; CPU model/auditor/statistical modules and native worker/auditor modules import in their actual separate environments without construction or sampling. Conditions remain prospectively singleT256 versus positionalT64, five seeds, four banks of8 contexts, samegroup16 controls and consumed32 validation contexts. A documented pre-execution implementation refinement uses synchronous native projection replacements in a causal model forward, avoiding25 redundant prefix forwards. Every actual bank input hash/native output and final per-token metric is saved; intermediate prefix task-loss diagnostics are omitted from this comparison, with earlier sensitivity studies retained. New batching8 is matched for baseline/both conditions and independently audited. Scientific conditions/data/update budget and success scope are unchanged.

Preparation and scoring are still gated on completed operator/control audits and a real4.5GB free-space check (roughly3.45GB worst-case raw neural outputs plus headroom; new outputs use exact lossless packing). The future protocol has not been sealed and no student model/scoring/sampling has run. The planned complete original-model/public native auditor and separate60,000-value score/bootstrap verifier are prepared, not yet proven by executing the cohort. All newly edited source versions are content-addressed; global artifact indexing intentionally waits for all writers.

Concrete progress: maintained live experiment, queued automatic full replay, fixed/preserved launcher import failure without touching production, and implemented/import-checked the next causal full-network comparison and independent auditors. Broader goal active; no new held-out quality, systems advantage or Extropic disproof.

[Live checkpoint](../../results/thrml_spw/positional_operators/LIVE_CHECKPOINT.json), [automatic pipeline](../../results/thrml_spw/positional_operators/AUDIT_PIPELINE_CHECKPOINT.json), [next-stage development readiness](../../results/thrml_spw/positional_operators/NEXT_STUDENT_READINESS.json).

## Full research pipeline queued and finite-control information audited (2026-10-27)

Previous goal turn classification: progress. It kept the original operator run live, queued its independent replay and implemented/import-checked the causal student sources. This turn adds independently written verification of all480 operator ratios/four seed summaries, an explicit resource/report module for the student, and three passing fail-closed preflight tests. Tests cover failed/altered prerequisite evidence, changed or duplicate sampling conditions, source changes and inadequate storage. These are startup checks, not a passed neural inference claim.

The actual second watcher (PID3652428, exec97001) waits on the existing operator-audit watcher (PID3647421, exec61065), checks its terminal successful state, then runs the summary audit, preflight tests, immutable student preparation, all10 causal student cases, complete original-model/public-native audit, paired statistics, independent60,000-value audit and report. Every stage exits on failure; it never restarts original sampling. Student conditions remain prospectively singleT256 versus positionalT64, five seeds/four8-context banks of consumed validation. Source bytes are sealed and archived at queue time, before comparative summaries; no new candidate/test selection. Actual student protocol preparation/scoring has not started while the prerequisite run remains live. The original producer remains PID3633671/exec75658, unchanged.

In parallel, a substantive Phase24 follow-up calculates information extracted from finite probability controls and optimal Bayes risk for112 conditions. It covers uniform256 field codes and a matched prior over16 desired scalar values, two boundsB4/B2, temporal samples through4096, positionalK2/4/8. Counts are sufficient for IID Bernoulli trajectories; I(control;count)<=min(H(control),log2(T+1)). This distinguishes programmed-setting information from the one-bit instantaneous state. At matched16-level prior/B4/four total updates, one-node information is.866118bits versus3.982583 for four positional nodes. At4096 samples, uniform256 one-node controls reveal5.033839bits, not the full8-bit setting label. Information is not ENOB or neural accuracy; priors are analytic, not the observed neural-control distribution.

Independent mode-centered binomial recurrences, KL information and posterior squared deviations checked3,372,940 likelihood entries and every112 result/bound/monotonicity row. The first producer had ten nonfinite Bayes risks from zero-mass0/0 and its independent verifier failed. Its original source, raw data, failed log and exit1 are retained. A separately hashed derivative excludes zero-mass outcomes and reuses every likelihood byte, preserving every information value and all finite risks. The unchanged verifier then passes. Very small risks/printed numerical zeros have explicit F64/absolute-tolerance caveats; they are not proofs of exactly zero error. No new native samples or neural scores were used by this analytical follow-up, and no hardware claim follows.

Concrete progress: fully queued the next large neural experiment and audits behind real process/quality gates, tested startup rejection paths, and completed the independent finite-control information analysis with the original numerical failure preserved. The broad goal remains active. Global artifact indexing waits for all active writers. Earlier numerical/quality failures and hardware-access limits remain unchanged.

[Queued student pipeline](../../results/thrml_spw/positional_operators/STUDENT_PIPELINE_CHECKPOINT.json), [operator pipeline](../../results/thrml_spw/positional_operators/AUDIT_PIPELINE_CHECKPOINT.json), [information report](../../results/thrml_spw/INFORMATION_BUDGET_REPORT.md), [analytic completion/lineage](../../results/thrml_spw/information_budget_corrected/completion.json).

## Original960-case production terminal0; public native audit actually live (2026-10-27)

The original production handle75658 now returns terminal exit0. All960 outputs/statistics/manifests are complete, with144,388,915,200 scheduled native updates. Every334 recorded source version resolves to retained content-addressed bytes, and all scientific source seals remain unchanged. This is production/source completion, not a passed public numerical audit.

The queued watcher has started the actual public auditor PID3663796; its first completed direct-sum/count groups replay exactly. Full independent audit remains running. Watcher PID3647421/exec61065 and the full student continuation PID3652428/exec97001 are independently verified live. Neither original sampling nor the subsequent neural cohort is restarted. Student preparation/scoring remains unstarted behind this audit and the independent summary/capacity gates. The next goal turn must poll these exact live handles/processes and checkpoint files rather than launching duplicates. The broad goal remains active.

## Learned Boolean accumulation with explicit activation ranges independently verified (2026-10-27)

Previous goal turn classification: progress (960 operator outputs/source lineage completed, independent replay live and finite-control information analysis audited). The main frozen Transformer operator audit remains an actual live process; no original sampling is restarted. In parallel this turn fills an R3 arithmetic gap: a native Boolean observer evaluates the entire64x32 first layer of each of five existing positive-weight MNIST models on the first8 consumed validation inputs, at T1/4/16/64/256 and five MC seeds x200 trials. Weight probabilities remain directp=w with finite8-bit fields.

The first attempt assumed saved inputs were in[0,1], but the independent auditor correctly rejected their actual standardized range[-.42421296,2.673521]. Its producer had silently clipped those inputs; despite valid native counts and programmed-mean tests it does not establish the original learned operator. Original source/data/4,364,800,000 updates and failed independent-domain audit exit1 remain in boolean_layer/. This failure is not relabeled.

A separate freshly executed corrected cohort maps x=l+s*a with l=min(x),s=max(x)-l fixed before sampling. Its observer performs only Boolean AND and integer accumulation of output counts, weight column popcounts and diagnostic per-weight counts. Readout l*weight_popcount/T+s*AND_count/T+bias restores negative activations using two retained F64 range values(16bytes). All25 cases pass independent public raw-state replay:58,400,000 AND/weight/column count entries match exactly; every mean/variance/covariance/resource calculation is checked. All coordinate means pass the frozen family Bernstein gate. Main corrected production4,364,800,000 updates plus independent replay4,364,800,000; all Boolean research attempts total13,094,400,000. Key schedules were held fixed across the failed/corrected executions, so these are not independent-seed replications and are not pooled.

The new arithmetic works, with a measured cost: AND sampling MSE is about4.60times the same-draw deterministic-programmed-activation F64 control across the fixed budgets. The theory predicts extra variance s^2 sum_i p_w,i p_a,i(1-p_a,i)/T and positive shared-activation covariance across outputs even though device states are independent. Those covariance matrices are saved and independently checked. Hardware-style Boolean arithmetic does not automatically reduce noise or establish task accuracy. There are2048 weight nodes plus512 activation nodes for eight concurrent64-feature inputs(25% extra); changing activation controls, accumulators, range/readout corrections and physical gate/fanout/driver costs are separate. The simulator's sampler itself still uses floating bias arithmetic. No physical circuit, full classifier, ReLU/second layer or new task score is claimed.

An unused report helper had a syntax error when the corrected production was sealed. Its failed syntax/source bytes are retained; production itself terminated0. Afterward a syntax-only report repair and an explicit lineage-aware auditor resolve only that historical unexecuted helper from its archive. All scientific producer/verifier sources, native draws, exact count/statistical gates and tolerances remain unchanged. The actually executed auditor source and correction journal are hashed. A separate documented figure/title clarification changes no numerical values. The full audit is not represented as the original current-source guard passing unchanged.

Concrete progress: established a correct all-Boolean temporal accumulation path for an actual learned layer, caught/preserved invalid activation clipping, independently verified the repaired affine formulation and quantified its extra variance/correlated outputs. The separate Transformer neural comparison remains queued behind its full960-case public audit. Goal active; physical advantage, complete classifier Boolean quality and broader model conversion remain open. Global artifact indexing waits for all active writers.

[Boolean affine report](../../results/thrml_spw/BOOLEAN_AFFINE_LAYER_REPORT.md), [combined completion](../../results/thrml_spw/boolean_affine_layer/completion.json), [independent audit](../../results/thrml_spw/boolean_affine_layer/evidence_audit.json), [preserved invalid attempt](../../results/thrml_spw/boolean_layer/evidence_audit_failure.json).

## Full operator cohort audited; fixed causal student cohort running (2026-10-27)

All960 public F32 outputs replay byte-exact, every separate integer-frequency IEEE envelope passes, and all480 paired ratios/four summary rows pass an independent arithmetic implementation. The original producer and independent pipeline both have terminal exit0. Source lineage resolves334 recorded versions. Production144,388,915,200 plus independent replay288,777,830,400 =433,166,745,600 scheduled native updates. Primary256updates/weight contrast has geometric positional/single targetMSE.473957 [approximate five-seed sensitivity.469293,.478668]; four-node positional means are lower at every24 projection for allfour budget conditions. This is reused FP32-teacher validation activations, not end-to-end quality. Positional uses4times the weight nodes, one quarter sample rounds, and8.03125 nominal controller/range/bias bits per weight versus SPW12.03125; no physical system advantage is established.

The source-sealed automatic continuation passed the summary audit and allthree preflight tests, then prepared the unchanged prospective student protocolSHA937eb4354e3bdb4d3817c7dcfdd9aab8bee9d13529c571ae11fd480fa43ba8d0. The actual causal neural producer now runs at PID3675569 with native worker3676231, monitored by continuation PID3652428/exec97001. Conditions remain singleT256 versus positionalT64, five seeds, four8-context banks of consumed validation, all24 projections/51.84% unique parameters, input hashes and native outputs recorded before downstream nonlinearities. At this checkpoint 8/10 complete model records exist; raw partial scores are not a fully audited cohort result. No selection or threshold change follows from them. Source seals are still intact.

The next goal turn should poll exec97001 and the actual student parent/worker/PID from their live checkpoint. Do not duplicate production for an observation timeout. The queued chain already includes complete original-model/public-native path replay, score/bootstrap verification and reporting. Broader goal active; all earlier strict-quality failures and actual hardware limits remain unchanged. Global indexing waits for active writers.

[Closed operator report](../../results/thrml_spw/POSITIONAL_OPERATOR_REPORT.md), [operator completion](../../results/thrml_spw/positional_operators/completion.json), [actual student processes](../../results/thrml_spw/positional_student/LIVE_CHECKPOINT.json), [continuation stages](../../results/thrml_spw/positional_operators/STUDENT_PIPELINE_CHECKPOINT.json).

## Ten causal student cases complete; independent path audit live (2026-10-27)

The automatic continuation records student production terminal exit0, all10 full-forward model records and960 projection/bank calls, totaling434,865,438,720 scheduled native node updates. This is the unchanged prospective singleT256 versus positionalT64 comparison on consumed validation, all24 converted matrices/51.84% unique parameters. Production manifest explicitly remains independently_audited:false.

Independent original-model and public native audit processes are now actually live, recorded by module/PID in positional_student/LIVE_CHECKPOINT.json. Source seals still match the prepared protocol. The continuation already queues statistics, independent60,000-value replay and final report after path audit success. Raw partial/complete unverified scores are not promoted to an audited quality result. No original sampling is restarted; global indexing waits for active writers. Goal active.

## Matched-update full stochastic student comparison closed and audited (2026-10-27)

Continuation97001 is terminal exit0, with all staged commands successful. All10 full-forward model cases,960 actual bank inputs/public native outputs, count/IEEE paths, every baseline/final token loss/prediction and60,000 crossed resampling values pass independent checks. A further independent resource audit reconstructs the81,912,576 unique parameter count from pinned checkpoint config, checks every control/range/bias tensor, counts41,472 affine biases once and verifies all resource fields. Runtime source versions resolve to retained bytes; root_review.json links the final audits without altering the immutable pre-audit production manifest or earlier completion.

On consumed32 short validation contexts(2048targets), FP32 PPL134.587349/accuracy22.9492%; one-nodeT256 PPL144.850032(+7.6253%)/accuracy22.4023%(−.5469pp); four-node positionalT64 PPL139.958861(+3.9911%)/accuracy22.3047%(−.6445pp). Paired single-minus-positional CE.03435042nats with descriptive sensitivity[.01839004,.05431019]; accuracy difference+.0977pp, interval[−.4590,+.6641] crosses zero. Four-node log loss is lower here; an accuracy advantage is not established. Five MC seeds and four8-context bank clusters are consumed-validation sensitivity, not fresh generalization confirmation. All earlier strict-quality failures remain unchanged; no new pass/fail quality threshold is invented after scores.

Both use256 scheduled node updates per weight per bank. One-node uses one quarter the dynamic weight nodes, four times sample rounds and a larger nominal controller/range/bias payload(12.03125 versus8.03125bits/converted weight). Converted weights42,467,328 are51.8447% unique parameters; embeddings/head/norm/bias stay deterministic. Including the remaining FP32 parameters, nominal parameter-plus-packed-state payloads are226,790,448bytes single versus221,482,032bytes positional. Digital packed-bit descriptions are not physical noise-element areas or measured RAM; a device-count benefit alone is not total memory/energy/latency proof. Production434,865,438,720 plus independent replay869,730,877,440=1,304,596,316,160 scheduled native updates.

The probability representation demonstrably supports useful bounded neural computation, but no current matched-quality hardware systems advantage is established. This internal positional word is not an asserted official Extropic architecture. The next hardware cost ledger must distinguish scheduled evaluations/readouts from actual state changes and retained-bias energy. A separately planned full-classifier Boolean comparison and broader model/conversion coverage remain open. Goal active; final source/evidence snapshot follows only after all writers are confirmed terminal.

[Final student report](../../results/thrml_spw/POSITIONAL_STUDENT_REPORT.md), [root review](../../results/thrml_spw/positional_student/root_review.json), [independent resource audit](../../results/thrml_spw/positional_student/resource_audit.json), [full trace](../../results/thrml_spw/positional_operators/STUDENT_PIPELINE_CHECKPOINT.json).
