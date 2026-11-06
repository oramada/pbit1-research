# Research log

## Initial native-simulator milestone

Created the active long-horizon goal, pinned Extropic THRML, reproduced its official
example, and implemented direct w=p fields through public unary-factor APIs. Verified
scalar concentration and learned positive-weight inference on a full test set. The
preceding goal turn is concrete progress: native code, tests, data and manifests were
created, with quantitative results determining the next experiments.

## Finite control, dependence, and matched resources

Implemented actual native scalar field quantization and a validation-only controller
selection rule. Six-bit fields retained nearly the float32-control neural accuracy.
Added exact/native coupled-bank analysis, field calibration preserving requested
marginals, and actual native temporal-versus-spatial comparisons. Audited sampler
semantics and corrected an extra warmup charge in the resource counts: the first
record follows warmup, so warmup=1 plus T-1 updates gives T total sweeps.

Checked the original Extropic reference: its sampled data/latent variables and its
programmed model parameters have different roles. Recorded baseline definitions so
the direct weight-bank proof is not misreported as a verified replacement of an
official four-p-bits-per-learned-weight design.

Started native sampler-aware fine-tuning. Audit caught a branch that accumulated a
second-layer output computed from each individually activated first-layer sample.
That schedule differs from averaging before each layer's activation. It is excluded
from target evidence and has been replaced by two sequential native layer programs.
The same-path reconstruction check now passes and detects the invalid ordering.
This is an implementation audit, not a negative result about the intended representation.

The corrected two-epoch one-checkpoint training pilot was re-evaluated independently
on all 10,000 test images in batches of 32, using five sampling seeds. T=16 accuracy
was 76.85% for native fine-tuning versus 75.99% for its expected-weight control;
at T=64, 86.47% versus 86.34%. Matched UINT4 of that native-trained checkpoint was
89.99%. These are conditional sampling intervals for one trained model, not a
multi-training-seed result. Native low-T adaptation remains exploratory.

The objective is unchanged and active. Next work retains the whole scope: improve
the corrected execution schedule, characterize accumulator precision and adaptive
budgets, expand models, and obtain an implementable hardware/control baseline before
claiming device, energy, or latency advantages.

Final coupled-operator validation compared 108 dot-product conditions with exact
spatial and temporal covariance. There were 1,000 native trajectories per model;
the largest MSE discrepancy was 2.904 empirical standard errors. Exact variance
relative to the independent model ranged from 0.240 to 7.320 because input signs
can cancel or amplify covariance. Float32 field rounding changes exact marginal
probabilities by at most 1.01e-8 in these calibrated examples.

Six native/control tests and the corrected training schedule self-test passed.
The independent evidence audit additionally recomputes scalar errors, selected
control test accuracies, matched update/read counts, and full-test training
accuracies from saved raw arrays. These checks validate the bounded evidence;
they do not complete the broader research goal or establish physical advantages.

## Native efficiency continuation

The previous goal turn is classified as progress: it added native finite-control,
correlation/calibration, matched-resource and corrected-training evidence. This
continuation addresses the next open milestones rather than repeating those runs.

A frozen validation rule searched all 25 pairs from layer sample values
16/32/64/128/256 across five trained models and two sampling seeds, using all 5,000
validation examples. It selected (64,256). Independent test-key confirmation used
all 10,000 images, five models and five sampling seeds each. Accuracy was 88.4756%
versus 88.7416% for uniform256, a paired -0.2660 +/-0.0546 pp. Logical reads fell by
64.8649% and free-node updates by 63.7681%. Near-equal-read uniform89/90 reached
86.6000%/86.6664%. A separate per-model free-update bracket reached
86.7600%/86.7704%, so the allocation result does not depend on unused coarse-grid
budget or treating clamped nodes as stochastic updates.

The initial grid process was interrupted after identifying LRU-cache thrashing:
ten layer/budget functions exceeded NativeLinear's eight-entry cache and were
recompiled within each batch. Keeping local compiled-function references fixed the
performance issue. All reported results come from the completed corrected run;
the sampler and scientific schedule were unchanged.

Native fixed-point observers quantize each dot-product increment and saturate the
integer running sum. Two tests check exact integer results against explicit THRML
trajectories and independently trigger/count increment and sum saturation. A
validation-only format sweep selected int16_frac0. On five models by five sampling
seeds, full-test accuracy was 88.6168% versus paired float32 88.6268%, with zero
saturation. The int16_frac8 stress control saturated and reached 9.8940%.
More fractional bits consume sum headroom as T grows. A separate exact four-state
negative control showed that rounding can leave a bias floor without saturation:
the native mean converges to 0.1875 instead of 0.30 for the integer format.

Factorial confirmation combined independently validation-selected six-bit [-4,4]
fields, int16_frac0 sums and (64,256) samples without further selection. All 10,000
test images, five trained models and five sampling seeds yielded 88.3012%, paired
-0.4404 +/-0.0717 pp versus the original uniform256 route on the same keys. This is
a measured joint result rather than an assumption that separate changes compose.

Eight native/control/accumulator unit checks passed. A separate float64 raw-output
audit verified 250 allocation-validation conditions, 75 allocation-test conditions,
50 read-matched and 50 update-matched comparisons, 40 accumulator-validation and
75 accumulator-test conditions, 100 joint conditions and nine bias controls.
EFFICIENCY_REPORT.md and two standalone figure pairs record results and limitations.
Software fields remain float32 and accumulator containers int32; nominal circuit
widths are bookkeeping assumptions, not measured software or silicon compression.
The goal remains active for confidence stopping, broader training/generalization,
larger architectures and an implementable physical control/readout cost comparison.

## Native online stopping continuation

The prior turn is classified as progress: it produced native layer-allocation,
accumulator and joint finite-format evidence. This continuation implements actual
adaptive sampling rather than treating a stopped prediction from a complete
trajectory as evidence of saved simulator work.

An independent-case implementation fixes first-layer T=64, tracks output-layer
means/M2 at T=16/32/64/128/256, and removes stopped cases before subsequent THRML
calls. Every case has stable keys and every padding lane is counted. Complete-path
validation chooses a margin/conditional-SE rule without test labels. A fresh online
test pass recomputes the first layer and exactly matches replay outputs/stop times.
Five trained models and one sampling seed each yield 88.2420% for both adaptive
and fixed-output256 inference; mean total update fraction is 0.829548. Near-equal
fixed budgets reach 87.5780–87.5860%. These independent-case counts are not compared
directly to earlier bank-sharing counts.

A second implementation broadcasts each actual THRML weight-bank sample to up to
128 inputs. Validation runs each candidate rule online with compaction; it does
not borrow paths from a differently grouped reference. Gamma=3 is selected using
all 5,000 validation images and five models. Full-test confirmation uses five
trained models by five sampling seeds and all 10,000 images. Adaptive accuracy is
88.4084% versus fixed-output256 88.4304%, paired delta -0.0220 +/-0.1284 pp.
Mean output samples are 150.8548 and total free-update fraction is 0.830575, a 16.94%
reduction including the fixed first layer and partial-bank-batch overhead. The
upper matched-update fixed comparator reaches 87.6416%; adaptive gain is
0.7668 +/-0.0590 pp. CIs aggregate trained-model means after sampling-seed averaging.

Shared-case regrouping changes future random paths. Its actual online decision
means/M2, outputs, stop times, bank counts and padded product-work counts are stored
and independently audited. No-stopping tests exactly recover the fixed path. A
separate exact/native negative control with 10,000 trials programs p=0.6, compares
its mean with a deterministic rival of 0.5, and stops at 16 or 256. At gamma=3 the
returned expectation is 0.61389128, showing selection bias despite correct p-bit
programming. This control generates full trajectories for validation and claims
no online savings.

Twelve native/control/accumulator/adaptive unit tests pass. Independent raw audits
verify 50 independent-route validation conditions and 20 test conditions, plus
50 shared-route validation conditions and 100 test conditions. Shared model-mean
uncertainty aggregation is recomputed separately. ADAPTIVE_REPORT.md and standalone
figures record costs and limits. Confidence estimates exclude first-layer noise;
repeated looks provide no anytime coverage guarantee. The M2 array adds 320 nominal
bits per pending example, and moment updates add substantial arithmetic versus
ordinary fixed summation. Lower stochastic updates do not certify lower energy.

The goal stays active. Larger native models and a second dataset, broader training,
finite-format adaptive composition, and an appropriate Transformer/application
reference remain open, along with physical controller/readout characterization.

## Larger-network, second-dataset and pretrained projection continuation

The prior turn is classified as progress: actual online confidence stopping and
its independent/shared path audits changed authoritative state. This continuation
tests generalization instead of repeating that established experiment.

Conventional Fashion-MNIST training uses 196->64->10 networks, 13,184 matrix weights
and 74 deterministic biases, with 55,000/5,000/10,000 train/validation/test examples.
Normalization uses training features only. Five paired positive/signed runs share
the same positive initialization and minibatch order; projection is their only
training-protocol difference. Validation CE chooses checkpoints over 20 epochs.
FP32 means are 82.806% positive and 87.704% released signed control. This gap exists
before native inference and is not a p-bit estimator failure.

Actual THRML validation tests all 16 budget pairs from 16/64/256/1024 samples for
three routes. Direct p=w fails the predeclared one-point/0.05-CE validation gate;
its best-CE (1024,1024) is retained as diagnostic. Per-output affine routes pass,
selecting (64,1024) for positive and (256,1024) for signed weights. All routes are
then evaluated on the entire test set with five sampling seeds per trained model.
Test means are 81.5264%, 82.2996% and 86.7240%, respectively. Positive affine loss
is 0.5064 +/-0.0592 pp; signed affine loss is 0.9800 +/-0.1957 pp. Gates were not
redefined after the direct failure. Each affine route adds 592 bytes of range
metadata and external scaling. Its output-stage 1024 samples remain a latency
cost even when weighted total node evaluations fall substantially.

The per-output native range implementation has an explicit same-Gibbs-path check
against signed and constant-column matrix reconstruction. Constant columns use
p=0 clamps plus their metadata value. Metadata is per group, not silently free.
The raw metrics metadata_bits field describes fitted native-route metadata;
storage_ledger.csv separately charges actual per-representation metadata (zero
native range metadata for FP32/FP16). Its 120 rows are independently audited.

A real cached DistilGPT-2 checkpoint (revision 2290a62682d06624634c1f46a6ad5be0f47f38aa)
contains 81,912,576 unique parameters. Separate official WikiText-2 validation/test
parquets are downloaded at a recorded dataset revision. The old 487-byte invalid
ZIP is not used. Sixteen/thirty-two random nonoverlapping 65-token windows provide
1024/2048 next-token targets. Full-corpus tokenization emits a maximum-length
warning, but only the 65-token windows enter the model. This is a short-context
subset evaluation, not a published full-corpus perplexity comparison.

The predeclared first attention output projection contains 589,824 weights (0.72%
of the model). Frozen inputs and outputs are exported; FP32 output replay recovers
all per-token losses/predictions exactly. Actual THRML affine observers sample and
accumulate outputs at T=16/64/256, five seeds, with eight contexts sharing each bank.
Torch then replays those outputs before all downstream nonlinearities, checking
every cached input exactly. Only this projection is converted; every earlier input
is therefore still deterministic. Future multi-operator conversion must regenerate
later activations after earlier native substitutions.

The first downstream attempt terminated during calibration metadata serialization:
a NumPy Boolean was not JSON serializable. It was converted to a built-in bool and
the downstream evaluation rerun; native samples and exported inputs did not change.
This is an implementation correction, not a scientific negative result.

Calibration chooses T=64 under a 5% relative-perplexity/one-point token-accuracy
gate. Test subset perplexity is 115.3259 versus FP32 112.2068 (+2.7798%), with a
sampling-seed relative interval [1.0239,1.0317]. T=256 yields 112.5996 (+0.3501%).
Affine4 yields 112.7737; the zero-matrix/bias-retained control yields 564.9344,
showing that the operator matters. These are five Monte Carlo seeds of one fixed
pretrained model, not five trained models or a corpus-population confidence claim.

All thirteen native tests pass. Independent audits verify ten training checkpoints,
240 Fashion validation conditions, 300 full-test conditions, 120 storage rows,
30 native projection linear conditions, and 38 downstream language replay conditions.
New reports/figures preserve negative gates, precision/controller costs and scope.
The model environment's 57 package versions are recorded separately; the version
inventory does not certify portable wheel resolution. The goal remains active for
whole-block and larger-fraction Transformer conversion, broader training, realistic
finite controllers and physical-system validation. No density or energy win is
inferred from selective software quality alone.

For future reproduction, transformer_export now explicitly pins the discovered
dataset revision b08601e04326c79dfdd32d625aee71d232d685c3 as well as the model.
The original discovery-time source is preserved at its recorded SHA256 in
source_snapshots; input arrays and native/replay results were not regenerated or
changed for this metadata pin. Current sources are separately snapshotted for
subsequent long-horizon edits.


## Complete first-block native projection experiment and CUDA portability

Implemented sequential four-projection conversion with atomic operator checkpoints,
source/input/output signatures and exact prefix input checks. An initial worker
launch used Path.resolve() on the venv Python symlink, accidentally selecting the
base interpreter without JAX. Switching to absolute() preserved the venv; the
failed launch produced no native evidence and was rerun. A constant-column worker
test checks cache reuse and invalidation after changed input.

Discovered the local NVIDIA GB10 and created a separate CUDA JAX environment at the
same JAX/THRML versions. An initial CUDA package installation was interrupted to
avoid redundant multi-GB libraries after existing Torch CUDA libraries were verified.
Missing cuDNN discovery was resolved with explicit child-process library paths;
version checks remain enabled. A large operator backend comparison gives RMS
CPU/GPU difference 2.84e-6 of stochastic output RMS at equal logical counts.
All five complete T64 calibration pairs agree within 5.21e-6 cross-entropy. Only
then was the redundant CPU runner intentionally stopped; 12 completed CPU cases
and explicit partial status remain. The CUDA run completed its entire schedule.

Calibration T64 and T256 fail; T1024 passes. Test perplexity 114.9865 versus
112.2068 FP32 supports useful quality for 8.64% parameter conversion. The original
non-zero-preserving affine4 control performs very poorly, prompting six explicit
stronger quantized controls. Zero-preserving affine4 with 128-weight groups scores
112.8055. All controls are reported; no claim of INT4 failure or native superiority
is supported. Native field/range/bias storage subtotal equals FP32 before overhead.

Independent hook replay recovers token losses for all 25 completed cases. Evidence
audits check 100 native operator checkpoints and 12 quantized-control conditions.
Fifteen CPU native tests pass; five native/range checks also passed on CUDA.
A separate analytical forecast predicts lower conditional variance with local groups;
it is not a native quality result. Groups of 128 require 0.5 total range-metadata
bits/weight versus 0.0625 for current per-output ranges. Next work must run actual
native grouping and finite-control experiments, preserving these baselines. Goal active.


## Actual block128 native range experiment: fourfold smaller sample schedule

The prior turn made concrete progress: completed the first four-projection native
block and stronger quantized controls. This turn tested the analytical localization
hypothesis with actual official THRML sampling, not a variance-only forecast.
Added a grouped observer with explicit sample-by-span scaling and deterministic
input@lower correction before downstream nonlinearities. Group metadata and two
expanded lower/span software arrays are disclosed. Signed, constant, singleton and
ragged-group tests compare native outputs against reconstruction from identical
THRML states on CPU and CUDA. The initial test used groups of two, whose weights
were all min/max clamps, but incorrectly supplied a free-block state; correcting
the test to include interior probabilities and separate all-clamped singleton
coverage fixed the test fixture. No sampler behavior or upstream code changed.

The new driver/worker are distinct source files and output directories; historical
operators and their artifacts remain unchanged. Sampling/key/batch/calibration
protocols otherwise match the prior experiment. T64 fails (calibration +23.79%
perplexity), T256 passes (+2.69%). Test T256 perplexity 116.3544 (+3.6964%) versus
FP32 112.2068, five seeds on fixed short-context text. All 20 scheduled cases
completed with 80 operator checkpoints. Independent pre/post hooks reproduce all
20 final token-loss arrays. Seventeen native tests pass; both new grouped tests
also pass on CUDA. The generalized evidence auditor retains passing coverage of
the old per-output experiment and checks the new grouping independently from
raw weights and the declared group size.

Selected sample budget drops 1024 -> 256; free updates drop 75.34%, but quality
is matched to a tolerance rather than exactly equal. Local metadata rises eightfold
to 0.5 bits/weight and precision is redistributed into the ranges. Fields are still
float32, with additional expanded range arrays in this software implementation.
The conventional affine4 block128 control remains better quality. Reused corpus
windows make this a development result; a fresh test subset remains necessary.
Next work: actual finite-field controls for these operators, nonuniform budgets,
larger conversion fractions, and fresh text before confirmatory claims. Goal active.


## Finite controllers: 8-bit native block and unscored confirmation reservation

Prior turn classified as progress: native local ranges reduced selected T by four.
This turn added a finite-control grouped layer using the existing audited uniform
Ising-b code bank and explicit endpoint modes. Same-path native trajectory checks
pass on CPU and CUDA; all eighteen native checks pass. The separate worker guards
control bits/range in cache signatures and counts actual field arrays plus uint32
codes. Conditional error is explicitly decomposed into quantized-control bias and
Bernoulli variance/T; the latter alone would underpredict total error.

Predeclared B=4/6/8/10/12 ascending calibration at T256, bound4 and five MC seeds
rejects4 (+100.88%) and6 (+6.92%), selects8 (+2.63%). Test8 perplexity116.6273
versus112.2068 FP32 (+3.9396%), relative conditional interval[1.02813,1.05078].
The interval crosses5% slightly; only the point-mean calibration gate is claimed.
Test4 diagnostic scores266.5505. Mean-weight controls across all five precisions
are retained: test4=228.491,6=115.751,8=112.386,10=112.097,12=112.207. They are
conventional deterministic diagnostics, not native sampling claims. Finite8 adds
little systematic error; finite4 creates a persistent floor at this field range.

All25 native runs completed (100 operators); independent hooks exactly recover
all final token losses. The raw audit independently checks each actual field
lattice, conditional bias/sampling/total MSE, counts, source/key/input lineage,
selection and all ten expected-control downstream models. All passed. Packed
control/mode/range/bias ledger=10.4063bits/weight versus4.3125 for the conventional
block128 affine4 control, before native stochastic-state/program/accumulator costs.
Actual Python/JAX fields and code arrays remain32-bit; analog area is not inferred.

Prepared128 unscored65-token confirmation tiles with seed20261105, excluding prior
32 native test tiles. Additional audit discovered the older generic pilot used
arbitrary starts, reconstructed its documented four-window/seed17 run, reproduced
its baseline148.8574 versus recorded148.8573, and confirmed no positional overlap
with any reserved tile. The reserved tokens need not be regenerated. No inference
was run on confirmation text. The tokenization warning concerns the full stream;
all future model inputs are the saved65-token windows. Frozen confirmation plan:
FP32, conventional affine4 block128, floating T256 and selected8-bit T256, five
MC seeds, no selection using confirmation scores. Goal remains active.


## Frozen eight-bit confirmation on 128 previously unscored windows

Previous goal turn classified as progress: actual finite-control calibration and
unscored token reservation changed authoritative evidence. This turn executed the
reservation's frozen four-condition plan without retuning: FP32, zero-preserving
affine4 block128, native floating T256 and native selected8-bit T256. The protocol
hashes calibration selection and token reservation/audit before inference. It
inherits earlier calibration metadata for provenance; no confirmation search runs.
Five Monte Carlo seeds use common domain960730 and sixteen eight-context banks.
All later student inputs are regenerated and checked after earlier replacements.

Fresh FP32 perplexity140.845997 is higher than the earlier subset112.206767,
illustrating text composition effects. Conventional affine4 scores143.746256.
Native floating controls score145.506337 (+3.3088%); native8 scores145.553504
(+3.3423%). Both pass the frozen point-mean tolerance; native accuracy22.8882%
versus FP3223.0957% loses0.2075 points. No precision or budget was changed.

Predeclared uncertainty analysis resamples five seed indices and sixteen bank
clusters of eight windows in a crossed paired bootstrap (10000draws,seed20261106).
Finite8/FP32 interval[1.02219,1.04269]; conditional MC interval[1.02713,1.03976].
Finite8/float interval[.99794,1.00254] includes equality; quantization adds little
at this precision. Finite8/conventional4 interval[1.00136,1.02249] favors the
conventional control under this resampling model. These are sensitivity intervals,
not distribution-free corpus guarantees. All outcomes and draw values retained.

Independent audit checks10native runs/40operators, frozen data/selection hashes,
source/key/input lineage, reference math, conditional sampling/bias MSE, counts,
all raw token losses, independent hooks for all10cases, and both full downstream
conventional baselines. Another audit recovers every100,000 saved CE/accuracy
bootstrap draw values through a different aggregation order. Two analytic
uncertainty invariants pass. Official THRML source remains clean at pinned commit.
The optional token-archive argument added to the hook auditor leaves historical
export behavior unchanged; its prior source was already archived.

Each native seed evaluates28,538,044,416 free nodes across16 bank batches, not
physical stochastic transitions. Sampling depth1024 applies per eight-context
batch; serial bank reuse across all16batches takes16384logical rounds. Controller
precision/storage and external arithmetic remain fully counted; no physical win
is inferred. The confirmation text is now consumed. Next broaden conversion on
development data (three blocks about25.9% of model), consider grouped partial-sum
execution/nonidealities, then reserve another confirmation set. Goal stays active.

## Three-block conversion: narrow calibration pass, development-test miss

Previous goal turn classified as progress: first-block frozen confirmation passed
on new text, with independent raw and bootstrap verification. This turn converted
all twelve projections in blocks0/1/2:21,233,664 weights,25.9223% of the model.
The new explicit-target runner reuses the audited evaluation method; two tiny
model checks verify orientation, same-output replay, regenerated student inputs
following an upstream substitution, restored weights and target order. Historical
first-block modules were left unchanged. Current confirmation text is never loaded
by the development driver.

Predeclared T256/512/1024 search at8-bit bound4 fields/group128/five MC seeds:
T256 fails calibration (+11.6846% perplexity). T512 narrowly passes (+4.9588%),
so the frozen rule selects512 and does not execute1024. On the reused32-window
development test, native perplexity118.3382 versus112.2068 FP32 increases5.4644%,
missing the same5% tolerance. Conditional relative interval[1.04497,1.06441]
straddles the boundary; this is a point-mean failure, not a confidence claim that
true degradation exceeds5%. Accuracy23.6133% versus24.1211% loses0.5078points.
Conventional affine4 gives118.6774 and affine8 gives112.2975; the programmed-mean8
control gives112.6550. Neither native superiority nor quarter-model confirmation
is established. All failed/near-threshold outcomes are retained.

All15 native cases completed,180 operator checkpoints. Independent pre/post hooks
reproduce all final token losses; the raw audit checks frozen source/key/input
lineage, all twelve-stage conditional references, bias/variance MSE, logical
counts, calibration gate and raw scores. Eight independent downstream baseline
replays pass. At512 samples, the selected32-window schedule evaluates42,807,066,624
free nodes per seed,6144rounds per eight-context bank batch and24576under serial
bank reuse for four batches. Controller/mode/range/bias nominal subtotal remains
10.4063bits/weight, versus affine4:4.3125 and affine8:8.34375, before stochastic
state/program/accumulator costs. Actual simulator arrays remain32-bit.

Reserved another128 unscored65-token windows (8192targets), excluding the original
32native tiles, the consumed128-window confirmation and four arbitrary-start
legacy windows. Independent token/source/offset/selection audit passed. Before
seeing quarter development-test outcomes, froze the confirmation's T512, eight-bit
bound4 fields, group128 and five seeds, domain990730. Fixed comparisons are FP32,
affine4, affine8, programmed-mean8 and native8, with crossed bank/seed bootstrap
seed20261108. No confirmation model scores have been computed. The prepared
confirmation runner requires audited development evidence and refuses changed
protocols; it is syntax-checked but has not yet executed.

Separately implemented the expectation-aware group-partial-sum observer. It
moves scaling after temporal accumulation and groups the deterministic lower
correction. Two CPU/CUDA same-path checks include ragged/constant/singleton groups.
An actual2,359,296-weight block2 MLP-down native pilot uses the canonical stochastic
student inputs and identical keys/counts/T256. Scaled maximum output difference
3.813e-6, RMS difference1.493e-5 (9.21e-5 of original stochastic-error RMS).
Independent raw/MSE/ledger audit passed. Per bank batch it reduces range-scale
multiplications603,979,776 ->9,584,640 (63.02x), but accumulator399,360 ->9,584,640
float32 values (24x). Binary products and stochastic draws are unchanged. This is
an arithmetic tradeoff and one-operator equivalence pilot, not full-model quality
or measured energy/latency benefit. The canonical quarter-model study did not use it.

Inspection found the original shared compiled-function LRU has eight entries,
so twelve alternating operators evict one another. Preserved the entire canonical
run unchanged. A separate cached worker holds each operator/T compiled function;
a twelve-operator CPU/CUDA test proves old eviction, new retention, exact old/new
outputs, repeat-output identity and unchanged logical counts. It is selected only
for the upcoming confirmation implementation, with the same frozen sampler/math.
Its performance is not a physical-chip claim. The long-horizon goal remains active.


## Frozen quarter-model confirmation: reproducible point-mean miss

Previous turn classified as progress: three-block development completed and was
audited, with a narrow T512 calibration pass and a development-test miss. This
turn executed the previously frozen quarter confirmation with every parameter
unchanged:21,233,664 weights,12 projections,8-bit bound4 controls, group128, T512,
five MC seeds, domain990730 and128 reserved65-token windows. The new cached worker
retains twelve compiled operators; its CPU/CUDA exact-output/cache-identity checks
were already passed. No numerical sampler changes or confirmation retuning occurred.

Final confirmation FP32 perplexity133.203487; affine4:142.002196;
affine8:133.192987; programmed mean8:133.625435; native8:141.185572 (+5.9924%).
The fixed5% point-mean tolerance fails. Native accuracy23.05664% versus FP32
23.69385% loses0.63721points, within the one-point bound. Conditional five-seed
normal relative interval[1.05292,1.06697] is above5%; crossed paired seed/bank
bootstrap[1.04828,1.07003] straddles5%. Do not conflate fixed-text MC variation
with text/bank sensitivity or claim distribution-free corpus confidence. The
programmed-mean floor is0.3168%; finite sampling dominates the loss here.
Native/affine4 paired perplexity interval[.98140,1.00609] includes equality;
native accuracy is0.50293points lower than affine4. Conventional8 remains nearFP32.
No quality, information-storage, energy or physical-density advantage is established.

All five native cases completed:60 operator checkpoints, exact separate pre/post
hook replay of all cases, four independent downstream deterministic replays,
frozen selection/token/source hashes, causal input lineage, conditional references,
finite-control bias/variance/total MSE, counters and raw token losses verified.
A second statistical auditor recovers every140,000 saved CE/accuracy bootstrap
values, using a different aggregation order. A report-generation syntax check
caught an unmatched list terminator before report execution; it was fixed without
changing samples, raw data or the frozen protocol. The figure was visually checked.
Official THRML source remains clean at the pinned commit. Raw artifacts are retained.

Per seed:171,228,266,496 free-node evaluations,16 banks,6144 sampling rounds per
bank batch and98304rounds with serial reuse. Controller arrays still retain83,607,552
bytes each of float32 fields and uint32 codes, plus169,869,312 bytes expanded range
buffers, metadata, probability/program/state/activation/accumulation and unconverted
model costs. Logical p-bit counts are not total system storage or physical area.

After preserving the failed confirmation, prospectively specified a new study:
T1024/2048, the same three-block/control/range settings, five MC seeds and a larger
128-window validation set. Selection requires the point-mean constraints plus
crossed-bootstrap97.5%-CE <=log(1.05) and2.5%-accuracy difference>=-.01, using
validation only (10000resamples,seed20261111). This is an engineering sensitivity
criterion under exchangeable banks, not a formal guarantee. Validation/test domains
1000730/1010730 and fresh128-window test bootstrap seed20261112 are frozen.
Higher T doubles/quadruples the prior stochastic cost; this must not be hidden.

Reserved new128 validation windows and128 test windows,8192targets each, seeds
20261109/20261110. Excluded the original16 calibration tiles, all288 consumed native
test/confirmation tiles and four arbitrary-start legacy windows. Independent
source/token/offset/selection audit passed and records prior archive hashes. No
new-study model scores were computed. The tokenizer warning concerns the full
stream; every eventual model input is a saved65-token window. This is new research
motivated by the failure, not a rescue of the consumed confirmation. The long-horizon
goal remains active for implementation/execution of that design and the broader
nonideal/training/hardware questions; no completion or blocked status is warranted.


## Conservative larger-calibration study implemented and running

Previous goal turn classified as progress: the original T512 quarter confirmation
completed with a verified point-mean miss and the successor's disjoint data/rule
were prospectively frozen. Implemented quarter_conservative.py against that exact
design, with new128-window validation/test archives, unchanged12 native matrices,
8-bit bound4 controls/group128 and ascending T1024/2048. Three targeted statistical
checks pass: constant acceptable error, favorable mean with excessive seed noise,
and an accuracy error concentrated within one shared bank. Both mean and resampling
constraints must pass; a favorable seed is insufficient. Confirmation baselines
and native evaluations are reached only after the selection JSON is written.

The actual official THRML/CUDA driver is live under exec session15504, observed
PID2932828 with native worker3012778. It proceeds autonomously through five seeds
per calibration budget, selection and five selected-budget confirmation seeds.
A complete first T1024 calibration case scores perplexity135.9202 versus131.8593
FP32 on the same new validation text (+3.08%). It is a single preliminary seed,
not a calibration pass, confirmation or cross-text improvement claim. No selection
or final manifest exists yet; the test set remains unscored. Continue the same job,
not a new duplicate. /tmp/thrml-quarter-conservative.log records progress.

Added independent conservative selection/raw-stage/model audit and report scripts,
syntax-checked but not yet run because the study is incomplete. The hook auditor's
new --data argument supports split-key archives without changing historical
--tokens behavior. A LIVE_CHECKPOINT.json stores observed handles, source archives
and exact pending audit commands; it explicitly is not proof of continuing
liveness. Poll the actual session/process again. Current new source versions are
content-addressed under source_snapshots. The full checksum index remains the last
completed checkpoint and must be refreshed after the active writer finishes.
The research goal remains active, with no completion or blocked claim.


## Ongoing conservative run and packed-controller evidence

Previous turn classified as progress: implemented and started the prospectively
frozen larger-calibration run. Revalidated actual exec session15504 and driver/
worker PIDs2932828/3012778; did not restart. Three T1024 calibration cases have
completed (perplexities135.9202,134.8207,135.0881 versus131.8593 same-text FP32).
The fourth is running; no selection/final manifest exists and confirmation remains
unscored. All five seeds and the conservative criterion remain required.

While that unchanged job runs, implemented a separate pure-NumPy packed control
format and actual native loader. Codes are uint8, modes use two bits per logical
node, lower/span and neural bias remain float32. The artifact contains no original
converted matrices. Decoder reconstructs finite bias fields and official unary
THRML factors, preserving endpoints and free indices. Two actual native CPU/CUDA
roundtrip tests give exact fields/outputs, including ragged/constant/singleton
cases. No new sampler, numerical representation or control setting was substituted
into the ongoing conservative study.

Exported all twelve matrices; actual measured archive bytes are27,643,445;
refer to packed_quarter_controls/REPORT.json for authoritative numbers. Numeric
payload27620352 bytes (10.40625bits/weight), actual archive27643445 bytes
(10.4149505bits/weight), versus nominal FP32 matrix/bias85017600bytes. Every
control/code/mode/range was independently vector-checked. Deterministic mean replay
on the already consumed original16-window development subset exactly matches its
previous per-token losses and predictions; no ongoing-study text was touched.
This is compact control serialization, not1-bit information storage or achieved
whole-runtime compression. The conventional8-bit ledger remains smaller.

Current new sources are saved by hash in source_snapshots and recorded in the
packed SOURCE_ARCHIVE.json and active LIVE_CHECKPOINT.json. Do not refresh the
full artifact index while the native writer is mutating outputs. The remaining
calibration/selection/confirmation and independent raw/hook/selection audits must
continue from the live job. Goal remains active with no completion or blocked claim.


## Conservative calibration gate and real packed native operator

Previous turn classified as progress: packed-controller artifacts and three native
calibration cases changed authoritative state. Revalidated the existing exec
session15504 and actual processes; did not restart. All five T1024 calibration
seeds completed: mean CE4.90613885, perplexity135.116700 versus131.859303 FP32,
relative1.02470359. Bootstrap upper ratio1.03064928 and lower accuracy difference
-.00390625 pass the frozen conservative constraints. Selected1024;2048 is correctly
not executed after an early passing budget. Independent calibration arithmetic
recovers all20,000 saved CE/accuracy bootstrap values and the selected result.
This is not full raw/model verification, which remains scheduled after confirmation.

Only after selection was atomically written did the script score new test baselines:
FP32 perplexity129.867525, affine4:136.617284, affine8:129.822757 and mean8:130.290559.
Five native confirmation seeds at selected T are now running; no final aggregate
exists. The numerical plan is unchanged. Full independent audit/report source is
ready; pending commands and observed handles remain in LIVE_CHECKPOINT.json.

In parallel, packed_operator_pilot programmed the real2,359,296-weight block2 MLP
projection solely from finite codes/modes/shared ranges, without loading its target
FP32 matrix. Using existing old development student inputs and matching T512 keys,
CUDA THRML outputs exactly match the canonical tensor (max difference0). Each
path evaluates2,378,170,368 free nodes. A separate auditor independently decodes
packed modes via unpackbits, reconstructs Bernoulli variance and verifies source,
hash, input/reference/output, keys and counters without original target tensors.
All passed. This extends tiny CPU/CUDA roundtrips to a large native operator;
it is not full stochastic packed-model inference, runtime compression, physical
TSU execution or measured energy/latency advantage. It reads no new-study text.

New source versions are content-addressed. The closed packed experiments have an
independent artifact index; the global index must not scan a still-mutating native
writer. Continue confirmation on the same live job, then execute the pending hook,
raw-stage/selection/model audits, report and final global index update. Goal active.

## Large-operator matched spatial/temporal control

Actual THRML evaluated a 2,359,296-weight projection with explicit independent
replicas: K1/T512 against K4/T128, ten seeds per mode. Both execute 2,378,170,368
free-node evaluations per condition. Mean output MSE is 0.0138259 versus 0.0138636;
the spatial-minus-temporal approximate interval includes zero. Four replicas use
four times the nodes and one quarter the parallel sampling depth. Independent raw
output/count/variance audit passed all 20 conditions; tiny CPU/CUDA trajectory
checks passed. This is a replica control, not positional encoding, an official
Extropic learned-weight comparator, a formal equivalence test or physical energy
proof. See [matched operator report](../../results/thrml_spw/MATCHED_OPERATOR_REPORT.md).

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

## Matched-observation weight-only probe placement — completed, goal active

A nominal Fisher-information rule was fixed before new sampling and text scoring,
using only target controller codes and local ranges. All 128 symmetric pairs per
stage were evaluated; an independent inverse-Fisher calculation verified all 1,536
scores. Two finite native probes remain P4096, inference T1024, with five fresh
paired synthetic profiles and 64 new nonoverlapping text windows. All 240 probe
windows and 120 operators passed public-factor replay, and independent original-model
hooks regenerated every runtime input with exact NPY hashes and reproduced all ten
model cases. Baselines, 130 prefixes, 60,000 bootstrap values and resource/diagnostic
summaries passed separate audits. Inputs are regenerated rather than duplicated on
disk; raw unique stochastic observations are retained.

Actual weight MSE falls from 7.8400e-5 to 5.8612e-5 (25.24%), compared with a 31.16%
nominal first-order prediction. Offset estimates improve; gain estimates worsen.
PPL is 139.133 versus fixed 139.798 and fresh FP32 133.043. Paired CE difference
-0.004775 has crossed sensitivity interval [-0.012932,+0.003242]. Both strict gates
fail; the Fisher candidate meets only the point rule. This is not a confirmed neural
win, and the earlier failed measured-calibration regime stays unchanged.

Equal production budgets total 869,730,877,440 calibration and 1,739,461,754,880
inference observed node updates across both methods. A post-hoc analytic IID/reset
recorded-state-change proxy is 66.14% higher for calibration and 13.87% higher overall
with Fisher probes. No physical switching/energy measurement or advantage follows.
All 120 recorded manifests resolve to preserved source snapshots. A completed cache
check passed without new sampling or changed manifest. Source and raw evidence remain
pinned to official THRML commit 2d54cd8404df85bb65703d7e8b4e84e48c1b72a9.

[Report](../../results/thrml_spw/PROBE_PLACEMENT_REPORT.md).
Next: prospective validation-only practical estimator/budget allocation and separate
confirmation, broader conversion and measured/clocked/correlated transfer laws. Disk
space is about 3 GB; avoid new large output collections without a preservation plan.
No unique evidence was deleted. Hardware access/costs and full-model quality remain
open. The long-horizon goal is active, with no sampler pending at this checkpoint.

## All-six-block scale test and evidence preservation — completed, goal active

Previous goal turn: verified progress (matched-resource probe placement and audits).
This turn: verified progress (doubled converted matrix subset with all native paths,
new prospective text, paired quality limits, source/history audit and storage integrity).

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

Paired all24-minus-first12 CE difference +0.033324, crossed sensitivity
interval [+0.022771,+0.042742]. Five sampling
seeds cross only four whole eight-context banks: intervals are descriptive, not
coverage/equivalence guarantees. All 25 prefixes per seed are preserved; no subset,
sample budget or quality rule was selected from these new task outcomes.
The clean study has 869,730,877,440 production observed node updates;
independent public replay repeats the count. A stopped launch retains thirteen native
operators and 94,220,845,056 updates separately; no
quality result or unique observation was erased. First12 and all24 share prefix
outputs, so the paired comparison costs no duplicate prefix sampling, while all24
requires twice the nodes/updates of its prefix at the same T.

The initial worker failed before samples because a shared IO import required Torch
in the THRML-only environment. Its original source/execution/stderr are retained;
using existing NumPy-only atomic IO fixes the environment dependency. A later launch
was stopped on suspected validation overlap. Independent token-level auditing showed
a false alarm: numeric offsets belong to different validation/test parquet streams.
All archived token rows were verified against their actual split. Prior studies do
not have the supposed reuse; no contamination correction to their numerical results
is warranted. The replacement conservatively excludes every prior numeric offset
vector plus the now-consumed stopped-launch test positions. Both failed/stopped
launches remain at original paths under `results/thrml_spw/all_blocks/`.

Storage maintenance verified and hardlinked only exact duplicate deterministic
input/reference arrays in closed studies. All 287 names and original hashes remain;
261 paths now share 26 read-only canonical inodes, reclaiming 4,613,440,128 duplicate
bytes. Independent hashing/inode checks and fixture tests cover fail-before-mutation,
atomic independent replacement and restoring private copies. No unique stochastic
output was removed. Artifact-index byte totals remain logical path lengths, not
unique allocated disk bytes. Restore instructions and original permissions/timestamps
are preserved in `results/thrml_spw/storage_deduplication/`.

Remaining scope: validation-only broader-model budget allocation/adaptive inference,
probability-aware Transformer fine-tuning, nonideal/correlated/clocked broader conversion,
embedding/head and longer-context/multimodel tests, relevant official application
comparators, and actual device/control/readout/accumulation cost measurements. The
52% ideal finite-controller result cannot establish full-model or hardware superiority.
No task gate is relaxed and no previous failure is reclassified. Goal remains active.

Numerical audit addendum: original2e-5 float64-frequency criterion fails for one final-projection operator. Exact publicF32/count replay and a derived roundoff envelope verify the actual implementation; compensated plus hostF64-center diagnostic still fails its1e-6 target. All failures and attempts remain. Full task metrics/gates are unchanged. Independent witness reproduction is byte-identical; aggregate ledger includes every stopped/failed/replay diagnostic update.

## Equal-update validation-selected allocation — completed, goal active

Previous goal turn: verified progress (all-six-block ideal conversion, retained
quality/numerical failures, independent replay and reversible storage sharing).
Current turn: verified progress (new domain-qualified cohorts, 120 isolated validation
native operators, exact allocation/MILP proof, 240 held-out native operators and ten
complete test cases, independent task/path/bootstrap audits and conditional proxy bound).

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

Validation uses 32 unused validation-split windows and test uses 32 separate unused
test-split windows. All historical offsets are split-qualified and token-verified.
T256 single-layer native sensitivities on a programmed-mean model are averaged over
five seeds; the fixed floor is 1e-4. Exact DP minimizes the declared inverseT proxy
under budget 288; SciPy/HiGHS verifies the global optimum at zero gap. Selection
never reads test quality. The chosen schedule is [512, 1024, 1024, 1024, 1024, 512, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 256, 1024, 1024, 1024, 256, 1024, 1024, 1024, 512, 1024, 1024], weighted mean 1024,
with sumT 22528 versus 24576 and maxT 2048 versus 1024. The validation proxy predicts
only 1.79% smaller excess CE; leave-one-seed/bank diagnostics change 0–6 choices,
without replacing the frozen schedule or testing alternative schedules.

Held-out allocated-minus-uniform CE is +0.001092, sensitivity
interval [-0.002960,+0.004367], and
accuracy delta is +0.1074 points with interval spanning
zero. Both conditions fail the unchanged point and strict 5% PPL quality gate.
Four-bank/five-seed intervals are descriptive, not formal coverage/equivalence.
No consumed test result selects a new schedule or rescues the target.

All 360 production outputs and ten complete test model paths were independently
replayed, with exact F32 public outputs, exact inputNPY hashes, independent labels,
all 250 prefix rows and 60,000 bootstrap values. Separate output/frequency observers
and an IEEE error envelope were fixed before test; the 2e-5 frequency discrepancy
flag is descriptive and its one test failure remains. The first combined count/sum
auditor passed ten operators and failed exact output agreement on MLPup; its source,
logs and eleven replay calls remain. Separating the observers restored the same
bitwise acceptance condition. No particular compiler optimization is isolated.

Lossless byte-plane NPZ packing retains every F32 output bit. Special/random-bit
unit tests and four representative shapes pass, and all packed outputs match public
native execution and whole-model input hashes. Storage changes do not alter windows,
seeds, precision, arithmetic or model payload and are not a hardware advantage.
Native updates: validation 217,432,719,360; test
1,739,461,754,880; total including separate verification
and the failed observer attempt 5,885,178,937,344.
Controller/range/bias payload remains 45,287,424 bytes, 8.53125 bits/converted weight,
plus 48 schedule bytes. SumT/maxT are distinct, not measured latency or energy.

The post-hoc validation-only Cauchy bound proves the best continuous allocation of
this fixed noisy/floored proxy predicts at least 5.916%
PPL increase at current budget; independent SLSQP confirms its optimum. This is NOT
a neural-loss or representational/hardware impossibility theorem. No new samples or
replacement decisions occur. It points toward variance/range/training or additional
budget work rather than assuming another redistribution will close the tested gap.

Source lineage resolves all 360 native manifests and every recorded source version.
Both completed-driver cache checks pass without new samples or changed test manifest.
Broad scope remains: better range/precision/training regimes, nonideal/correlated/
clocked broader conversion, embedding/head, long contexts, additional models, fair
application comparisons and actual device/system costs. About 1 GB disk remains;
retain all unique evidence and make a storage plan before another large cohort.
Goal stays active; no experimental process pending at this checkpoint.

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

## Fixed compiler follow-up in progress (2026-10-27)

The previous range-localization goal turn made concrete progress: 360 raw operator cases, a verified ideal range theorem, explicit controller costs and a retained failed exact-output gate. This turn adds numerical diagnosis, not another task-quality claim. Repeated calls to one executable are exact; production/public integer counts match. Replaying the preceding 15 native calls leaves the cold-stage1 discrepancy unchanged. Two independent pilot processes with XLA autotuning disabled produce identical outputs and counts.

A frozen prospective check now runs all 24 projections and three conditions at seed0 in two independent processes, with public SumOnly replay and previous count-hash checks. Both processes are live; no final all-case result is claimed. Original 270/360 exact failures stay unchanged. See [live checkpoint](../../results/thrml_spw/range_localization/fixed_compilation/LIVE_CHECKPOINT.json).

The next end-to-end neural comparison is frozen separately: fine group16/T512 versus coarse group128/T1024, five seeds, 32 unused short test contexts, unchanged FP32 point/strict quality rules, full stochastic student activation propagation through all24 affine projections. The reservation independently reconstructs all 2,080 token IDs and reproduces exclusions and random selection; no model has been scored. This is still 51.84% parameter conversion, not whole-model weights. See [reservation audit](../../results/thrml_spw/fine_range_confirmation/reservation_audit.json). Actual execution requires the compiler cohort gate and at least10GB persistent free space; the resource request is pending. No success is inferred from a protocol or from elapsed waiting.

## Completed fixed-compiler check and confirmation implementation (2026-10-27)

The earlier in-progress numerical follow-up is now complete: all 72 cases (all 24 affine matrices, three control/sample configurations, seed0) match exactly across two independently started processes with XLA autotuning disabled. Public-factor/SumOnly replay also matches every output and count hashes match the previous audit. The original 360-case default-compiler gate remains failed; there is no retroactive quality or numerical criterion change. Maximum relative target-MSE difference from original seed0 outputs is about 1.99e-8. Historical compiled executables were not saved, so the original cause remains incompletely identified. This is scoped numerical reproducibility, not a full model or hardware result.

The prospective end-to-end fine-range confirmation now has CPU driver, separate native worker, independent original-model/public-factor auditors, paired-statistics audit and atomic result writes. Two isolated gate tests pass; the real preflight refuses execution for insufficient persistent storage before Torch/JAX model/runtime imports. No model has been scored on the reserved 32 windows, no execution-source seal or native confirmation manifest exists, and full integration remains unverified. Source snapshots and implementation_readiness.json state that limit explicitly.

The previous goal turn made concrete pilot/data-reservation progress. This goal turn adds completed broad compiler replication, actual source implementation and executed gate tests. It also identifies a possible resource remedy: a read-only probe of 66 historical single-link F32/F64 arrays restores every original byte, including headers, and estimates about 10.23 GB savings over 68.55 GB of eligible files (sample-max-ratio estimate 9.90 GB). These are extrapolations; no original file was changed. A durable lossless codec, independent disk restore verification, complete migration/restore manifest and explicit historical replay instructions are required before archiving anything. Capacity must actually satisfy the confirmation preflight before scoring.

[Compiler report](../../results/thrml_spw/RANGE_REPRODUCIBILITY_REPORT.md), [independent array audit](../../results/thrml_spw/range_localization/fixed_compilation/evidence_audit.json), [confirmation implementation](../../results/thrml_spw/fine_range_confirmation/README.md), [archival probe](../../results/thrml_spw/archival_feasibility/probe.json). Goal active; no live sampler remains after this checkpoint.

## Lossless storage recovery running (2026-10-27)

The previous goal turn completed all 72-case compiler replication and prepared the guarded neural confirmation. This turn implements a reversible exact-NPY-byte archive plus an independent byte-wise decoder and tests. Six tests pass, including endian/order/NaN bits, corrupt/truncated input refusal, interrupted journal and archive-sync failures, shared-link protection and exact restore/release. A real 102 MB historical F64 file was archived, independently read, durably restored with original hash/mode/mtime, then re-archived exactly.

The frozen plan covers 2,970 single-link completed-run arrays from the preserved pre-migration index. Every raw unlink follows verified persisted archive bytes and a durable restoration record. Historical original paths must be restored before older commands; [instructions](../../results/thrml_spw/lossless_archive/README.md) describe restoration and optional release after exact raw restoration. Source files, controls, tokens, shared links and new/current study data are excluded. All original scientific values and failure records remain unchanged.

Migration is live and capacity is growing. The all-file independent post-migration integrity audit has not yet run. No full neural scoring may begin until that audit and the actual capacity gate pass. The goal remains active; do not mistake per-file durability or an estimated saving for a finished dataset audit or task-quality result. The next CPU/native modules also import correctly in their separate environments without model construction or sampling.

## Storage recovery verified; neural confirmation can start (2026-10-27)

The previous goal turn made concrete progress by testing reversible archival and launching migration. This turn finishes it: 2033 completed-run arrays were losslessly archived, recovering 9,782,951,378 file bytes. An independent decoder verified all 2,970 planned original checksums, including 937 unchanged raw files, against the frozen pre-migration index (68,546,530,560 original bytes). Headers, floating bit patterns, modes and modification times are retained. No scientific value, quality rule or earlier failure changed. Restore archived paths before historical commands using the documented utility.

The real capacity condition now passes. The reserved 32-context fine-range confirmation remains unscored at this closure, with candidate/group/sample/quality rules unchanged. Its report generator has been prepared, but refuses to report before the actual model/native/statistical audits. Next execute the frozen source/sealed comparison, then independent full student-input, public-native, score and paired-bootstrap audits. The broad research goal remains active; storage success is not neural quality or physical hardware proof.

[Integrity audit](../../results/thrml_spw/lossless_archive/evidence_audit.json), [restore instructions](../../results/thrml_spw/lossless_archive/README.md), [next frozen study](../../results/thrml_spw/fine_range_confirmation/README.md).

## Frozen full student confirmation started (2026-10-27)

After the independent storage audit passed all 2,970 original checksums and actual free capacity exceeded 10 GB, the confirmation preflight passed without constructing a model. Source hashes then sealed before any new scoring. The five-seed/two-condition full student comparison is now actually running: group16/T512 versus group128/T1024 on the reserved 32 unused test contexts. First native operators complete successfully on current student activations. No completed cohort or audited task-quality result is claimed.

Poll exec session 96305 or the actual parent/native worker processes documented in the [live checkpoint](../../results/thrml_spw/fine_range_confirmation/LIVE_CHECKPOINT.json). Never restart for observation timeout; never edit frozen production sources mid-run. After production succeeds, execute complete original-model/public-native path and independent score/bootstrap/resource audits before reporting quality gates. The broad goal remains active.

## All inference cases complete; launcher recovery; audits live (2026-10-27)

All ten frozen model cases and 240 native operators completed. The launcher then failed its 60-second child-shutdown wait before summary creation; its source, terminal exit 1 and full original log are preserved. Both inference processes subsequently disappeared. A separate finalizer verified every completed run/metadata/output hash, recomputed summaries and sealed a recovery manifest with zero new model constructions or native samples. No production source or scientific gate changed.

Preliminary means: FP32 PPL 129.541; coarse 136.834 (+5.63%); fine 134.816 (+4.07%). Fine passes the fixed point 5%/1pp rule but fails the stricter uncertainty rule (PPL upper sensitivity +5.96%). Coarse fails both. The paired CE sensitivity interval includes zero, so there is no confirmed paired improvement at this stage. These remain preliminary until independent audits finish. A separate early original-model replay of coarse_seed0 matches all 24 input NPY hashes and all final token losses/predictions exactly.

The full original-model/public-native audit is now live at exec 96507; it has passed all completed operators so far. After it terminates successfully, run the independent 60,000-draw score/bootstrap/resource audit and the report generator. Do not rerun completed inference, change quality thresholds, or classify a timeout observation as a stopped audit. Teacher-forced short-context evaluation and 51.84% parameter conversion remain explicit limits; no generation/hardware superiority result is claimed. Goal active.

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


## Recorded-state activity follow-up closed (2026-10-27)

Official THRML witness covers all256 finite field codes, five independent seeds and1000 trials/code/budget, T1/4/16/64/256. All11,520,000 state/counter entries replay exactly through public raw-state sampling; all2560 frozen mean checks pass and exhaustive small-sequence tests pass. Production88,576,000 and independent replay88,576,000 scheduled updates are charged, including initializing draws. Overlapping reset/retained windows are explicitly dependent. Full24-matrix learned-control expectations independently checked in192 rows, not presented as measured full-neural trajectories. At256 equal updates/weight, expected reset changes are86.5693 for one node versus2.16913 across four positional nodes; retained values86.4068 versus.171642. This is a discrete recorded-state proxy; hidden events, holding power, physical rates, control/readout/programming and correlation remain unmeasured. No joule/latency/area assertion or new task score. Earlier quality results and failures unchanged.

The preceding SITREP turn was status-only (no progress). This continuation closes audited evidence and starts a new lossless archival delta with unchanged historical codec and frozen original plan; preexisting audits/records must remain byte-identical. Full Boolean classifier remains next after actual capacity verification. Goal active.


## Full-classifier validation-only preparation and native counter witness (2026-10-27)

Prepared all five fixed64->32->10 positive models using5000 validation inputs only, common affine input map and per-model hidden range[0,1.25*validation max]. No new test task scores. The paired sample schedules[98,392,1568] and[74,296,1184] match2368 versus3136 active updates exactly, with planned304,584,000,000 native updates/regime before complete independent replay. Original storage preparation understated aggregate counter width:64*1184=75776 exceeds uint16. Original artifact/source bytes remain retained with CORRECTION.md; corrected preparation_v2 uses uint32 AND/column counts and uint16 individual weight counts. Uncompressed required payload1,459,500,000bytes plus250,000,000 reserve means at least1,709,500,000 actual free bytes before production, with no assumed compression gain. Four preparation unit tests pass; separate range/encoding/budget audit checks all11,840 learned weight-control entries.

New generic native count observers cover both planned layer shapes and both execution regimes. A separate public raw-state sampler reproduces6068 count entries and all affine F64 readouts exactly/within declared floating tolerance, including a high-probability T1184 case that actually exceeds65535. Native observer6,139,136 updates plus independent replay6,139,136 charged. This bounded synthetic witness is not full two-layer causal classifier quality; full producer, causal replay, statistics and frozen production source manifest remain required.

Lossless archival delta is live under the existing exclusive operation lock, unchanged frozen codec and historical2970-file plan. The new versioned delta preserves all previous records/audits and independently decodes each newly persisted archive. No scientific data are discarded or resampled. Full classifier production remains unstarted pending capacity and source gates. Global evidence index is intentionally stale while archival writes are live. Goal active.


## Full learned Boolean classifier source-sealed and production live (2026-10-27)

Actual GPU two-layer preflight on128 validation inputs at largest planned samples passes exact count/public-state/causal output replay for both execution regimes. Charges118,816,768 observer plus118,816,768 independent public updates; these are separate from full cohort. Seven meaningful presampling checks pass, including equal full-forward active budgets, packed-time AND padding and actual worst-case75776 aggregate counts. All362 current research source versions archived by SHA before test scoring; official pinned THRML tree is clean. Critical producer/replay/statistics/finalizer/pipeline source versions are frozen. No sources should change during this cohort.

Full protocol now seals300 cases:5 positive models x10MC seeds x3budgets x2regimes, all10000 already-consumed test images. R1[98,392,1568] versusR3[74,296,1184] match total2368 versus3136 active nodes. Full planned production609,168,000,000 updates and complete independent public replay another609,168,000,000. Raw individual weight counts uint16; AND/column aggregates uint32. Every original count, derived hidden activation code and final logit is retained/reconstructible. Validation-only negative-input affine ranges and25% hidden margin are unchanged. No posthoc confirmatory quality threshold is introduced. Five model-level units remain five despite ten MC seeds; all budgets/controls/contrasts reported.

Production pipeline is now actually live, with real child PIDs in boolean_classifier/pipeline.json. Independent public raw-state causal audit, all predetermined statistics, independent bootstrap/baseline audit and final report follow sequentially only after terminal success of their predecessors. No partial task score is promoted. The earlier lossless delta finished463 additional arrays/661,773,578bytes saved and preserved closed audits/records; a second delta handles remaining frozen-plan candidates to increase actual capacity. At production admission actual free space1,871,429,632 exceeded1,709,500,000-byte minimum. External storage changes are watched per case; completed case bytes survive any later explicit capacity stop. Global evidence index remains stale while writers run. Goal active.


Lossless capacity delta is terminal exit0 after474additional arrays/271,051,232bytes saved. Combined original2033 +first463 +second474 independent proofs cover all2970 candidate IDs without overlap,68,546,530,560original bytes and10,715,776,188totalbytes saved. Every preceding record and historical audit remains checksum-identical; restore records include all candidates. The desired2.1GB free target was not reached after eligibility exhausted, but actual free capacity at full classifier admission exceeded its separate1.7095GB requirement. No claim that free capacity equals bytes recovered; external disk activity and new research writes also change it. Classifier production is live and recording completed cases; no task summary is independently audited yet.


## Live classifier receipts and scope reconciliation (2026-10-27)

Previous turn was progress: the complete sealed300-case native cohort was launched. This continuation verifies the same parent/child processes remain live, with monotonically increasing terminal case receipts; it does not restart sampling. Source seals are unchanged. An early read-only recorded-count check reconstructs every10000-image causal output in18 committed cases exactly, validates counter bounds and captures exact producer-manifest bytes before inspection. This is neither public-native replay nor a quality audit; full audit remains queued. The first14-case check recorded a reread manifest hash while the producer could append rows; retained v2 checker/snapshot fixes that reference without altering any scientific evidence.

Root/research README now identify the live experiment rather than stating all current experiments are closed. REQUIREMENTS_EVIDENCE.md maps the complete master design/current goal to existing native evidence, pilot-only coverage and explicit remaining scope, including CNN/model-family breadth and actual physical characterization. Every cited local target exists; this is not a completion certificate or a relaxation of earlier failed gates.

Actual free storage transiently fell to~545MB while current classifier occupied only~79MB, then returned above1.7GB without intervention. This indicates concurrent filesystem activity rather than current-cohort output volume alone. No unrelated processes, caches or datasets were changed. Producer retains its per-case capacity assertion and committed-case durability. Continue observing the same live pipeline and preserve frozen source versions. No global evidence-index refresh while writers are live. Goal active.


## Validation-only analytical companion during unchanged live classifier run (2026-10-27)

The preceding turn was progress and a verified wait: live pipeline persisted; recorded-count checks and full-scope evidence map were completed. This continuation confirms the same production processes continue advancing. Frozen scientific source files are unchanged. Independent work now derives conditional linear R1/R3 moments on all5000 validation inputs for all five models at the exact three paired budgets, no labels, no new sampler draws and no change to production. Layer1 analysis explicitly conditions on deterministic FP32-weight/F64 validation hidden states, not random production hidden activations.

For fixed x_p=low+span*q_a, R3 numerator is sum x_p^2*q_w*(1-q_w)+span^2*q_a*(1-q_a)*q_w. Relative to R1 with deterministic x_p, R3 variance is at leastT1/T3=49/37 under the sealed equal-active-update design. Actual requested-input conditional variance ratios across models are5.58–5.79(first layer) and10.95–12.34(output layer at fixed hidden). Input/control bias squares are separately recorded. These are exact conditional linear moments, not measured full-network variance, accuracy prediction or device cost. No new quality gate or budget is selected. A separate per-feature E[Y_i^2]-E[Y_i]^2 verifier checks6.3million coordinate components/30summary rows over1.05million distinct linear coordinates. Reproducible scripts and raw per-example reductions are retained outside the source-frozen production module tree; current critical seals remain unchanged. Full classifier production/audits are still required and goal active.


## Live cohort and independent accumulator-state accounting (2026-10-27)

The previous turn was progress plus a verified wait, closing the validation-only conditional moment derivation while the same production process continued. This continuation confirms source-frozen classifier sampling remains live and advances beyond the first two starting models. A deterministic register/operation ledger now fills an actual hardware-accounting gap: R1 tracks2368 per-weight counters versus378 implemented R3 AND/column counters. At largest paired samplesT1568/T1184, hypothetical minimum unsigned register totals are26048bits versus6336bits, while R3 adds768 activation nodes/nominal6144command bits and additional activation sampling variance. Current native counters are int32 and retained raw counts are uint16/uint32; these are distinct descriptions and not physical area. Actual output-layer diagnostic columns count even though hidden-range lower offset is zero. Peak-layer and all-resident node/register totals remain separate.

An independent verifier instantiates the actual native bank/observer objects, validates12 carry shapes/dtypes, all-one count bounds/minimal widths, all six paired resource rows and all five code/bias payloads. No new native samples or task scores. Arithmetic AND terms, column contributions, count reconstruction and dense MAC terms are listed separately; no unsupported energy equivalence or total memory ratio is inferred. The first standalone verifier launch failed before study import because /tmp omitted workspace PYTHONPATH; the corrected explicit workspace path passed with unchanged source. Incident/source evidence is retained. Frozen production sources and original quality rules remain unchanged. Full classifier native/statistics audits and broader hardware/model requirements remain open; goal active.


## Confirmed-live wait and committed classifier coverage (2026-10-27)

Previous turn was progress(register/operation ledger) plus verified wait on session41358. This continuation again verifies the same actual parent/child PIDs/start ticks and polls that original session rather than restarting. Production passes100/300cases and advances monotonically. The unchanged captured-manifest count-path checker reconstructs all outputs of87 committed cases exactly without labels or new samples. A separate receipt checker validates the first98case IDs/order/unique paths/every hash/exact scheduled-update counters against the sealed300-case list. Captured manifest bytes and checker sources are retained. These are early saved-record checks, not the complete public-native causal audit or task-quality result.

At98committed cases, nominal remaining uncompressed payload800,940,000bytes and actual free1,062,105,088bytes are separately recorded. This is diagnostic accounting, not a new production capacity rule, compression assumption or physical memory measure. No source/budget/range changes, scoring, restart or unrelated cleanup. Complete independent public replay, statistics audit and report remain queued in the same live pipeline. Goal active; no global evidence-index refresh while writers run.


## All reconstruction cases complete; actual Boolean execution live (2026-10-27)

Previous continuation made receipt/recorded-count verification progress and polled the confirmed-live session41358. This turn is a verified wait on the same actual producer/parent start ticks, advancing without any sampling restart through150R1 cases(all five models, ten MC seeds, three budgets). R1 production charges exactly304,584,000,000 scheduled node updates. Producer automatically enters R3 within the same sealed300-case protocol and process. Source checksums remain unchanged. All full task-quality summaries remain pending complete production and the queued independent public-native causal/statistical audits.

The unchanged early recorded-count checker now captures155committed cases and exactly reconstructs all10000-image outputs, including first five R3 cases with verified derived hidden activation codes/count bounds. This remains a saved-count path check, not full raw-state replay or task-quality confirmation. No budgets/ranges/gates/sources changed and no labels scored here. Live checkpoint records R1/R3 receipts and current capacity; no global index refresh while writers remain live. Full broader objective remains active.


## Boolean output covariance and conditional top-two margins (2026-10-27)

Previous continuation was a verified wait with all150R1 cases completed and first R3 actual saved-count paths checked. This turn continues the same confirmed-live process. New validation-only companion avoids overinterpreting the earlier individual-logit variance result: shared sampled activations induce positive cross-output covariance for these positive-weight models. With deterministic validation hidden inputs, Var(Yj-Yk)=Var(Yj)+Var(Yk)-2Cov. All5000 validation examples/five models use top-two deterministic stored-weight/F64 logits without labels or sampled outcome selection.

A distinct eight-outcome per-feature probability enumeration checks25000 chosen margins,125000 scalar components and15 summaries, including exact activation command agreement with the frozen production source. Ignoring covariance overstates selected-pair margin variance by31.3–34.5% relative to correct variance; covariance cancels23.8–25.7% of the sum. Equal-active-update R3/R1 conditional margin variance remains7.38–8.22. These are nominal Bernoulli moments from rounded fields, not measured physical/sampler probabilities, full stochastic-network moments or accuracy forecasts. No new samples, labels, range/budget/stopping-rule change or frozen source edits. R1 adaptive evidence remains separate; covariance positivity is not generalized to signed weights/devices. Full cohort production/native/statistical audits still required; goal active.


## Complete classifier production; full native audit live; static-cache reuse control sealed (2026-10-27)

This continuation advances the same source-frozen process to all300 production cases, terminal exit0 at2026-10-27T19:51:49UTC. All150R1 and150R3 cases are retained with609,168,000,000 scheduled native updates. No quality summary is promoted yet. Same original parent pipeline automatically launches complete public-native causal audit; wrapper3764534/native3764535 are actually live, with original producer terminal. Early saved-count reconstruction now checks all300cases/three million image outputs and derived second-layer codes exactly; this does not substitute for raw-state replay or statistical audit.

A separately sealed derived static-weight-cache control addresses an important amortization assumption. Bank0 is fixed from all150R1 cases before examining derived quality, no favorable bank/seed selection. Selected counts and first8example cached causal prefixes(12000logit values) match parent evidence exactly. A future deployment could reuse these reconstructed matrices for all10000 inputs, reducing sampling-bank count1250fold but requiring18944bytes of extra F64 reconstructed weights per model and conventional neural arithmetic. The selected243,667,200 updates are already within parent304,584,000,000R1 updates; no executed cost is erased and zero new sampler draws occur. Expected dataset-average quality matches by linearity under identical marginal weight law; finite-run dataset variance differs because perturbations are shared. Followup sources/statistics are sealed, with reused parent bootstrap indices and all budgets/models/seeds. Actual waiting helper3760254(session17675) gates evaluation on parent native/statistical audits and final artifact hashes. Original cohort/gates remain unchanged.

Storage headroom recovered by targeted disposable installer-cache eviction: four cached wheel/HTTP entries for the project's installed torch2.14.1, nvidia-cudnn-cu139.24.0.43, jaxlib0.12.0.1 and jax-cuda13-pjrt0.12.0.1. Every cached metadata/largest binary payload matched the installed bytes before deletion; installed metadata/major library hashes remain identical afterward. Exact inventory/body/header checksums retained. Cache bytes removed1,314,686,854; actual free rises from258,719,744 to1,573,416,960 at cleanup closure. No installed environment, research source/model/evidence file changed; cache packages can be downloaded again. Scientific evidence lossless recovery proofs remain intact. This is routine installer-cache cleanup, not deletion of research data or a change in runtime versions. Goal active; global evidence index waits for parent/followup writers to finish.


## Direct integer-activation gating operator passes bounded native replay (2026-10-27)

Previous continuation was progress: all300 classifier production cases closed, full native audit actually launched, static-cache reuse control sealed/queued and redundant installer caches safely reclaimed. This continuation confirms the same actual native audit/waiting-helper handles remain live and source seals unchanged. No scoring or original sources/ranges/budgets changed.

An aligned next execution design uses deterministic nearest8-bit affine activation values with sampled positive weights. The native observer keeps only integer output gated sums and weight-column popcounts: G=sum_t,i a_i*Z_i, C=sum_t,i Z_i, followed by external F64 readout lower*C/T+(span/255)*G/T+bias. This preserves negative standardized inputs and needs no activation p-bits or per-weight operational counters. The sampler still uses floating probability/control arithmetic; only the neural temporal observer avoids floating multiplies. Weight fields remain the same finite8-bit commands, all weight nodes free.

Bounded official-native witness covers model0/16 validation inputs, both64x32/32x10 shapes,2 sampling chains andT1/7/98/392/1568. A separate public raw-state sampler reproduces all3780 gate/column counters exactly and verifies the integer identity/readout against public per-weight-count reference. Native observer9,784,576 +independent public9,784,576 updates charged. This is not ten independent trained models or a full two-layer stochastic-network result: second-layer inputs are fixed deterministic hidden values. The largest bound25,589,760 needs25 unsigned bits; native int32 carries pass without overflow. AtT1568 a hypothetical minimum output-counter total9360bits versus26048 per-weight counter bits is a logical register comparison, not physical memory/energy. Wider integer gates/activation quantization/readout remain costs. No hardware bit-serial timing claimed. Sources, counts, environment/pinned vendor commit and completion hashes retained in integer_activation_native/witness.

A full same-trajectory native integer-gate classifier cohort remains required, including all five models/ten MC seeds/full consumed test set, fixed validation ranges, three budgets, deterministic quantized-activation controls and complete causal verification against the already generated parent R1 trajectories. The parent sampler weight fields do not depend on inputs, so those audited counts provide an exact integer identity reference; a bounded witness or a derived cached computation must not substitute for actually executing the full native output-counter observer. No new full-network quality claim. Goal active; global index remains stale while original parent/cached followup writers are live.


## Full native integer-activation classifier prepared and queued (2026-10-27)

Previous turn was progress: bounded integer-gate native/public witness closed while original300-case native audit remained live. This continuation verifies that original sampler/audit handles advance unchanged. A GPU preflight now executes both actual causal layers on128 previously consumed test inputs(no labels/quality), all three T98/392/1568 budgets, model0/seed0/first16banks. All18144 integer counters and final causal outputs match same-key already public-replayed parentR1 count references exactly;77,973,504 native observer updates are charged, with zero new public raw-state draws. Preflight shares trajectories with later production and is not an independent replication.

The full150-case integer8 cohort is source sealed before quality scoring:5fixed positive models x10MC seeds x3budgets on all10000 consumed test images, bank8/chunk16,2368 free weight p-bits/no activation p-bits, original finite8-bit field commands and validation-only ranges unchanged. Actual direct output-counter observer will execute304,584,000,000 updates. Complete integer identity and causal audit uses parent per-weight counts already independently public-replayed on the same keys/fields/initialization/schedule. Official pinned block_sampling.py shows observer memory flows separately from chain/sampler state; activation codes are passive observer inputs and fixed within each layer interval, so the reference is pointwise exact. This is not a substitute for executing the new observer: native production is still required and currently unstarted. No additional full raw-state sampling is claimed for that reference.

All code/source versions, source-case hashes, counter bounds(<signedint32), exact150-case updates and701,500,000-byte minimum capacity(451.5MB raw payload+250MBreserve) pass pre-scoring checks. CPU reporting imports pass. Controls cover ACT8 with FP32/INT2/INT4/INT8/programmed-mean weights. All paired R1/R3/FP32 contrasts and inherited5000 five-model/ten-seed bootstrap draws are fixed; no new confirmatory gate or fresh-generalization claim. Different activation coding/rounds in the R3 comparison are explicit, not attributed solely to one noise mechanism. A new actual waiting pipeline3778471/session43259 gates production on original full native/statistical completion; original cached-reuse helper remains separate. No original core source changed.

A report-only register-width correction is retained with old report/completion archives: the earlier9360bits atT1568 is a uniform-width-per-layer packing, not the true minimum for implemented counters. Separate gated/column widths yield9024bits(256*25+32*17+80*24+10*16). Column bounds are ni*T rather than255*ni*T. All actual native int32 counters/source/keys/data and old witnessed update counts are unchanged; current report/completion hashes verify. This correction introduces no scientific gate change. Goal active; native CNN/broader pretrained/device studies remain outside this cohort, and the global evidence index waits for live writers.


## Current official Torx stack reproduced and stationary semantics audited (2026-10-27)

Previous continuation was progress: full integer8 cohort prepared/source-sealed/queued, with actual GPU paired preflight and register-label correction; original native audit remained live. This continuation confirms the same original audit process advances unchanged through all150R1 and into R3, with no restart or neural quality promotion. Both queued followups remain live and gated on parent audits.

A fresh primary-source review finds the2026 Torx framework alongside THRML. Official repo cloned read-only at2142045276f24849a5ef2587a729ed961de8cf18(version0.0.3). Its required ihoop0.1.4 and actual local-source Torx package are installed only into torx_probe/deps, without dependencies or changes to existing venvs. Initial source-only import failed on missing distribution metadata and was resolved by genuine isolated package installation, without source patch/version shim. Base .venv-thrml still cannot import torx/ihoop absent the explicit extra path. Runtime JAX0.12.0.1/equinox0.13.8/numpy2.5.4 and source checksums are retained. Pinned THRML source/runtime remains unchanged.

Official Torx PSWAP quickstart(seed0/20000samples) gives swap.30135/stay.69865 for nominal.3, exact-density agreement and two-sided binomial p.67696 under a frozen.001 gate. This uses two logical sites/one circuit application and is not a stationary one-weight experiment.

New one-logical-site semantics study covers5targets(.1,.25,.5,.75,.9),3modes(flip-only,ideal reset+flip,finite reset+flip),2initial states,T1/16/256,5seeds x1000trials per condition. All90conditions/180fixed mean/second-moment bounds pass. StateVectorSimulator density kernels and BranchingSimulator paths agree with the nominal Markov model. Separate public-path loop reproduces all40,950,000 bit outputs exactly, and full count-distribution recursion verifies the producer's conditional raw-moment recursion. Each production/replay uses40,950,000 logical circuit cycles/68,250,000 logical gate applications; counts are not physical switching energy or independent extra replications.

PNOT's parameter controls flip probability; its repeated occupancy is1/2 for these interior controls. Ideal reset+flip prepares the desired probability at whole-circuit boundaries, using one logical site and an explicit infinite-logit reset abstraction. Finite reset(logit8,not8-bit quantization) shifts stationary occupancy to f/[r+2f(1-r)]. Intermediate reset-phase occupancy differs, so this is not proof of an autonomous physical biased cell, physical compiled-device count or hardware reset/readout energy. The earlier unary-spin THRML setup remains the direct biased-kernel reference. All source/data/logs/environment and complete artifacts are retained in TORX_STATIONARY_SEMANTICS_REPORT.md/torx_probe. No neural task score or original scientific source/budget/range/gate changed.

The reviewed official categorical domain-wall tutorial concerns categorical-state encoding, not an established four-positional-p-bit learned-weight architecture. Existing K4 comparisons remain internal research controls. Source review and pinned new-library experiments add programming evidence, not an Extropic architecture disproof or measured systems advantage. Broader neural/hardware scope stays open; goal active and global artifact snapshot waits for live writers.


## Verified wait after tool-handle retirement (2026-10-27)

Previous turn made progress by closing the independent official Torx scalar probe/quickstart while original neural replay advanced unchanged. This continuation is a verified wait on the original actual OS processes. The former unified-exec session41358 now returns Unknown process id, but original pipeline3707781/native audit3764535 remain live with identical start ticks and advancing atomic case receipts. No scientific job is restarted. Handle observation incident is retained under boolean_classifier/execution_checks/tool_session_handle_loss.json; monitoring now uses actual PID/start-tick/cmdline and saved case progress.

Native replay has passed all150R1 plus all three initial30-case R3 model groups and advances into the fourth R3 model. Full quality summaries remain queued behind complete native/statistical audits. Original and both followup source seals rechecked unchanged; static-cache helper3760254 and integer8 waiting pipeline3778471 are actually live. Global artifact index is intentionally stale until those writers finish. Goal active; this is not an external scientific blockage or a completed research objective.


## Native convolution prerequisite closed while original audit advances (2026-10-27T20:55:10.501120+00:00)

Previous SITREP turn was a verified wait: actual original native auditor and both waiting followups were confirmed live with unchanged start ticks. This continuation makes new bounded operator progress without editing any sealed original/followup source. One free p-bit per normalized signed kernel weight drives integer activation-code and validity-mask output counters in official THRML. Four synthetic convolution geometries(valid/padded/stride2/dilation2) xT1/7/64, two MC keys, two shared input images,54kernel nodes/chain. Source/inputs/pinned vendor/environment are retained. Production31104 and independent public replay31104 scheduled updates; all5904 counters exact and separate nested spatial convolution outputs agree to4.884981308350689e-15 absolute error. No full CNN training/task score or additional independent spatial samples claimed.

Padding validity requires explicit care: nominal zero maps to a nonzero rounded affine-grid value. Separate mask-gate counters preserve exact zero padding rather than charging that value to padded pixels. A separately recorded post-witness algebraic negative control using the same retained counters shows up to0.013017494515353703 spurious output error if rounded zero is used, with zero effect for valid convolution. Signed center/radius offset, field commands, diagnostic column counters, activation codes/masks, external F64 readout and software sampler floating arithmetic remain costs. Physical energy/latency/area and full native CNN coverage remain open. Closure report/artifact hashes independently reread and verified; no failed experiment, gate or prior source modified.

Original classifier native replay now267/300 with exact counters/causal outputs so far; original processes and both followups stay intact. Static-cache and integer8 followups await full original native/statistical/report closure. Global evidence index remains stale while writers are live; goal active.


## Five CNN checkpoints and full-causal native validation prepared (2026-10-27T21:00:51.618192+00:00)

Previous continuation was progress: bounded signed-range/padding/stride/dilation convolution operator independently closed. This turn preserves the same actual original native audit and both waiting helpers, with no sealed-source mutation/restart. CPU-only conventional FashionCNN training now completes five independent standard signed initializations and12epochs/model, train55000/validation5000,2x2 pooled14x14 pixels, conv1x8/conv8x16/pooling/head144x10.2664 learned weights/34deterministic biases. Lowest full-validation CE selects each checkpoint; all histories/minibatch-order hashes and distinct initial/final state digests retained. Independent audit checks split/raw data, every selected/exported coordinate, all250000 validation logits and validation-only ranges exactly. Test data/labels are not loaded/scored by training/audit, though the existing standard test set is historically consumed. Full retraining is not claimed.

New native validation-only causal preflight covers every model atT1/7/64,two banks x4inputs,all three sampled layers. Signed per-output center/radius weights encode one normalized probability each; finite8-bit free fields±4, deterministic biases, integer activation/mask/output counters, accumulation then ReLU/maxpool before next sampled layer. All566700 gate/column entries exact against separate public THRML states; alternative spatial correlations/pooling/head contractions check65520outputs withmax absolute error1.8474111129762605e-13 underfrozen2e-11/2e-12tolerance. Canonical native F64 outputs feed later quantization; alternate float-contraction ulps do not define an independently executed alternate-boundary network. Native1918080+public replay1918080 scheduled updates; same-key replay is not independent replication. No task score/prediction-quality gate or physical advantage inferred.

Future full test design retains all5checkpoints/10MCseeds/T64,256,1024: planned223776000000 active weight updates at8images/bank, not executed. Raw patch-counter retention would require28309500000bytes, abovecurrent free capacity. Full inference/control/statistical protocols and compact per-bank counter/output hash receipts with complete public regeneration remain to implement/source-seal before test scoring; GPU throughput validation also remains. Current preparation closure/artifact hashes verify, with one report-only grammar correction/history retained and no data/source/number change. Goal stays active, broader pretrained/physical scope remains open, global evidence index stale while live original/followup writers continue.


## Complete classifier/cached reuse closed; CNN compact recovery ready (2026-10-27T21:06:29.411867+00:00)

Previous continuation was progress: five CNNs trained/independently exported and all15native causal validation witnesses closed. This turn advances the unchanged original classifier through all300public-native replays and all automatic statistics/audit/report stages, terminal exit0. All514875000counter entries/derived hidden controls/final outputs are exact. Independent statistics cover300case metrics/35model controls/90000bootstrapvalues. Production609168000000 and replay609168000000 scheduled updates are separately charged. Root review rereads all300case hashes, original completion-bound artifacts and critical sources. Final largest-budget R1T1568accuracy89.6428%,R3T1184accuracy88.1458%,FP3289.8800%;R3minusR1-1.4970pp(descriptive95%[-1.6588,-1.3594]). Allbudgets/contrasts retained, five model units/ten MC seeds, historically consumed test images, no invented outcome gate. Frozen original report accidentally emits literal clipping placeholders; companion BOOLEAN_CLASSIFIER_REVIEW.md/root_review.json supplies actual0hidden/0input counts. No completion-bound original report/source/statistics bytes changed.

Original static-cache helper autonomously runs after full parent completion and closes150derivedcases, all15millionlogitvalues/1.5millionpredictions-losses/30000bootstrapvalues. Bank0was preselected for everycase. Cached largest-budget accuracy89.6548%vsper-bank89.6428%. Existing selected243667200updates already belong to parent executed work; zero new sampler draws. Reusing two reconstructed F64 matrices requires18944extra bytes/model;1250times fewer sampling banks is a conditional deployment comparison, not physical speedup or retroactive work reduction. Completion artifacts independently rehashed. Original integer8followup autonomously starts actualGPUproduction after parent audited completion; current15/150cases, no newqualitypromotion.

CNN compact format now validated on all15originalbounded validation cases: independent public sampler/patch/integer-contraction recovery reproduces420bank arrays byte-exact, including canonicalF64causal boundaries. Explicit name/dtype/shape SHA256 headers and independent serializer checked;60value/shape/dtype/name mutations detected. Additional1918080public updates/no newnative updates, not independent quality replicas. The hash receipt is not lossless counter decoding; pinned-source/key/data/public regeneration and exact digest checks are required. New vectorized patch preparation actually executes the same officialnative observer for all15cases, all420arrays byte-exact, additional1918080native updates/no public draws. Closed raw bounded evidence remains unchanged. Planned150-case compact hashes84000000B+finalF64logits120000000B=204000000B before supporting artifacts/reserve, rather than28309500000B patch counters. Full testproduction/control/statistical sources remain to implement/source-seal before scoring.

A new actual live watcher prepares/source-seals GPUCNNvalidation benchmarkT64/256/1024/model0/16validationinputs/two8imagebanks. It waits on actual integer-parentPID3778471/start44679072 and its full audit/completion hashes. One cold+one warm native pass plus separate public recovery perbudget are planned, no testlabels/scores or claimed physicaltiming; none executed yet. Nativefuture14321664/public7160832updates are planned/charged only whenrun. Runtimepreparation uses identical observer/counters, no reduction of budgets or taskscope. Goalactive, original/followup seals unchanged, globalindex remains stale until allwritersclose.


## Full native CNN protocol sealed and real runner queued (2026-10-27T21:15:57.944523+00:00)

Previous continuation was progress:300classifier/cached reuse audits closed; CNN compact/public recovery and vectorized native witnesses closed; GPUthroughput prerequisite source-sealed and actuallywaiting. This turn preserves the same integer8producer/parent and GPUwatcher. FullCNNcohort now has9newsource modules and a complete fixed150-case protocol:5trainedmodels x10MCseeds xT64/256/1024, full10000historicallyconsumed testimages,8images/bank,chunk4, per-output signed range/finite8bit freefields, deterministicACT8/maskedpadding, outputaccumulationbeforeReLU/maxpool. Tenconventional conditions x5models and all30pairedcontrol contrasts/5000hierarchicalbootstrapdraws fixed. Nativeproduction223776000000+publicreplay223776000000are planned, not executed. Other FP16/BF16/matched-spatial CNN ablations remain broader-roadmap scope; no fullmastercompletion claimed.

Prescoring gates: all50Torch/NumPy control validation paths, all150independent control-weight arrays byte-exact, newfast public recovery of all420oldnative bank arrays byte-exact(1918080additionalpublicupdates/0native), and1800distinct chunk-invariant keys acrossallmodel/MC/T/layerroots at first4banks. Allsource versions archived. Fullprotocol hashes these gates, source/data/model/rawIDX and parentGPUprotocol before anytestexport/scoring. Actual fullCNNpipeline nowwaits on GPUvalidation parentPID3790833/start44969679; no CNN testdata/control/native manifest yet. Sequentialcontrols/production/publicaudit/statistics/independentaudit-report autoqueue after parentauditedclosure. No frozenoldsource modified.

Independent prescoring resource audit checks37746counterentries/bank,7077375000fullcohort integerentries/2625000semanticdigests,204000000bytes uncompressedreceipts+logits vs28309500000counterbytes notretained. Nominalparameterpayload3392B/model explicitly includes2664commandbytes+544rangebytes+136biasbytes+48activationrangebytes;2664dynamicweightbits separate. Actualsoftwareobservercounterbytes150984acrosssavedlayers(maxlayer100384), notsmallweight-onlymemory, physicalmust-coexist claim or totalRAM/area estimate. LargestT1024gatecounter37601280fitsint32. Inspecting official schedule confirmswarmup1firstrecordplusT-1sweeps=Ttotal, so activeupdateledger correctlyincludesinitialstep. Source guards and capacity450MBbeforecontrols/400MBbeforeproduction/50MBpercase fixed. No hardwareenergy/latency/area, freshgeneralization or favorablequalitygateclaim.

Current integer8production71/150unchanged cases, fullquality stillawaitsallnative/statisticalchecks. Goalactive, broaderrealhardware/pretrained/nonideality/training scope intact, globalartifactindex stale until livewritersclose.


## Conventional FP16/BF16 CNN supplement queued (2026-10-27T21:21:06.019270+00:00)

Previous continuation was progress: complete150caseCNNprotocol/controls/replay/statisticsrunner source-sealed and actualwaiting. This turn confirms unchanged integer8producer/parent/GPUwatch/CNNparent identities and sourceguards; main integer8advances to101/150 without restart or quality promotion. The broader master-plan FP16/BF16 controls are now separately implemented before test evaluation, leaving the original fullCNNprotocol/case budgets intact. Fourconditions x5trainedmodels: FP16/BF16weight-onlyrounding+F64arithmetic/originalFP32biases, and full pinned TorchCPU lowprecisioninputs/weights/biases with observed output dtypechecked. Internalaccumulator width/hardwareprecision not asserted. No extra native sampler calls planned or executed.

All20validation-only paths/64inputseach pass. Independent NumPyFP16conversion/BF16RNEbitrounding matchesactualTorch weight/biascasts; separately constructed functional lowprecisiongraphs reproduce module outputs exactly, and weight-storageF64controls match independent NumPyspatialcontractions at2e-11absolute/2e-12relative tolerance. No labels/taskscoring in gates. Source versions archived and checks/protocol bind original five npz+ptcheckpoints, parentprotocol and all criticalsources. Actual newwaitingprocess has two dependencygates: evaluateCPUcontrols onlyafterparentcontrol/testexports arecompletehash-valid, then waitforfullparentnative/statisticalcompletion before12pairedcontrasts reuseparent5000bootstrapdraws. Allcases/contrasts retained onhistoricallyconsumedtestdata; no freshgeneralization/favorableformatgate/independentMCclaim.

Formatpayloads5328weightbytesplus136biasbytes(storage-only)or68biasbytes(fullformat) are nominalparameterbytes, nottotalframeworkRAM/hardwarememory or energylatency. Expectedrawcontroloutputs16MB andbootstraparrays~1MB;50MBfreeguardbeforeevaluation. Parentglobalinput/protocols unchanged, realCPUbackendversionsTorch2.14.1/NumPy2.5.3recorded, GPUunusedforthissupplement. Newpreparationclosure rehashed; globalindex stale untilallwritersfinish. Goalactive foractualfullCNNexecution, matched-spatial/nonideality/training/broaderpretrained andphysicalcharacterization scope.


## Verified live wait and conservative remaining-capacity audit (2026-10-27T21:25:31.333296+00:00)

Previous goal continuation was progress: FP16/BF16supplement implemented,20validationpaths passed and realrunner source-sealed/queued with0newnativeupdates. This continuation verifies the original actual integerGPUproducerPID3790893/start44970110 and allfouroriginal/queued parentprocesses, with unchanged sourceguards. Native committed cases advance106→115→119→127during observation; no job is restarted or scientific source changed, and quality remains pending complete native/statistical checks. This is a verified wait on the actual producer, not a stopped/blocked experiment.

A reproducible read-only capacity helper nowcaptures immutablemanifestbytes/sourceidentity/liveness and bounds allremainingoutputpayloads, includingZIPoverheads, integercontrol/statistics, GPUpreflight, CNNcontrols/fullcases andFP16/BF16outputs/statistics. At115committedintegercases it verifies102sourcechecks, observed674775040freebytes,409200000remainingoutputbound and265575040residualbytes beforeexternalactivity. Conservative projectedCNNguardchecks pass(542075040beforecontrols vs450000000guard;474075040beforenative vs400000000guard). Existing phase guards remain authoritative; the estimate is notspace reservation, totalRAM or physicalhardwarecost. In-flight/partialfiles maybedoublechargedconservatively. Originalcapturedmanifest/sourcehashes rechecked. No cleanup, deletion, new native draws or protocol/qualitygate change.

Goalactive; currentintegerquality/GPUCNNvalidation/fullCNN/precisionevaluations remainpending. Globalartifactindex stays stale whilewritersrun.


## Complete direct integer classifier reviewed; full native CNN production starts (2026-10-27T21:35:25.518358+00:00)

Previous turn was verified wait plus capacity-audit progress. This continuation monitors the unchanged actual producer through129→139→145→all150committed cases, then its terminal exit0 and automatic count/causal/statistics/independent-report stages, all exit0. Root review rereads all150case+public-parent hashes and completion-bound artifacts/critical sources.70875000integer counters/all hidden codes/15million final logits exact;25deterministicmodelcontrols/90000bootstrapvalues checked. Actual new native observer work304584000000scheduled updates; full identity audit0newrawpublic draws because parentR1counts already public-replayed exactly and passive observer inputs cannot change weight state. No extra independent MC/generalization claimed.

Direct integer accuracyT98/392/1568=86.7908/89.0200/89.6378%,FP3289.8800%. Largest paired accuracy difference vsR1=-0.0050pp,descriptive95%[-0.0224,+0.0132];CE+0.00016472nats. Difference vsFP32=-0.2422pp,interval[-0.3756,-0.119595]. Difference vsR3equalactiveupdates=+1.4920pp[1.35998,1.645005]; activationcoding/activationnoise/sample-rounds all differ, so not isolatedcausalsinglenoise proof. Five modelunits/tenMCseeds/fullhistoricallyconsumedtestset; no outcome gate added. All2368matrixweights free8bitfields;42biasesdeterministic,0activationpbits,378nativeint32outputcountersvs2368reconstructioncounters. Sumcarry1512vs9472B/logicalpacked9024vs26048bits atT1568; wider integergates/activationcodebuffers/F64readout/controlretention/hardwarecost stillcounted. No hardwareenergy/latency/area or totalmemoryadvantage.

Original GPUCNNwatcher autonomously releases afterintegercompleteaudit. AllT64/256/1024full-budgetvalidationcold+warmnative/publicrecoverypass, all84bankarrays byte-exact. Native14321664+public7160832executed updates, not independentreplicas; wallclockcold~2.2-2.4s/warm0.0147/0.0243/0.0387s for2banks includinghostpreparation/transfers/readout, notphysicaltiming or reliablefullcohortforecast. Oldsource/artifactsunchanged. Original fullCNNpipeline thenruns all50conventionalcontrol paths terminal0 and launches actualGPUproduction; now24/150committedcases. Precisionwatcher autonomously evaluates all20FP16/BF16fulltestcontrols, terminal0, no new native calls and nostatisticalcomparison untilfullparentaudits. Allsourceguards intact.

The broader long-horizon scope remains active. FullCNN150caseproduction/public-recovery/statistics/controls+format audits, broader pretrained/nonideality/training/physicalscope remain open. Globalartifactindex stays stale while new actualwriters are live; no jobrestart, earlynegative suppression or originalgatechange.


## Full CNN control audit and matched4nodeoperator validation (2026-10-27T21:46:32.406559+00:00)

Previous continuation was progress:150integerclassifier fullyclosed/rootreviewed, full-budgetGPUCNNprerequisite closed and realCNNproduction/precisioncontrols released. This continuation confirms originalCNNGPUproducer/parent/precisionwatch actuallylive with unchanged startticks/sourceguards, production advances29→36→58→62→73/150. No jobrestart or scientific source change. To resolve pending numerical-control audit risk before expensive full publicreplay, an independent early50-path check now reconstructs rawIDXpooling exactly and every conventional control on all10000inputs, using the already frozen final auditor weights/NumPyspatialreference/tolerances. All5millionlogits/500000predictions agree, includingF32numericalreference argmax;0failedpaths. No labels/accuracyselection/stochasticdraws or gate relaxation. Original final auditor remains unchanged and stillruns after complete native evidence.

The roadmapmandatorymatched-spatialCNNcomparison now has an actual native internalK4operator and complete all-five-model causal validation witness. Fourfreepbits/normalizedweight program logicalwordbits atfinitefields±4, per-outputminmaxc/r, samefixedACT8ranges, maskedpadding, deterministicbias/F64range-readout/ReLU/pool. Observer sums bit-gated activationcodes with shifts1/2/4/8; denominator15*T; no operational per-weight reconstructioncounter array. All25cases(5models,Tword1/4/16/64/256,2banksx4validationinputs) reproduce944500integer entries and700bankarrays byte-exact via separatepublicstates/independent wordcodes/fields/integercontraction/readout. Native36336960+public36336960executed updates, not independentextraqualityreplicas. All10656nodesfree,0activationnodes, largest bound141004800fitsint32. No labels/task scores used. Protocol/source/rawcases/closure hashes retained and independentlyreread.

This baseline is an internal positional reference, not an established Extropic productweightformat or disproof. Future comparison T1=64/256/1024 vsT4=16/64/256 matches4*T4=T1schedulednodeevaluations, but nodes/sample rounds/controllercodes/readout/biasretention/holdingenergy differ. LogicalK4targetcode4bits/weight vsone-node8bitfieldcommand; same range/biasmetadata plusdifferent sharedfieldset descriptions. Nominalsigmoid(-8) wordmean epsilon+(1-2epsilon)*code/15 hasfinitebias/variance; exact1/0 notclamped. GPUvalidation/full150-case matchedresourceexecution/statistics remain toimplement before fullwordqualityclaim. MainfullCNN/public/stats andprecisionaudits stillpending; broader pretrained/nonideal/training/physicalscope intact, goalactive, globalindexstale duringwriters.


## Full150case matched word CNN comparison sealed/queued (2026-10-27T22:00:22.335080+00:00)

Previous continuation was progress: full50-path conventional CNNcontrols checked andK4nativeCPUreference/witness closed. This turn preserves original mainCNNproducer/parent/precisionwatch identities and sourceguards while advancing79→105→114→122→131/150. New word comparison8source modules and fullsequential runner are source-sealed before wordtest scoring. All5models/10MCseeds/Tword16,64,256 matchparentsingleT64,256,1024 by4*Tword=Tsingle, identical c/r/ACT8ranges/padding/bias/data. Nodecount10656vs2664; newwordproduction223776000000/public223776000000areplanned, not executed. GPUfullbudgetvalidationcold/warm/publicreplay→nominalmeancontrols→150nativecases→allpublicregeneration→metrics→independentTorch/NumPycontrols/bootstrap/resources/report allautoqueueafterfullparentclosure.

All5nominalwordmeancontrols validate2664weightcoords/model frombitprobabilities, then canonicalF64weights feed distinct NumPy/TorchCNNgraphs over64validationinputs;max errors~1.4–1.8e-14 underprespecified2e-11/2e-12outputtolerance. Nominalweightmeanformula crosscheck2e-14relative/2e-15absolute accountsF64association/cancellation; no labels/scores or nativecalls. CanonicalmeanpluginnotE[nonlinearfiniteCNN]. WordMCbootstrapseed1587520independent ofparentMCseedindices, models paired usingparentmodeldraws. Cross-envNumPy2.5.3/2.5.4indexarrays bothSHA2fc65eed19f6bac6afd65b40fbad1cb481ae398468664141a5275711d97cb767. Initialshell-cinvocationslostdictquotes/NameError; correctedquotedheredocs bothterminal0; failure/repairrecorded, noexperiment/source/datachange.

Independentprepareauditchecksbudgetproducts, source/model/data seals,141004800counterbound,int32safety, nominalwordpayload2060Bvs3392Bsingle+dynamicstate1332Bvs333B andsame37746observercounters. This internalwordisnotassertedExtropicarchitecture or a hardwarewin. Existing capacityhelperextendednonexperimentally(olderbytesarchived/sourcechecksretained) todetectactualwordrunner andinclude217500000remainingoutputbound. Atmain122casesitvalidates141sourcechecks/actualprocessesorclosedreceipts,829837312freebytes,276860000allremainingbound,552977312residual;wordprestageprojection770477312clears450MBguard. Source-bound mainexperiments unchanged; nosamplingrestart/deletion/calibrationhardwaremessage.

Goalactive; complete one-nodeCNNnative/public/statistics/formatcontrols and allmatchedwordexecution/results pending; broaderpretrained/nonideality/training/physicalscope stillopen. Globalindex stale duringactualwriters.


## One-node CNN production complete; independent raw recovery actually live (2026-10-27T22:09:40.930126+00:00)

Previous turn was progress: fullmatchedwordCNNprotocol/runner source-sealed aftermean/control/environmentgates and actualqueued. This continuation is a verified wait that advances the unchanged original CNNproducer143→147→all150, terminalexit0 at2026-10-27T22:05:28.207490UTC. Full production223776000000scheduled evaluations. No quiet-log/tool timeout restart or source/protocol/gate change. Root review independently rereads all150case hashes, exactcoverage(5models/3budgets/10MCseeds), receipt/logitdtype-shapes/finiteness/clippingmetadata and finalproducer/source/datahashes. Producer manifest stays immutable with its pre-audit independently_audited:false flag; separate finalaudit/closure supplies outcome later. No labels/taskmetrics consumed by thisreview.

Original parentpipeline autonomously launches fullpublicnativeGPUaudit, wrapper3801644/actual3801645start45335513. All14semanticcode/mask/integercounter/canonicalF64output arrays perbank mustregenerate byte-exact plusretainedfinallogitsdirectly. Snapshot21/150passes/990832500exactcounters sofar. Fullpipeline/precisionwaiter/wordwaiterremain actuallylive with unchangedstartticks/sourceguards. CompleteCNNquality remains unpromoted until allpublic/stat/control/format auditsaresentinalpass. Broaderresearchscope and fullwordcomparison/physicalrequirements remain open; goalactive andglobalindexstale duringwriters.


## Verified wait on unchanged full CNNpublicaudit (2026-10-27T22:16:36.056451+00:00)

Previous continuation completed all150nativeproductioncases/rootcasehash-schema review and launched the actualfullpublicauditor. This continuation verifies same parent3792481/start45024359 and auditor3801645/start45335513 live, plusprecision3794515/start45061769 andword3800388/start45280544 actuallywaiting. Public committedcases27→41→52; 2453490000integercounters/allassociatedsemanticcode-mask-F64outputdigests exactsofar. Allfrozen sources and finalproducer manifestcb5f2ad442ede4b83b50a9b21b6652c9fdaf0fd5e8e3f86864859ee65d788f3a unchanged; no restart/newtests/protocolchange/qualitypromotion. This is verifiedlivewait, not externalblockage. Fullpublic/stats/format/matchedword results and broaderscope remainpending, goalactive/globalindexstale.


## Verified public-CNN wait (2026-10-27T22:18:29.287635+00:00)

Previous turn was a verifiedwait with allsourceguards intact. Sameactualauditor3801645/start45335513 andparent3792481/start45024359 remainlive; source-sealedprecision/wordwaiters3794515/3800388 alsoverifiedlive. Publiccasecounts52→55→58 and 2736585000integercounterentries exactsofar. No source/data/gatechange, samplingrestart, redundanttest or prematurequalitypromotion. Goalactive; finalpublic/statistical/format/matchedwordresults pending, globalindexstale whilewritersrun.


## Verified unchanged CNNpublicwait (2026-10-27T22:21:02.268833+00:00)

Previous turn wasverifiedlivewait. This continuation pollsactualauditor3801645/start45335513, unchangedparent3792481 andbothprecision/wordwaiters withcorrectstartticks/sourceguards. Committedpubliccases58→66→72; 3397140000integercounters andassociatedcode/mask/canonicaloutputdigests exactsofar. No source/data/protocolchange, newstochasticsamplesoutsideplannedreplay, restart or prematurequalitypromotion. Goalactive; remainingfullpublic/statistical/format/matchedwordaudits pending. Globalindexstale duringwriters.


## Verified unchanged public-CNNaudit wait (2026-10-27T22:23:25.623700+00:00)

Previous continuation wasverifiedlivewait. This turn confirmsactualauditor3801645/start45335513 andoriginalparent plusprecision/wordwaiters livewiththeiroriginalstartticks. Allsourceguardsunchanged. Committedpubliccases72→77→82;3868965000integercounters/allassociatedcode-mask-canonicaloutputdigests exactsofar. No restart, source/data/gatechange, newtestorqualitypromotion. Goalactive, finalpublic/statistical/format/matchedwordchecks pending; globalindexstale whilewritersrun.


## Verified unchanged publicCNNwait (2026-10-27T22:24:56.942815+00:00)

Previous turn wasverifiedlivewait. Same actualauditor3801645/start45335513, parent andprecision/wordwaiters remainlivewithunchangedsourceguards. Publiccases83→85→87 and4104877500integercounters/allassociatedoutputdigests exactsofar. No source/protocolchange,restart,newtestorqualitypromotion. Goalactive; finalreplay/statistics/format/matchedwordresults pending, globalindexstale duringwriters.


## Verified publicCNNwait (2026-10-27T22:26:23.446855+00:00)

Previous turn verifiedlivewait. Same auditor3801645/start45335513, parent andcomparisonwaiters actuallylive, allsourceguardsunchanged. Publiccases88→90→95;4482337500counters/allassociatedsemanticoutputs exactsofar. No source/data/gatechange, restart, newtest or prematurequalitypromotion. Fullpublic/statistical/format/matchedwordresults pending, goalactive/globalindexstale duringwriters.


## Verified ongoingCNNpublicaudit (2026-10-27T22:27:43.370210+00:00)

Previous turn verifiedlivewait. Sameactualauditor3801645/start45335513, originalparent andprecision/wordwaiters remainlive/sourceguardsunchanged. Publiccases95→99→103;4859797500integercounters/allrelatedsemanticoutputs exactsofar. No scientificsource/protocolchange,restart,newtests orqualitypromotion. Goalactive, finalreplay/statistical/comparison results pending, globalindexstale.


## Verified unchanged CNNpublicwait (2026-10-27T22:29:02.709136+00:00)

Previous turn verifiedlivewait. Actual originalauditor3801645/start45335513 andparent/comparisonwaiters remainlive/sourceguardsunchanged. Publiccases103→106→109;5142892500counters/associatedsemanticoutputs exactsofar. No source/protocolchange,restart,newtests orqualitypromotion. Finalpublic/statistical/format/matchedwordresults pending, goalactive/globalindexstale.


## Verified unchanged CNNpublicwait (2026-10-27T22:30:21.621716+00:00)

Previous turn verifiedlivewait. Sameactualauditor3801645/start45335513 andoriginalparent/precision/wordwaiters live/sourceguardsunchanged. Publiccases110→112→114;5378805000counters/associatedsemanticoutputs exactsofar. No source/data/gatechange,restart,newtests orqualitypromotion. Finalpublic/statistical/format/matchedwordresults pending, goalactive/globalindexstale.


## Verified unchanged CNNpublicwait (2026-10-27T22:31:40.385816+00:00)

Previous turn verifiedlivewait. Sameactualauditor3801645/start45335513 andoriginalparent/precision/wordwaiters remainlive/sourceguardsunchanged. Publiccases114→116→118;5567535000counters/allsemanticoutputs exactsofar. No scientificsource/protocolchange,restart,newtests orqualitypromotion. Goalactive; finalpublic/statistical/comparisonresults pending, globalindexstale duringwriters.


## Verified unchanged CNNpublicwait (2026-10-27T22:32:57.688451+00:00)

Previous turn verifiedlivewait. Sameactualauditor3801645/start45335513 andoriginalparent/precision/wordwaiters live/sourceguardsunchanged. Publiccases118→121→125;5897812500counters/allassociatedsemanticoutputs exactsofar. No source/data/gatechange,restart,newtests orqualitypromotion. Goalactive, finalpublic/statistical/comparisonresults pending, globalindexstale.


## Verified ongoing CNNpublicwait (2026-10-27T22:34:15.012633+00:00)

Previous turn verifiedlivewait. Actual originalauditor3801645/start45335513, parent/precision/wordwaiters remainlive withunchangedsourceguards. Publiccases125→129→132;6228090000counters/associatedsemanticoutputs exactsofar. No scientificsource/protocolchange,restart,newtestorqualitypromotion. Finalpublic/statistical/comparisonresults pending, goalactive/globalindexstale.


## Verified unchangedCNNpublicwait (2026-10-27T22:35:33.404623+00:00)

Previous turn verifiedlivewait. Sameactualauditor3801645/start45335513 andoriginalparent/precision/wordwaiters live/sourceguardsunchanged. Publiccases133→136→139;6558367500counters/allassociatedsemanticoutputs exactsofar. No scientificsource/protocolchange,restart,newtestorqualitypromotion. Goalactive; finalpublic/statistical/comparisonresults pending, globalindexstale.


## 2026-10-27T22:39:49.112850+00:00 — full native CNN and precision closures

The previous SITREP turn was a verified wait: actual CNN auditor PID3801645 and dependent waiters were live; replay progressed to148/150. This turn observes terminal success of all five CNN stages, final public replay at150/150 and all7077375000counters exact. Production223776000000updates and replay223776000000updates remain separately charged. Independent verification checks50controls/150metrics/300000bootstrapvalues. Full CNN quality T64/256/1024 is66.0010/78.5182/83.0638%vs85.0860%FP32; largest loss2.0222pp is retained, not a declared gate pass. All20FP16/BF16 controls/full metric contrasts also independently close successfully, no additional native draws. New review_closed.py is archived by SHA before execution; it rechecks completion/source/input/case hashes and recomputes all150retained-logit metrics, exit0, writes root_review.json. Added human-readable FASHION_CNN_NATIVE_REVIEW.md and current scope/status links without editing completion-bound artifacts. Matched four-node CNN waiter advances to actual GPU validation at22:38:51UTC, PID3806775wrapper; no fullword quality yet. Global index remains stale during writes; goal active.

Follow-up actual-state check: word GPUvalidation and nominalcontrol both exit0; all84preflightbankarrays byte-exact. Fullwordproduction started2026-10-27T22:39:09.710106UTC, actual childPID3807224/start45537658. Five cases committed at22:39:53UTC; no word quality promoted before fullaudits. Capacityaudit20261027T223953records141sealedsourcechecks and~755.9MBfree against~210.65MBremainingoutputbound. MainCNN/precisionparents are terminalsuccessful, not stalled; originalwordparentPID3800388remainslive.


## 2026-10-27T22:41:48.517279+00:00 — conditional CNN first-layer moments

Previous goal turn classified progress: fullCNN/precision audits closed, reviewed and documented. This turn confirms actual wordproducerPID3807224/start45537658live;14/150cases committed at22:41:25UTC,5541120000scheduled nativeupdates. No restart/sourcechange. Added separate first_layer_moments.py, SHAarchived and protocolfrozen before computation, no labels/new native draws. Exact conditional nominal mean/variance/bias and shared-kernel aggregate covariance on5models/16validationinputs/2banks/8channels/2methods/3budgets yields480rows. Mainformula checked against512-state single-kernel enumeration and16-state per-weight word enumeration; separate verify_first_layer_moments.py(archived before run) independently reconstructs indexed patches and enumerates2/16 scalar states, all2880moments andsummaries pass. Report documents low-variance/coarse-bias tradeoff and shared-kernel covariance without predicting nonlinear full-CNN quality. Original completionbound experimental sources/artifacts unchanged. Globalindexstale; goalactive.


## 2026-10-27T22:46:38.554567+00:00 — CNN probability fine-tuning preparation

Previous turn progress:480rowfirstlayermomentaudit. CurrentwordproducerPID3807224/start45537658verifiedlive;36cases at22:45:51UTC. Newprobability_finetune sourcesarchivedbeforeselfcheck. Initialgradientcheckfails endpoint16/72 dueTorchclampderivative0; sourcebytes/failurepreserved. Removedredundantforwardclamp, optimizerprojectionkept; all30gradients/5graphs/15exportspass. ProtocolSHAdae682caf4653217cfb9413c0b1589f49a230d554e3df910efc967f795a778cefrozenbeforetraining. CPUactualPID3809757/start45569886completes10fits/80epochs/17200steps exit0. Onlyfieldslearned, originalc/r/biasfixed, spanscastF32fixed; expectedandT256binomialmeanSTEallretained. Selection4fixedseedvalidationreplicas epoch0through8, no test/nativecalls. Prematureauditguardrefusal/failedlogpreserved, trainingnotrestarted; afterexit0 independentverify_probability_finetune.py exactlyreplays2millionselectedvalidationlogits,80orders/10exports/selectionchecked, passes. MaxtrainingF32vsnominalexportpdelta1.48694e-7retained. ReportgenerationfirstusedsystempythonwithoutNumPy; failedimportbeforewrites, rerunvenvsuccess. preparation_completionseals30artifacts. Nativeadapterpreserveslearnedcommands/originalranges; separateTHRMLvalidation/fullconfirmationnext. Originalgatefailureunchanged, globalindexstale, goalactive.


## 2026-10-27T22:49:20.071039+00:00 — learned-command native validation prepared and queued

Previous goal turn progress: all10probabilityfits/selectedvalidationaudit closed, failuresretained. Current turn verifies originalwordproducerPID3807224/start45537658live,52cases at22:48:36UTC. Implements newprobability_native/{adapter,replay,preflight,watch,prepare_check}.py; sources archived bySHA beforeuse. PureNumPy adapter consumeslearnedcommands/originalfixedranges withoutTorchimport orrange-refitting. Independent publicreplay rederivescommands fromlatentfields andrangesfromoriginalweights;30cases plannedall10models/T64,256,1024/2banks8validationinputs,all14arraysmustmatch. Planned71608320native+samepublicupdates,2264760counterentries, no labels/scores. Static all26640commands/30layerexports/counterbounds pass; payload16305120B raw,20MBsupportbound. ProtocolSHA072ebb36338d5c8fab8ca9c2df71e72228e4692702493b2e162ff9b9f804fcf8sealedbeforeexecution,37sources/23inputs. ActualwaiterPID3849492/start45594336 waits onrealwordparentPID3800388/start45280544/fullsuccessreceipts. Capacityhelperupdatedoutsidefrozenexperiment,oldsourcearchived,newSHA890d5fe23f41faf56d8b7068873316d4fc39ad4c5a60bb5caf36a877790ef3d3. Audit178sourcechecks passes,53wordcases at22:48:53,669319168Bfree,164890000Bremainingword+newvalidationbound,residual504429168B. No nativefinecasesyet/fullqualityconfirmationstillopen. Humanreport/preparationseal/docs/evidencemap updated withouteditingcompletionbound originals. Globalindexstale; goalactive.


## 2026-10-27T22:54:31.795713+00:00 — full learned-probability CNN cohort prepared and queued

Previous turn progress:30caselearnedcommandnativevalidation sourcefrozen/waiteractual. This turn implements probability_cohort/{common,run,verify,pipeline}.py withall300plannedcases, completepublicrawrecovery,10meancontrols,33pairedmodel/independentMCcontrasts andindependentfullcontrol/statisticsverifier. Sourcebytesarchivedbeforeuse;10model64validationcontrolgraphs passmax2.14e-14/identicalpredictions; newbootstrapindices(2,5000,5,10)byte-identicalNumPy2.5.3/2.5.4SHA d1afbc9dc57d9326da0482cfd3c637036731acea7ae017ca5ce21572cbfab449. Preparationinitiallyrefusesparentcmdcheck becauseNUL-separatedcmdlinewasmatchedwithspace; parentactuallylive, no protocolwrittenthen. Failure/sourcepreserved, normalizeNULsinprepare/waitsource, archivecorrectedsourcebeforefreeze; initialfailedapply_patchatomic/nochanges, correctedpatchsucceeds. ProtocolSHA4229d8d98b18e771a78bc4abd7d3b8eb188bf0991ede5c6614f028ebae075414sealedbeforefulltestscoring. ActualnewwaiterPID3851463/start45624181waitsonnativevalidationPID3849492/start45594336; bothlive. Planned447552000000native+samepublicupdates,14154750000counters,430MBfulloutputbound; no finefullnativecasesyet. Capacityhelperupdatedoutsidefrozenexperiment,oldsourcearchived,newSHAf6467e6d153c7ac4156321ddb2bebd072865fdb72735398774ce124822bd8d89. Currentaudit219sourcechecks pass,80wordcases22:53:51UTC,630853632Bfree,557900000Bremainingallwriters,residual72953632B;projectedbeforefinecontrols502953632>480M/native494953632>472M. Runtimeguardsunchangedauthority. Report/evidencemap/docspreparationsealwritten; baselinequalitytargetmisspreserved, trainingeffectnotisolatedrepresentationclaim. Globalindexstale, goalactive.


2026-10-27T22:56:39.273228+00:00 — verified wait on actual wordproducerPID3807224/start45537658, wordparent3800388/start45280544 and two learned-probability waiters3849492/start45594336,3851463/start45624181, allconfirmedlive now. Wordproduction progressed84to91/150;134478720000updatescommitted. Initialcapacityaudit219sourcechecks/remainingwriterbounds passes. Newpollcheckpointresults/thrml_spw/fashion_cnn_native/execution_checks/20261027T225639_verified_wait.json capturesactualcmdlines/starttimes/stages/manifesthash; no restarts, sources or scientificoutcomes changed. Goalactive/globalindexstale.


2026-10-27T22:58:34.555515+00:00 — verifiedwait: originalwordproducerPID3807224/start45537658stillactuallylive, pluswordparent3800388/start45280544 andlearnedvalidation/fullwaiters3849492/start45594336,3851463/start45624181. Productionprogress92to103/150,138954240000updatescommitted. No sources/restarts/qualitymetrics changed; newactualprocess/stage/manifestcheckpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T225834_verified_wait.json. Fullwordreplay/statistics andqueuedlearnedCNNexperimentspending. Goalactive/globalindexstale.


2026-10-27T23:00:31.720483+00:00 — verifiedwait: wordproducerPID3807224/start45537658actual live, plusparent3800388/start45280544 andlearnedvalidation/fullwaiters3849492/start45594336,3851463/start45624181. Wordproduction104to113/150,155151360000updatescommitted. Process/case/stageevidence recorded results/thrml_spw/fashion_cnn_native/execution_checks/20261027T230031_verified_wait.json. No sources/restarts/qualityclaims changed. Fullreplay/statistics/fineexperimentspending, goalactive/globalindexstale.


2026-10-27T23:01:45.606577+00:00 — capacity progress while originaljobsremainlive. Previous turnverifiedwait(actualwordproducer/waiters). Initialcapacityaudit219sourcechecks passes butforecastresidual64.1MB, fullcohortstageguardmargin14.1MB. Read-onlyinspection findsdisposablepipcachecuBLAS13.1.1.3matchinginstalled .venv metadata. Newseparatecleanupsource26647f3c47728708dc9021ed9ef8fc80734ef55e6cfc0d9fdaf20de944e8abc9 archivedbeforeexecution, targetedonlythatpackage. CacheMETADATA/largestruntimebinarybytehasheschecked,inventoryfsyncedbeforeunlink,cacheowner/inode/symlinkguards andruntimeafterhashespass; actualexit0frees542850184B. Installedpackages/source/models/evidenceunchanged. Rootreviewcheckscompletion/inventory/sourceandactualdeletedpaths; wholepayloadafterhashretainedbycleanup. Freeafter1114390528Bversus571539456before. Originalwordproduction117/150at23:01:27UTC,168791040000updates; producerPID3807224/start45537658andbothdependentwaitersconfirmedlive. Postcleanupcapacityforecastrefreshed; globalindexstale/goalactive.


2026-10-27T23:03:36.457085+00:00 — verifiedwaitaftercapacityprogress: actualwordproducerPID3807224/start45537658andparent3800388/start45280544, pluslearnedwaiters3849492/start45594336,3851463/start45624181alllive. Wordproduction118to129/150,180938880000updates. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T230336_verified_wait.json. No experimentalcode/qualityclaims/restarts changed. Goalactive/globalindexstale whileproductionandnextauditspending.


2026-10-27T23:05:23.890234+00:00 — verifiedwait: originalwordproducerPID3807224/start45537658,parent3800388/start45280544 andlearnedwaiters3849492/start45594336,3851463/start45624181allactuallylive. Wordproduction130to140/150,189676800000updatescommitted. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T230523_verified_wait.json. No scientificcode/qualityclaims/restarts changed. Replay/statistics andqueuedlearnedconfirmationpending; goalactive/globalindexstale.


## 2026-10-27T23:08:48.158675+00:00 — four-node CNN production complete, public audit live

Previous goalturnverifiedwait140/150originalproducerlive. This turn monitors140→143→146→150 withactualprocesschecks, no restart. Productionactualterminalexit0 at23:08:03.153799UTC,223776000000schedulednativeupdates, manifestSHAacbf8db24acaa989345c7ec5455a1466ff3f43f49925cdd2e3c3d75c052d12f6, logSHAeee93550b0fabcdc7e0e3a9a3259fe696c611e16cb9de5f24beea4160cc2f7fa. Automaticpublicaudit starts23:08:03.188355UTC wrapper3854179, actualauditorPID3854180/start45711005confirmedlive. Newindependentword_comparison_review/production.py source793a6452ec28c3cbebee3b4e88e18057e7c99af0696ca5f092e83056daf616e9archivedbeforeexecution; exit0reviewsall150casehashes/coverage/schemas/finiteness/clippingmetadataandmatched4Tresourcecounts, no labels/taskmetrics. production_root_review.json records2publiccasesatreview,3at23:08:33UTC; completeaudit/statisticsstillpending. Oldproducer3807224nowterminalsuccessful, not failed. Wordparent3800388andbothlearnedwaiters3849492/start45594336,3851463/start45624181remainlive. Documentation/evidencemapupdated; native/sourceclosure artifactsunmodified, globalindexstale/goalactive.


2026-10-27T23:10:47.923771+00:00 — verifiedwaitonactualwordpublicauditorPID3854180/start45711005, parent3800388/start45280544 andlearnedwaiters3849492/start45594336,3851463/start45624181, allconfirmedlive. Publicreplay5to18/150,849285000integercounterentries exact. Firstmonitorreadoldsingle-nodebooleanfieldnames triggeredmonitor-onlyKeyError; inspectedactualschema all17500_bank_array_digests_exact andcorrectedread, no jobsource/data/restartchanges. Actualprocess/stage/auditcheckpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T231047_verified_wait.json. Fullstatisticaltaskcomparisonstillpending, goalactive/globalindexstale.


2026-10-27T23:12:41.481884+00:00 — verifiedwait: wordpublicauditorPID3854180/start45711005,parent3800388/start45280544 andlearnedwaiters3849492/start45594336,3851463/start45624181allactuallylive. Publicreplay19to27/150,1273927500integercounterentries exact. Actualcheckpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T231241_verified_wait.json. No scientificsource/data/restart/qualityclaimchanges. Fullstatisticaltaskcomparisonpending, goalactive/globalindexstale.


2026-10-27T23:15:28.081781+00:00 — verifiedwait: actualwordpublicauditorPID3854180/start45711005,parent3800388/start45280544 andlearnedwaiters3849492/start45594336,3851463/start45624181alllive. Publicreplay28to45/150,2123212500integercounterentries exact; checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T231528_verified_wait.json. No source/data/restarts/qualityclaimchanges. Goalactive/globalindexstale, taskstatisticsandqueuedfineexperimentspending.


2026-10-27T23:17:24.488517+00:00 — verifiedwait: actualwordpublicauditorPID3854180/start45711005,parent3800388/start45280544 andlearnedwaiters3849492/start45594336,3851463/start45624181alllive. Publicreplay46to56/150,2642220000integercounterentries exact; checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T231724_verified_wait.json. Capacity/sourceaudit23:15:33UTCpasses219sourcechecks,554205312Bconservativefutureoutputmargin. No experimentalchanges/restarts/qualityclaims; goalactive/globalindexstalewhilefullaudit/statistics/finefollowupsrunorwait.


2026-10-27T23:20:04.743430+00:00 — verifiedwait: actualwordpublicauditorPID3854180/start45711005,parent3800388/start45280544 andlearnedwaiters3849492/start45594336,3851463/start45624181alllive. Publicreplay56to73/150,3444322500counterentries exact; checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T232004_verified_wait.json. No source/data/restarts/qualityclaimchanges. Goalactive/globalindexstale; fulltaskstatisticsandqueuedfineexperimentspending.


2026-10-27T23:22:55.886954+00:00 — verifiedwait: actualwordpublicauditorPID3854180/start45711005,parent3800388/start45280544 andlearnedwaiters3849492/start45594336,3851463/start45624181alllive. Publicreplay75to89/150,4199242500counterentriesexact; checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T232255_verified_wait.json. No sources/data/restarts/qualityclaims changed. Goalactive/globalindexstale; completecomparisonandfinefollowupspending.


2026-10-27T23:25:32.529069+00:00 — verifiedwait: actualwordpublicauditorPID3854180/start45711005,parent3800388/start45280544 andlearnedwaiters3849492/start45594336,3851463/start45624181alllive. Publicreplay89to106/150,5001345000counterentriesexact; checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T232532_verified_wait.json. No source/data/restarts/qualityclaimchanges. Goalactive/globalindexstale; completecomparisonandfinefollowupspending.


2026-10-27T23:28:15.021965+00:00 — verifiedwait: actualwordpublicauditorPID3854180/start45711005,parent3800388/start45280544 andlearnedwaiters3849492/start45594336,3851463/start45624181alllive. Publicreplay107to120/150,5661900000counterentriesexact; checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T232815_verified_wait.json. No source/data/restarts/qualityclaimchanges. Goalactive/globalindexstale; fullcomparisonandfinefollowupspending.


## 2026-10-27T23:35:20.665435+00:00 — matched CNN closure, learned-native validation passed, full fine-tuned task cohort live

Previous turnverifiedwait120publiccases. Currentturnactualwordaudit121→127→132→137→142→146→150 andallsubsequentstagesexit0, finalcompletion23:33:23.318255UTC. Wordaccuracy83.0896/83.1714/83.2036%vsone66.0010/78.5182/83.0638 atmatchednodeupdates; largestword-minus-one+.1398pp interval[-.711605,+.736225],CE+.00510284 interval[-.01120221,+.02661559],bothcross0/not equivalence. All7077375000wordcounters/15mlogits/120000bootstrapvaluesand5nominalcontrols checked. Newarchivedword_comparison_review/close.py rechecksallclosure/source/input/log/casehashes/coverage and150retained-logitmetrics,exit0/root_review. Actuallearnednativewatcherautomaticallylaunches23:33:37.268670andfinishes23:33:55.161096exit0:all30cases/all840bankarrays/2264760countersnative/publicexact,71608320updateseach. Newarchivedprobability_native_review/close.py checksallclosure/source/input/log/casehashes/coverage/schemas,exit0/no metrics. Humancompanionreports/reviewclosureswrittenwithoutchangingoriginalcompletion-bound artifacts. Full300caseparent3851463/start45624181advancescontrols23:34:05.802926exit0at23:34:09.561170,actualGPUproduction3859045/start45867644starts23:34:09.569433(wrapper3859044). Two cases/426240000updatesat23:34:43UTC; no fine-taskqualityyet. Capacity219sourcechecks pass,958042112Bfreeand419260000Bremainingbound,residual538782112. Oldwordauditor/parents/learnedvalidationwaiterterminalsuccessful, not blocked; no restarts. Docs/evidencemapupdated; globalindexstale, fullgoalactive.


## 2026-10-27T23:38:32.727521+00:00 — full same-code probability-training reproduction

Previousgoalturnprogress: matchedwordCNN/nativelearnedvalidationclosed, fullfinecohortactuallyrunning. Currentturnchecksoriginalfineproducer3859045/start45867644live(10casesat23:36:05UTC). Addsprobability_training_reproduction/reproduce.py, sourcebb702d6616676e25d31599d5f03e8d978a03dc8157e291223870178ae319d640archivedbeforeexecution. Protocolfrozenbeforefreshtraining, fixedall10fits/nooutcomeselection, outputpathonlyredirected atruntimeinnewprocess; source/originaloutputsunchanged. ActualCPUreproPID3859601/start45883166executesall80epochs/17200additionalsteps andexit0, audit23:37:54.369539. All10selectedepoch/CE values, completehistorybytesand240exportedarrays(2107640values,including2mvalidationlogits)byte-identical. Original+repeattrainingcost34400optimizersteps; no newnativecalls/modelseeds/generalizationevidence. Completion/source/input/artifactlinksrechecked, report/reviewseal/docs/evidencemaprecorded. Nativecohortactual18casesat23:37:32andcontinuesunmodified. Goalactive/globalindexstale.


2026-10-27T23:40:28.427061+00:00 — verifiedwaitaftercoldtrainingreproductionprogress: actualfine-taskproducerPID3859045/start45867644andparent3851463/start45624181confirmedlive. Production22to27/300,34525440000nativeupdatescommitted; sourcecapacityaudit219checks pass,530432224Bconservativeoutputmarginat23:38:46UTC. Process/case/stagecheckpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T234028_verified_wait.json. No code/data/restart/qualityclaimchanges; fulltaskauditpending, goalactive/globalindexstale.


2026-10-27T23:43:16.696682+00:00 — verifiedwait: actualfine-taskproducerPID3859045/start45867644andparent3851463/start45624181live. Production28to42/300,48591360000nativeupdatescommitted; checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T234316_verified_wait.json. No source/data/restart/qualityclaimchanges. Fulltasknative/public/statisticalclosurepending, goalactive/globalindexstale.


## 2026-10-27T23:44:22.644120+00:00 — current research status consolidated

Previous turnverifiedwait42/300finecases. Actualfineproducer3859045/start45867644andparent3851463/start45624181confirmedlive, 47cases/52853760000updatesatcurrentread. ConsolidatedSTATUS.md into currentactivepipeline/completedreviewedresults/limitations/openrequirements instead of contradictoryhistoricalleadparagraphs. PriorSTATUSbytespreservedverbatiminsamedirectory results/thrml_spw/STATUS_HISTORY_20261027T234422Z.md SHA06345bb2002688b4769964adb492065cf3cdc079fd86cbb96345c7f774aeb30e sohistorylinksremainvalid. OriginalmainCNN/word/learnednativecompletion-bound artifacts rehashedbeforewriting; unchanged. Root/researchREADMEslinkcurrentstatusandlabelmilestoneshistorical. Consolidationreceipt results/thrml_spw/fashion_cnn_native/execution_checks/20261027T234422Z_status_consolidation.json. No source/experiment/data/seed/qualitygatechanges, globalindexstale/fullgoalactive.


2026-10-27T23:46:50.419591+00:00 — verifiedwaitafterstatusconsolidation: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production50to56/300,75870720000updates. Source/capacityaudit219checks passes,528016544Bconservativeoutputmargin. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T234650_verified_wait.json. No sources/restarts/qualityclaimchanges; goalactive/globalindexstale.


2026-10-27T23:49:45.046159+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production57to70/300,91641600000nativeupdates. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T234945_verified_wait.json. No source/data/restarts/qualityclaimchanges; fulltasknative/public/statisticalclosurepending, goalactive/globalindexstale.


2026-10-27T23:51:57.333452+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production72to80/300,100166400000updates.219sourcechecks/capacitybounds pass23:50:10UTC. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T235157_verified_wait.json. No source/data/restarts/qualityclaimchanges; goalactive/globalindexstale.


2026-10-27T23:53:46.103093+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production81to86/300,120625920000updates. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T235346_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullnative/public/statisticalclosurepending, goalactive/globalindexstale.


2026-10-27T23:57:31.071285+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production87to105/300,140659200000updates. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T235731_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullnative/public/statisticalclosurepending, goalactive/globalindexstale.


2026-10-27T23:59:46.645941+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live/nonzombie, production107to114/300,158561280000updates. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261027T235946_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullnative/public/statisticalclosurepending, goalactive/globalindexstale.


## 2026-10-28T00:02:23.317083+00:00 — full-input nominal learned-mean control verification

Previousgoalturnverifiedwait114finecases. Currentturnactualproducer3859045/start45867644andparent3851463/start45624181live,118casesat00:00:46UTC. Addedprobability_control_review/complete.py, sourceeb05a7bf7236aa31bc4bf7acc1e8f7d7a6b7c5a0495bbe7040191d34ebc86b1darchivedbeforeuse, reviewprotocolfrozenbeforefull-inputcheck. IndependentTorchF64graph reconstructscommands/originalranges vsNumPyF64producer onall10models/full10000inputs, usingexistingfrozenreferenceimplementation; maxerror3.552713678800501e-14,all1mlogits/all100kpredictions pass. No labels/loss/accuracy/nativecalls; no modifiedoriginalsources/data/stageorder. Finalfrozenverifierstillrerunsrequiredchecks. ActualCPUcheckexit0; completion/source/inputlinksreviewed/reportandrevisedcurrentSTATUSwithverbatimpriorsnapshotwritten. Fullnativecases120at00:01:24UTCand127atstatussnapshot. Goalactive/globalindexstale, fine-taskqualitystillpendingfullaudits.


2026-10-28T00:05:23.155918+00:00 — verifiedwaitafterfullmeancontrolreviewprogress: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live/nonzombie, production133to142/300,196496640000updates. Source/capacity219checks pass00:03:18UTC,459409488Bconservativeoutputmargin. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T000523_verified_wait.json. No source/data/restarts/qualityclaimchanges; goalactive/globalindexstale.


2026-10-28T00:08:46.742641+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production144to158/300,225480960000updates. All150expected-weight-trained casesproduced, sample-trainedconditionstartedinsamefrozenprocess; methodcounts{"0": 150, "1": 8}. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T000846_verified_wait.json. No source/data/restarts/qualityclaims; fullpublic/statisticalclosurepending, goalactive/globalindexstale.


2026-10-28T00:11:16.180114+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production162to171/300,237841920000updates, methodcounts{"0": 150, "1": 21}.219sourcechecks/capacitybounds pass00:09:20UTC,421082016Bconservativeoutputmargin. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T001116_verified_wait.json. No source/data/restarts/qualityclaimchanges; goalactive/globalindexstale.


2026-10-28T00:14:39.041807+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production173to186/300,269809920000updates, methodcounts{"0": 150, "1": 36}. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T001439_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullnative/public/statisticalclosurepending, goalactive/globalindexstale.


2026-10-28T00:17:09.773362+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production190to200/300,279187200000updates, methodcounts{"0": 150, "1": 50}. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T001709_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullnative/public/statisticalclosurepending, goalactive/globalindexstale.


## 2026-10-28T00:20:59.677401+00:00 — verified installer-cache capacity recovery

Previousgoalturnverifiedwait200cases. Currentinitialaudit219sourcechecks passes,203cases,256790528Bfree/143890000Bremainingbound/residual112900528B. Free space haddeclinedfasterthanoutputforecast. Read-onlymetadatareview identifies sixlargecachedwheels matchinginstalled .venv versions:triton3.8.0, nvidia-cusolver12.0.4.66,cusparselt-cu13 .8.1,nccl-cu13 2.30.7,cufft12.0.0.61,cusparse12.6.3.3. Newseparatecleanupsource87f7f74f1889191faf5ad062cd69725691b23c9838e05ebc53c212cb66f6eb0earchivedbeforeexecution; cachemetadata/installedmetadata/largestpayloadbytehashes match, owner/inode/symlinkguards pass, inventoryfsyncedbeforecacheunlink, allmajorruntimepayloadsafterhashunchanged. Exit0frees1263194252B; free239702016→1502879744B. Rootreviewcheckscompletion/inventory/source/actualdeletedpaths andinstalledmetadata/sizes, no source/env/resultsmodified. Experimentproducer3859045/start45867644andparent3851463/start45624181remainedlive,208casesat00:19:23UTC. Capacityauditrefreshed; STATUSsnapshotversionarchivedbeforecapacity/progressupdate. No restarts/qualityclaims, globalindexstale/fullgoalactive.


2026-10-28T00:24:05.014124+00:00 — verifiedwaitaftercapacityrecovery: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production223to232/300,330762240000updates, methodcounts{"0": 150, "1": 82}. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T002405_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullnative/public/statisticalclosurepending, goalactive/globalindexstale.


2026-10-28T00:26:27.101345+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production233to239/300,354631680000updates, methodcounts{"0": 150, "1": 89}. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T002627_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullnative/public/statisticalclosurepending, goalactive/globalindexstale.


2026-10-28T00:28:38.660012+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production241to253/300,362730240000updates, methodcounts{"0": 150, "1": 103}.219sourcechecks/capacitybounds pass00:26:44UTC,1361013008Bconservativeoutputmargin. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T002838_verified_wait.json. No source/data/restarts/qualityclaimchanges; goalactive/globalindexstale.


2026-10-28T00:31:19.267154+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production257to264/300,382337280000updates, methodcounts{"0": 150, "1": 114}. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T003119_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullnative/public/statisticalclosurepending, goalactive/globalindexstale.


2026-10-28T00:33:41.024685+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andproducer3859045/start45867644live, production265to273/300,403436160000updates, methodcounts{"0": 150, "1": 123}. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T003341_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullnative/public/statisticalclosurepending, goalactive/globalindexstale.


## 2026-10-28T00:42:24.644595+00:00 — all300learned-probability native cases produced, public replay started

Previousgoalturnverifiedwait273cases. Currentturnactualproducer280→286→290→294→297→300, no restart. Originalproducer3859045terminalsuccessful; stageexit0at2026-10-28T00:39:55.659483+00:00,447552000000scheduledupdates, manifestSHAa875c0c9e5e30827ef2fe648d81d76f5a8d4e3f4e52f11fb94e027efdb8cc4a8,logSHA0f08d95367e9b836fd37c9153a6bc7c2ba6383b0cf732c23d1d51a5ba761a51e. Automaticpublicauditstarts2026-10-28T00:39:55.667801+00:00,wrapperPID3877303,actualauditor3877304/start46262254confirmedlive. Newprobability_cohort_review/production.py source1f755866f36d2487cd272af5fbef8dc234862f9872a95a2e9e8782612db331b3archivedbeforeexecution, calledonlyaftercomplete/exit0; all300casehashes/coverage/schemas/finiteness/clippingmetadata pass, no labels/taskmetrics/newdraws. Rootreviewat2publiccases; actual3at00:41:06UTC. Report/reviewseal/STATUSwithverbatimpriorversion/evidencemap updated. Whole publicreplay/control/statisticalqualitygates stillpending; globalindexstale/fullgoalactive. Originalsources/evidenceunaltered.


2026-10-28T00:45:52.512427+00:00 — verifiedwaitafter300productionclosureprogress: actualfine-taskparent3851463/start45624181andauditor3877304/start46262254live. Publicreplay12to17/300,802102500counterentries exact,allfinallogits/code/counterreceiptsmatch. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T004552_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullpublic/statisticalclosurepending, goalactive/globalindexstale.


2026-10-28T00:49:03.377139+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andauditor3877304/start46262254live. Publicreplay21to25/300,1179562500counterentries exact,allfinal-logit/code/counterreceiptsmatch. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T004903_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullpublic/statisticalclosurepending, goalactive/globalindexstale.


2026-10-28T00:51:52.750960+00:00 — verifiedwait: actualfine-taskparent3851463/start45624181andauditor3877304/start46262254live. Publicreplay27to31/300,1462657500counterentries exact,allfinal-logit/code/counterreceiptsmatch. Source/capacity219checks pass00:50UTC,1361155904Bconservativeoutputmargin. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T005152_verified_wait.json. No source/data/restarts/qualityclaimchanges; fullpublic/statisticalclosurepending, goalactive/globalindexstale.


## 2026-10-28T00:57:59.550727+00:00 — closed-cohort scientific resource figure

Previousgoalturnverifiedwait31finepubliccases. Currentturnconfirmsauditor3877304/start46262254live,36casesat00:53:27UTC. NoexistingCNNPNG/SVGfound. .venvhasnoMatplotlib; existingCPU .venv-thrmlhasMatplotlib3.11.2/NumPy2.5.4, no packagesinstalled/changed. Newmatched_figures/{plot,verify}.py archivedbeforeexecution, protocolsealedbeforeplot. Usesonlyclosed originalsingle/word statistics/resourceledgers, all3budgets/bothmethods, nofinequalitydata. ExportsstandaloneSVG/PNG/CSV/arrays, fixedaxes/alloutcomes anddescriptivepairedmodel/withinmodelseedintervals. Separateunit-accumulationchecker passesall30000absolute bootstrapaccuracyvalues/6points/intervals/CSVfields/workidentities; no newMCdraws. PNGvisuallyinspectedreadable/notcropped. Report/reviewseal/statushistoryandevidencemapupdated, captionclarifiesreferencearithmetic,notlatency/energy/fullsystemsPareto/equivalence. Actualfinepubliccases50atnewstatussnapshot; fulltaskqualitypending. Goalactive/globalindexstale.


## 2026-10-28T01:08:55.226482+00:00 — learned-control parameter mechanism diagnostic closed

Previous goal turn classified verified wait: live public replay observed74cases. New separate study/verifier sources archived before execution, protocol sealed before analysis. All five original and ten learned checkpoints retained, no images/labels/task scores/new native draws. All39,960 weight conditions,45layer summaries,15model summaries match scalar two-outcome variance/covariance derivative and independent code/entropy checks. Mean paired-model nominal variance reductions4.5750038544% expected-weight and5.8815615839% sampled-weight; endpoint-code fraction falls while broader extreme-probability fraction slightly rises. No task quality, sample-budget reduction or physical benefit inferred. All completion/source/input hashes checked, report and review closure sealed. Capacity219sourcechecks pass,1,244,632,896B residual after remaining output bounds at01:08:29UTC. Current live auditor3877304/start46262254 and parent3851463/start45624181 verified, public80/300 and3774600000counter entries exact. STATUS and requirements prior bytes archived; stale original-CNN audit wording corrected. Full fine-tuned quality gates pending; goal active/global index stale.


## 2026-10-28T01:19:06.755932+00:00 — full frozen-CNN Gaussian negative control complete

Previous goal turn progress (39960-weight mechanism diagnostic). New separate gaussian_control sources archived before full300case protocol freeze; all5models/2centers/3budgets/10MCseeds/full10000consumedimages retained. Nominal variance4r^2p(1-p)/T, originalFP32 versus programmed-mean centers, common normal draws across modes, full bank/spatial kernel sharing, deterministicACT8 and original fixed metadata/biases. All60validation paths and13320scalar variances pass. CPU pipeline3886457/start46457120 executes all300cases then full independentNumPy contractions: all30millionlogits pass, maxabs5.684341886080802e-14/allpredictions exact, terminalexit0. Separate stats source/protocol frozen before scoring. All3million scalar perexample losses and150000bootstrap scalar values across15contrasts pass. T1024 native83.0638% vsGaussianoriginal83.1428%/programmed83.0118%; allsixGaussian-vs-native accuracy/CE intervals includezero, not equivalence. No nativecalls added, no physicalspeed/energy claim. Separate root review all300casehashes/schemas, input/source/completion/loghashes and report sealed. Original/fine-trained native artifacts unchanged. Source/capacity219checks pass01:18:39UTC,870274880B residual. Current fine-task auditor3877304/start46262254 and parent3851463/start45624181 live: 109/300cases,5142892500counters exact. STATUS/requirements prior bytes archived and updated; broader goal active/globalindex stale.


## 2026-10-28T01:29:40.819716+00:00 — exact-budget CNN validation allocation and mixed-count native witness

Previous goal turn progress: full300Gaussian negative controls closed. New source/protocol frozen before validation-label scoring. All150isolated-layer Gaussian cases plus5nominal-mean baselines on all5000training-validation images complete; all7750000logits/775000scalar losses checked independently. All5separate schedules solve declared flooredinverseT surrogate over20009feasible integer schedules, each681984observed nodeupdates/bank equal uniform256. Scalar exhaustive enumeration independently verifies coefficient/global optimum/tie-break. Frozen schedules(312,195,302),(220,162,333),(260,157,335),(292,195,303),(240,177,320); no task quality inferred from surrogate. Separate source-frozen officialTHRML CPU witness all5models/3small mixed-count schedules,15cases/566190counters/210arrays exact public replay,114120retained-sample updates plus39960warmup perpass, production/replay both charged. New full100case native protocol includes equal observedupdates, separately chargedwarmup/schedulemetadata, full20caseGPUpreflight, publicaudits and declared3statistical contrasts; no new test scoring yet. Sources archived before execution. QueuePID3893677/start46547427 confirmedlive waiting on actual fine-pipeline parent3851463/start45624181; old auditor3877304/start46262254 live, now138/300cases/6511185000counters exact. Allselection/witness/reportcompletion hashes and155case schemas reviewed; no originalfrozen scientificsource/artifact altered. Combined forecast186000000Bremaining/residual463355264B after legacyhelper219source checks. STATUS/requirement previousbytes archived, log/statuslinks updated. Fullnativequality/stats and broad goal remain open; globalindexstale whilelive/queuedwriters.


## 2026-10-28T01:40:38.861687+00:00 — prospective allocated-native statistics and safe capacity recovery

Previous goalturn progress: fullCNNvalidation allocation/CPUwitness closed andnativecomparisonqueued. New allocated_statistics sources archived/source-frozen before newtestmetrics. Hierarchical bootstrap uses5model slots andindependent10seed-resampling arrays percondition; all3contrasts retained. Interactionfixture verifies30000vectorized-vs-unit scalarvalues plus600fullyscalarvalues. Actual watcher3897515/start46583127 waiting onnativeparent3893677/start46547427; bothconfirmedlive. Future actual1050000scalar losses/30000bootstrap values andresourcechecks required, no actualallocatednativequalityclaim. Diskfree fell while priorauditreadonly. Read-only inventory identifies619018318logicalbytes duplicated butstrict all-hardlinks-owned/modeguards select465801216allocatedbytes: newfrozen dedupsource47d88c25bc193b1fb67ea843d673744c273bf59faab472a8a6b3225c89db6883 inventoryfsynced beforeatomic hardlink replacements;56groups/105paths preserveallfilecontenthashes, rootreviewallchecks. Inode/ctime/sharedmtimechangesexplicitlyinventoried; notmodelcompression. Separatepipcachederivativesourcedcd74fe2713a74776dbb617964abd4d2d7f626bf6c489adfcf6ca84722d530ec verifies7installedMETADATA/largestpayloads thenremovesonlymatchedZIP/HTTPpairs306210154B; allruntimepayload/metadata hashesunchanged. No source/runtime/scientificcontent edits. Legacycapacity219sourcechecks pass; combined186000000Bfuturebound/residual1095486848B. Currentfineparent3851463/start45624181 andauditor3877304/start46262254 live,169/300cases/7973842500countersexact. STATUS/requirements priorbytes archivedandupdated; fullgoalactive/globalindexstale.


## 2026-10-28T01:43:34.532502+00:00 — combined queue audit and verified native replay wait

Previous goal turn progress: prospective statistics/fixture and capacity recovery. New read-only allocated_audit helper c8b64c8c32142689571f9cb83b006381ffb1eb857594d33badabf70db163aaf2 archived before execution. Keeps original helperf6467e6d153c7ac4156321ddb2bebd072865fdb72735398774ce124822bd8d89 unchanged, adds actualstarttick/cmdline/terminalreceipt checks for both allocation queues,58native+10statistics source/input checks beyondlegacy219sourcechecks, and dynamic preflight/case/analysis forecast≤175Mextra. Runs pass; combined186000000Bremaining/residual1030532480B. Verified45swait confirms oldaudit progressing170→175 then176/300 and8304120000countersexact; allfourhandlesremainlive. No restarts/draws/scientificsource orresultchanges, no newtaskqualityclaim. STATUS previousbytes archived; globalindexstale/fullgoalactive.


2026-10-28T01:46:04.672554+00:00 — verifiedwait: precedinggoalturnprogress(combinedcapacityauditor). Actualfineparent3851463/start45624181,auditor3877304/start46262254 andbothnative/statisticsqueueparents3893677/start46547427,3897515/start46583127 remainlive. Two45swaits, finepublic176→179→182/300, 8587215000countersexact. Combinedsource/input/capacitycheck passes; residual994106752B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T014604Z_verified_wait.json. No restarts/newdraws/source/data/qualityclaim changes; fulltaskstatistics pending, goalactive/globalindexstale.


2026-10-28T01:48:08.708053+00:00 — verifiedwait: finepublic183→186→189/300, 8917492500countersexact. Actualauditor3877304/start46262254,parent3851463/start45624181 andqueuednative/statisticsparents3893677/start46547427,3897515/start46583127 confirmedlive. Combinedfrozen-source/input/capacity checks pass,residual941202816B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T014808Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaim; goalactive/globalindexstale.


2026-10-28T01:50:53.188735+00:00 — verifiedwait: finepublic190→195→197→198/300,9342135000countersexact. Auditor3877304/start46262254,parent3851463/start45624181 andbothqueuednative/statisticsparents3893677/start46547427,3897515/start46583127 confirmedlive. Combinedsource/input/capacitycheckpasses,residual873512320B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T015053Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaim; goalactive/globalindexstale.


2026-10-28T01:52:51.228663+00:00 — verifiedwait: finepublic198→201→203/300,9578047500countersexact. Actualauditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual837610880B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T015251Z_verified_wait.json. No restarts/newdraws/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T01:54:53.068353+00:00 — verifiedwait: finepublic203→205→207/300,9766777500countersexact. Actualauditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual797203840B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T015453Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T01:56:52.957199+00:00 — verifiedwait: finepublic208→210→213/300,10049872500countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual850828672B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T015652Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T01:58:52.700094+00:00 — verifiedwait: finepublic214→217→220/300,10380150000countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual760040832B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T015852Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:00:55.465771+00:00 — verifiedwait: finepublic221→223→226/300,10663245000countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual739466624B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T020055Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:02:54.147353+00:00 — verifiedwait: finepublic227→229→232/300,10946340000countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual717806976B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T020254Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:04:59.264760+00:00 — verifiedwait: finepublic232→234→236/300,11135070000countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual724286848B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T020459Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:07:05.708810+00:00 — verifiedwait: finepublic237→239→241/300,11370982500countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual722013568B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T020705Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:09:05.488303+00:00 — verifiedwait: finepublic242→245→248/300,11701260000countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual721481088B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T020905Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:11:08.780705+00:00 — verifiedwait: finepublic249→251→254/300,11984355000countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual721452416B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T021108Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:13:11.914342+00:00 — verifiedwait: finepublic255→257→260/300,12267450000countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual720002432B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T021311Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:15:16.885709+00:00 — verifiedwait: finepublic261→262→265/300,12503362500countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual718728576B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T021516Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:17:19.086955+00:00 — verifiedwait: finepublic265→267→269/300,12692092500countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual717753728B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T021719Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:19:20.609443+00:00 — verifiedwait: finepublic269→272→275/300,12975187500countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual717114752B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T021920Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:21:21.359538+00:00 — verifiedwait: finepublic276→279→281/300,13258282500countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual716463488B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T022121Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


2026-10-28T02:23:26.620181+00:00 — verifiedwait: finepublic282→285→288/300,13588560000countersexact. Auditor3877304/start46262254,parent3851463/start45624181,bothqueuedparents3893677/start46547427,3897515/start46583127live. Combinedsource/input/capacitycheckpasses,residual715255168B. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T022326Z_verified_wait.json. No restart/newdraw/scientificchanges/qualityclaims; goalactive/globalindexstale.


## 2026-10-28T02:32:08.920107+00:00 — full learned-probability native CNN confirmation complete

Previousgoalturnverifiedwait288cases. Currentpublic288→293→296→298→300; all14154750000counters/30millionlogits exact, publicauditexit0. All10full learnedmean controls/all300metrics/all330000bootstrapscalarvalues pass; originalpipeline3851463terminalsuccessful at2026-10-28T02:28:44.538261UTC. Frozencompletion/report sealed. Separate rootreview7f59456517ffc8e1ad6ce130f11748e0731758b9f6ff4061e072256f3105cf44 archivedbeforeexecution; allcasehashes/schemas/coverage/protocolinput/source/completion/loghashes pass. Allsixaccuracies: expected72.5532/82.1514/84.9786%; sampled79.7850/84.0760/85.0262%; FP3285.0860%. T1024sampled−FP32−0.0598pp, descriptiveinterval[−0.419475,+0.289405]pp; T256−1.0100pp,[−1.350040,−0.688180]. Extra training andconsumeddata limitations retained; frozenbaselinefailure unchanged, no physicaladvantage claim. Warmupseparately999000000/fullfinepass, originalsingle499500000/word199800000; retainedsamplefigures unchanged. Existingqueue3893677automaticallystartsGPUafterverifiedparentclosure:20preflightcases/1509840countersexact, nativeproductionstarts02:29:41UTC, now12/100cases; actualparents/producerhandlesverified, statisticswatch3897515stillwaiting. Sharedfreejump~1.2TB observedexternally, notattributedtohousekeeping; combinedremaining138560000B. STATUS/requirements oldbytesarchivedandupdated. Fullgoalactive/globalindexstale whileallocationwriterslive.


## 2026-10-28T02:37:51.504127+00:00 — complete fine-tuning resource/quality figure

Previousgoalturnprogress: all300fine-nativecases/statistics/rootreviewclosed. Current new figure sources archived before prepare/run. All4methods×3budgets andbothaccuracy/CE plotted, fresh unit-accumulation verifier checksall120000bootstrap scalar values/12points/references/CSVfields/retained+warmupcounts andaxisendpoints. PNGvisuallyinspectedlegible/no crop. SVG/PNG/CSV/report andreviewcompletion sealed; no newmodeldraws/qualityreplicates. ExtraSGD,consumeddata,controlpayload,warmup,andphysicalcostlimitations explicit. Sourceenvironments unchanged. Nativeallocation producer3924405/start46920802,parent3893677/start46547427 andstatswatch3897515/start46583127 remainlive, now42/100cases,35804160000observedupdates. Combinedcapacity residual1203907383520B. STATUS consolidated tocurrentevidence withpreviousbytes archived verbatim/requirementspriorbytespreserved, avoiding obsolete pending-fine language. Fullgoalactive/globalindexstale whileallocationwriterslive.


2026-10-28T02:40:05.943341+00:00 — verifiedwait followingfigureprogress: matched-budget nativeproduction44→49→54/100,46033920000retained-sample updates. Actualproducer3924405/start46920802,parent3893677/start46547427,statswatch3897515/start46583127confirmedlive. Source/input/capacitychecks pass. Full publicreplay/statistics stillpending; no scores/restarts/sourcechanges. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T024005Z_verified_wait.json; goalactive/globalindexstale.


2026-10-28T02:42:18.246954+00:00 — verifiedwait: matched-budget nativeproduction55→60→66/100,56263680000retained-sample updates. Producer3924405/start46920802,parent3893677/start46547427,statswatch3897515/start46583127confirmedlive. Source/input/capacitychecks pass. Fullpublicreplay/statistics pending; no scores/restarts/sourcechanges. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T024218Z_verified_wait.json; goalactive/globalindexstale.


2026-10-28T02:44:26.099139+00:00 — verifiedwait: matched-budget nativeproduction67→72→78/100,66493440000retained-sample updates. Producer3924405/start46920802,parent3893677/start46547427,statswatch3897515/start46583127confirmedlive. Source/input/capacitychecks pass. Fullpublicreplay/statistics pending; no scores/restarts/sourcechanges. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T024426Z_verified_wait.json; goalactive/globalindexstale.


2026-10-28T02:46:33.560496+00:00 — verifiedwait: matched-budget nativeproduction79→84→89/100,75870720000retained-sample updates. Producer3924405/start46920802,parent3893677/start46547427,statswatch3897515/start46583127confirmedlive. Source/input/capacitychecks pass. Fullpublicreplay/statistics pending; no scores/restarts/sourcechanges. Checkpoint results/thrml_spw/fashion_cnn_native/execution_checks/20261028T024633Z_verified_wait.json; goalactive/globalindexstale.


## 2026-10-28T02:51:01.527753+00:00 — all100matched-budget native cases produced, public replay begins

Previousgoalturnverifiedwait89cases. Currentproducer90→96→100, successfulstageexit0at2026-10-28T02:48:31.071212UTC,85248000000retained updates/4718250000transient counterentries. Automaticpublicauditstarts02:48:31.120782UTC onwrapper3934489/actual3934490/start47033798. Newproductionreviewsource4f5a95c7eaea3a563ac372781c8c0933b8c68406b849c2ff8d29bd9ef75900baarchivedbeforeexecution; all100casehashes/schemas/finiteness/coverage/exactbudgets andsource/input/loghashes pass. No labels/taskmetrics/newdraws inreview. Report/closuresealed; STATUS/requirements oldbytesarchivedandupdated. Currentpublic13/100,613372500countersexact; actualauditor,parent3893677/start46547427,statswatch3897515/start46583127live. Fullpublic/statisticalqualitygatespending, goalactive/globalindexstale.


## 2026-10-28T03:06:36.135076+00:00 — five-model compact RGB training and native witness closed

Previousgoalturnprogress(all100allocatednativeproductioncases reviewed/publicstarted). ExistingCIFAR data available, oldsourcepilotlargerarchitecture/singleseed nottreatedasnativeconfirmation. Newcontrolled16x16RGB8/16conv/head256 CNN3928weights34biases,sourcearchived beforefreeze, fixed45k/5ksplit/5independentseeds/20epochsAdam/conventionalCPU. First64preprocessing49152coords exactseparateNumPy/Torch. All5fits complete80.77s,17600optimizersteps, selectionvalidationonly/no testscoring. Fullseparateauditall250000selectedvalidation logits/100orders/exports/splits/initializations/ranges passes, selectedepochs20/20/18/20/20,valaccuracy56.28/56.58/56.26/56.62/55.10%; moderatebaselineexplicit/differentfromoldpilot. Newnativewitnesssource5e0d744242b72957678cb9d9dd8a86717f2bf35b2567b33324acb2c450656c60 archivedbeforeexecution; all5selectedcheckpoints/first16valinputs/T1,4,16,15cases1477980counterentries420canonicalarrays exactCPUofficialTHRMLpublicreplay;824880retained+117840warmupupdates perproduction/replay pass. Newreviewsourcebd9db06412fec325c53bb49927235b88eabbee4ebe8f13a9450699ae9bd08734 closedallsource/input/completion/loghashes/checkpointshapes/report. NativefullbudgetGPU/fulltest/statistics stillrequired, no quality/systemclaim. Existingallocationauditor3934490/start47033798,parent3893677/start46547427,statswatch3897515/start46583127remainlive, public95/100,4482337500counters exact. STATUS/requirementsoldbytesarchivedandupdated. Fullgoalactive/globalindexstale whileallocationwriterslive.


## 2026-10-28T03:15:20.000183+00:00 — full matched-budget native allocation statistically closed

This goalturn also closes all5compactRGBtraining/nativewitness asrecordedabove. Allocationpublic100/100 and4718250000counters/10millionfinallogits exact, terminalnative03:07:32.712239UTC. Statistics/independentverification terminal0at03:08:05.929818UTC: all100metrics/1050000scalar losses/30000bootstrapvalues checked. Newrootreviewsource7aa01037e8a1ebcd063c19e5e03a7fc47b5c1b07b539020fab7f482c702abf4d archivedbeforeexecution; source/input/completion/log/casehash/schema/coverage checks pass. Uniform78.5270%/CE.6122843706→allocated78.8002%/.5987616873; delta+.2732pp, descriptiveinterval[+.04999,+.51601]pp, CE−.0135226833[−.0215033757,−.0054202571]. Exactequalobserved/warmup update budgets andschedulemetadata counted, still6.2858ppbelowFP32; no physical/jointtraining claim. No live scientific moduleargv andallknownpipelines terminalsuccessful. STATUS and requirementsoldbytes archivedandcurrentevidenceconsolidated. Globalinventoryrefresh followedbyindependentfullindexhashcheck nowplanned, receiptresults/evidence_inventory_verification_20261028T031520Z.json outsideindexedroots toavoidselfreference. No furtherindexedwrites untilverification; broadgoal remains active.


## 2026-10-28T04:52:53.519979+00:00 — full150case compactRGB native execution started

Previousgoalturnprogress: CIFARtraining/nativeCPUwitness andfull100allocationclosure/inventory verification. NewRGBcohort6modules archived beforeexecution,50validationcontrol paths pass, finalbackendgitpin checksource archived beforeprotocolfreeze. Fresh1737520nativekeys/1747520bootstrapseed, all5models/T64,256,1024/10MCseeds/full10000consumedtestimages, all10detcontrols/30contrasts predeclared; no favorable selection. Frozenprotocol2b3ff56fe6f3c5e174688d6b5b5be4728d7e8b91ec4d75b816e0170419f65a27. Actualpipeline3964689/start47733621 launches04:45:09UTC, all15full-budgetGPUpreflightcases/1477980counters/420arrays exactpublicreplay;52792320retainedupdates pernative/publicpreflight plus117840warmup. All50fullcontrols saved; productionstarts04:45:34.457282UTC, actualproducer3965140/start47736133 live, now28/150cases/55934720000retainedupdates. Wholepassplanned329952000000retained+736500000warmup, countertotal9237375000, no actualnativequalityreleased. Separateprerequisitereview32544b732792a305b116ed42cfddecbc9a46912919dfecffac0aa9255ecdbb9c andcapacityhelper28b8f9237978288d597bcdbac5c768a3a4dfa66ad892bd40c79d9c18ea186f7e archivedbeforeexec; source43/input22checks pass. STATUS/requirements priorbytesarchived/oldinventory nowexplicitlyhistoricalstale due newRGBwriters. Originalfrozen sources/results unmodified; fullgoalactive.

## 2026-10-28T04:57:26.147288+00:00 — Partial RGB production integrity review

Previous goal turn: verified wait (live parent3964689/start47733621 and producer3965140/start47736133). Current turn: progress. Added a separate read-only integrity checker without changing frozen scientific sources, protocol, or completion-bound evidence. Archived source before execution: `96ba25556377a9323bad9d557e7a2b447e184123d2438742d6806899d454570a`.

Checked all48atomically recorded production cases: source/input hashes, case SHA256s, exact expected prefix coverage,1250x14x32uint8 receipt arrays,10000x10F64 finite logits, int64 clipping counts/ranges, and cumulative retained-update/counter accounting. Actual native preflight array dimensions independently establish49266int32 counter entries per8-imagebank; positive/negative convolution gates are separately retained. Both exact process identities confirmed live. Record: `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T045706Z_production_integrity.json`. No partial task metrics computed; this does not substitute for public native replay or statistical/control verification. No jobs restarted. Full150case pipeline continues and the global inventory remains historical while writers are live.

## 2026-10-28T04:58:51.414623+00:00 — Synthetic prospective RGB statistics check

Previous goal turn: progress (new48case file-integrity audit). Current turn: progress. The live frozen150case RGB pipeline continues unchanged. Before any cohort quality release, executed the unchanged frozen statistical entry point in a separate clearly labeled synthetic namespace, with asymmetric20example/10class generated logits for150case outcomes and50controls. No native calls, real task metrics, replay acceptance or model selection occurred. Synthetic passed-audit flags and full counter totals are explicit fixture placeholders only, not native receipts.

Archived checker source before execution: `89a96dca02fb10636c30132128755ecba9d21e7f4473f9fd4b14230207826ab8`. Independent per-example Python/math metrics, independent index generation, and full scalar enumeration of300000bootstrap values across all30paired contrasts matched the production computation, point differences and descriptive interval endpoints. Fixture subprocess exited0; all completion-bound fixture hashes independently reread. Source production guard remained active; only its output directory was redirected for this standalone process. All derivative evidence resides in `results/thrml_spw/cifar_cnn_native/statistics_fixture_evidence/`; `fixture_completion.json` explicitly declares fixture_only/native_calls0/task_quality_claimfalse. This supports implementation correctness only; actual full public replay, raw-data/control and statistical verification remain mandatory.

Follow-up integrity check passed all53recorded native cases with both exact process identities live. The reused checker carries its initial static previous-turn classification; this continuation and its predecessor are correctly classified as progress in `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T045852Z_classification_clarification.json`. No scientific receipts were rewritten.

## 2026-10-28T05:00:59.676663+00:00 — RGB internal four-node witness closed

Previous/current goal turns: progress. Added source-frozen validation-only CPU witness using existing unchanged generic four-node native operator and separate public-state recovery. Checker source3e25d988718c93e033a65896d984440c5cccd7705b887d20fcf0be331f76be2c and rootreview sourced341424c6ad6d1dda226ff22c6b1d71cdbd5533ae346dc5d750ea0582b5b3b8d archived before execution. Witness exited0. All15cases across fiveRGBmodels/T1,4,16/two8imagebanks matched1477980integer counters/420canonical bank arrays exactly. Native/publiceach3299520retained plus471360warmup node updates. All case/source/input hashes reviewed. No task scoring and noGPUwork added to live single-node execution.

Prospective equal-retained-update comparison pairs singleT64/256/1024 with wordT16/64/256:329952000000updates perfullpass, but word fullwarmup2946000000vs736500000single. Word logicalpayload2692Bvs4656Bsingle; dynamicstate1964vs491B separately. These are software resource counts, not physical area/energy/latency. Full word test execution remains open. Actualsingle-node producer and parent identities verifiedlive with59/150cases. New reportCIFAR_CNN_WORD_WITNESS_REPORT.md and documenthistory/change receipts recorded in `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T050059Z_word_witness_progress.json`. No frozen existing source or completion-bound content changed.

## 2026-10-28T05:06:03.256630+00:00 — Full150case RGB matched-word comparison queued

Previous/current turns: progress. Added six separately archived/source-frozen word_cohort modules; derivative statistics lineage is the unchanged Fashion word comparison with newnamespace and stronger coverage/public receipt checks. Nominalmean selfcheck exited0 for allfiveRGBmodels/19640weightcoordinates/3200validationlogits. New synthetic-only statistics fixture a514dbbc439e09376245fa8267a12594d86ab510f6419d1602d0c5d0d15ce9f6 independently enumerated120000bootstrap scalars across12contrasts and exited0; dummy native flags/accounting remain clearly labeled synthetic and outside scientific evidence. No real word test scoring.

ProtocolSHA21af9a3234c78bfb61a18503644f837958967af7d9c56d6fee99cb3580c0f545sealed before wordtest execution; newkey1777520/wordbootstrap1787520. QueuePID3967754/start47852613confirmed live waitingforparentfullaudits. Stage sequence: all15fullbudgetGPUvalidation,controls,150nativecases,fullpublicreplay,stats,independentTorch/metric/bootstrap/resources. No concurrentGPUwork or restartedjobs. An immediate first observation ran before pipeline.json creation and raisedFileNotFoundError; the same livePID was rechecked, not relaunched, and its atomic waitingstate appeared successfully. This was an observation race, not a scientific failure.

Single-node actualproducer andparent stilllive at82/150cases. Source/input/fixtureseals independently checked and preparationreport/reviewsealed. Conservative660MBcombined remainingoutputs clear1203462492160Bfree; oldcapacityhelperexcludesword and currentreceiptsreserve330MBextra explicitly. Allmethods/budgets/negativeoutcomes retained; no actualwordquality or physicalsystems claim.

## 2026-10-28T05:10:40.044407+00:00 — Independent RGB scalar moment comparison and combined capacity

Previous/currentgoalturns: progress. Newread-only combinedcapacityhelper a1942de1e30163ba71e7c743e9197fa3bb87a95b25436470e17c8f9d1d5c0b60 archivedbeforeexecution; it validates wordqueue/source/inputseals and preciseactualGPUmoduleargv, includes330MBwordfutureallowance plus earlierremainingbound. At05:07UTC combined443550000Bfits1203457871872Bfree, exactlyoneactualGPUproducer3965140. A45swait then confirmed allthreeexactprocessidentitieslive; no restart or concurrentGPUwork.

Newlabel-free scalaranalysis sourceea3ad7fbdee635e731ee07242e94b29e3f618e7035cb34dadb098880f3fa54cd and independentcheckersource1a76ace744f90b52a864408a7a5e9f88f086911e03cf00b36c936aeee95887f3 archivedbeforeexecution. Both exited0. Actualfixedcheckpointweights and programmedF32fields determine nominalF64sigmoid biases/variances; independent scalarweightencoding and exhaustive2/16outcome laws checked all19640conditions and15model/budget summaries. No labels ornativecalls. Allbudgetresults retained inCIFAR_CNN_REPRESENTATION_MOMENTS_REPORT.md; nominalMSE favorswordat64/16and256/64, singleat1024/256. This neither selects a new budget/gate nor predicts nonlinear taskquality/equivalence/sample-saving/physicaladvantage. ActualF32samplerempiricalmoments/correlations are distinct.

Allcompletionhashes reread; documenthistory/change receipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T051040Z_representation_moments_progress.json`. Existingfrozen scientificsources/results unchanged. Single-nativeproduction100/150andwordqueuecontinue; fullnative/public/control/statisticalqualityclosure remainsrequired.

## 2026-10-28T05:25:09.770664+00:00 — All150compact RGB native production cases complete

Previousgoalturn: verifiedwait. Currentturn: progress. Monitoredsameproducer3965140/start47736133 throughsuccess, withoutrestart/conditionchanges. Productionstagewrapper3965139 exited0 at2026-10-28T05:23:59.765000+00:00. Fullmanifestcomplete150cases/329952000000retainedupdates/9237375000scheduledcounterentries. Warmup736500000updates separatelycharged. Frozenproductionreviewsource6c2571c2b27740927dc3b606a1997010bfb8f79b150bd7d11ef41f5e4821c79d archivedbeforeexecution; reviewexited0aftercheckingall150casehashes/schema/15millionfinitefinal-logitvalues/exactexpectedcoverage/source-inputseals/successfulloghash. No taskmetricscomputed orcountercorrectnessclaimbeforepublicreplay.

ActualindependentGPUpublicreplay3969544/start47966673 nowlive behindwrapper3969543/start47966671; parent3964689/start47733621stilllive. Publicprogress6/150atthissnapshot. Wordqueue3967754/start47852613stilllivewaitingforfullsingleaudits. Combinedcapacityvalidatesprotocolsources/inputsandonlyoneactualGPUmodule;354500000Bremainingboundfits~1.2TBfree. No physicalenergy/latencyoraccuracyclaim. Status/requirementsversionsarchived and phasechange recordedin `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T052509Z_production_complete.json`. Fullqualityrelease awaits allpublic/control/statisticalchecks.

## 2026-10-28T05:28:09.744725+00:00 — RGB FP16/BF16 conventional controls generated

Previous/currentgoalturns: progress. Added separatelysourcearchived precision_baselines/{common,run,verify,watch}.py as derivatives of unchangedFashionprecisionmodules; actualCifarCNN16class/newpaths/parentPID/start/successstatus/payload/report corrected, exact12contrastcoverage added. Allsources archivedbeforeexecuting selfcheck. Validation20paths/64images each passed independentFP16/BF16weight-biasrounding plus separateNumPy/functionalTorchgraph checks; exited0. SourceSHA/common adfa14b7f151645590598a08460fa763fe61be67079491d9e7bc9b3338ff1209;run ddc2f3ffba9d2c1e8dee29b1d33aa180a20dd809a085b8ce74e62861b22e58d2;verify d47ab7699958a3606091e23ddf096c2a8907382cbdd67147faf4f17e4821bb40;watch f521a0151140a5bc8d754efbf92fb22e0c9ff75c2093870e94d21df5964f925b.

Newprecisionwatcher3970310/start47983435 launchedonce, source-sealed protocolSHA52894264a85ac9f673726c644a645c0abdc13c0ae66e24b850f9ddb04b7866cabeforefulltestcontrols. CPUevaluator exited0andall20fulloutputs/2millionfiniteF64logitvalues/hash/schema checked. No taskmetricsreleased. Actualwatcher waits for parentfullnative/public/control/statisticalclosure before its12contrasts/independentfullgraphverification. Newnativeupdates0, noGPUuse. Nominalweightpayload7856Bplus136or68biasbytes; internalaccumulatorwidth/physicalcostunclaimed. InitialprocessobservationconfirmedactualPIDduringstartupbeforepipeline.jsonappeared; noerror/restart.

Additional50MBfutureoutputguardclears~1.2TBfree with earlier354.5MBsingle/wordbound, and is explicitly separatebecauseexistingcombinedhelperexcludesprecision. Currentreport/progress/docversions preservedin `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T052809Z_precision_progress.json`. Frozenoriginalscientificsources/protocols/resultsunchanged. FullRGBtaskqualityawaits allnative/public/control/statisticalchecks.

## 2026-10-28T06:05:45.598802+00:00 — FullRGBnative/precision cohorts independently closed

Previousgoalturn: verifiedwait. Currentturn: progress. Samepublicreplay3969544/start47966673completedall150cases/9237375000counterentries/15millionlogitvaluesexact; parentfullstats/independentcontrolverificationallstagesexit0. All50controls/5millioncontrol-logits,7680000rawRGBinputcoords/labels,150casemetrics/300000bootstrap scalarvalues checked. No preexisting source/result/condition/gate changed. NativeT64/256/1024accuracy34.5380/47.5452/53.5644%vs56.2580%FP32; largestloss2.6936pp, descriptive95%lossinterval[2.304985,3.081805]pp, missesapprox1–2target. Lowerbudgetsretained. T1024exceedsminmaxINT4by1.8664pp butthisisoneof30contrasts/noadjustment/newsuccessgate, whileT64/T256areworse. No physicalsystemsclaim.

Precisionfull20controls/2millionlogits/120000bootstrap valuesalsoaudited; CPUlowprecisiongraphsreplayexactly, weightonlyF64separateNumPyerrors<2e-14. Meanaccuracies56.2400/56.2660/56.2400/56.2940%forFP16weight/BF16weight/FP16CPU/BF16CPU, no newnativedraws. Oldparent3964689/public3969544/precisionwatch3970310nowterminalsuccessful/absent; no restart. Newwordpipeline3967754/start47852613automaticallypassedall15fullbudgetGPUpreflights/1477980counters andsavednominalcontrols; actualwordproducer3977809/start48195137livewith20/150cases. Wordfullquality/public/control/statsstillpending.

Sourcefrozenrootreview07ab70d241aa281a04a5952f83312353eb6d52a125685874d012f1dfe5822c58archivedbeforeexecution; checksexited0aftercompletion/source/input/logstagehashes andall42contrast/420000bootstrap-valueunitchecks. ReportsCIFAR_CNN_NATIVE_REVIEW.mdandCIFAR_CNN_PRECISION_REVIEW.mdsealedinclosure_review/completion.json. Priorcompletion-boundreportsunchanged. ConsolidatedSTATUSnowleadswithactualclosedRGBresults, strongerearlierFashionresultandopenscope; rawpreviousSTATUS/requirementsbytesarchived. Progressreceipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T060545Z_cohort_closure_progress.json`. Combinedcapacity330MBremainingwordboundclears~1.2TBfree; precisionterminalfutureallowance0. Globalinventoryremainsstaleandmustnotrefreshuntilactivewordwritersclose. Goalremainsactive.

## 2026-10-28T06:13:54.042731+00:00 — RGB probability training/validation preparation complete

Previous/currentgoalturns: progress. BasedonverifiedfrozenRGBqualitytargetmiss, continuedoriginaltrainingcompatibilityroutewithoutchangingbaselineconditions/gates. Reusedunchangedshape-genericFashionProbabilityCNNviaRGBwrapper, preservingfixedranges/biases/ACT8spans and8bitfieldcommands; all5RGBfunctionalgraphs/30declaredsurrogategradientchecks/15exports passed, no nativecalls. Newselfcheck/modelsourcehashes d6e7b450f0ed7d43f5dac544f4d91a1165d73947a6258d48a400be91f1438fb9/127c074e83272952e1aef1c325b47ea54a0de5d48cbe79bb51ea4c69012c2769. Allsourcesarchivedbeforeexecution. Trainingderivative840761db8a424cbafdc017404bbe7ec57d526de257bf90e94a6334860211e7ef usesonlyrawCIFARtrainbatches andoriginal45000/5000split; verifiesallcachedvalidationinputs exactly. Pre-freezecomment-onlysourcebce496b302b023bce3ef1a6c40601a12a1c1a776197b7b0c599da3ca64cf7ad4archivedtoo. Independentvalidatora460893bc1b8c049fcefeef81c48fdde01e3546c132feb2280fe2b2c9ff4e204; pipeline18693832cbf1863d43c8b96ee8f6cfb3d9c0b9a7859eda79c06e31264061bf2e.

ProtocolSHA42c5616396b6b75671506ee648c34a2a45f6eb40423ebcb1e3d151c64cd32cf8frozenbeforetraining: fiveoriginalmodels/twofixedmethods/eightadditionalepochs,fieldlatentAdam0.01→0.001,originalfixedF64ranges/biases/F32castspans,poststepclampand8bitforwardSTE. Freshorder1817520/sample1827520/validation1837520roots; allmethods/epochsretained, selectionminvalidationCEoverfourfixedT256surrogatereplicas,epoch0eligible. No testexamples/labelsenteroptimization/selection, parentcompletionartifacthashesonlyprovenance. AdditionaltrainingconfoundscomparisonwithoriginalFP32; expectedmethodcontrolsparameterization/epochs/orders; surrogategradientsbiased/nativevalidationstillrequired.

Pipeline3979530/start48245931startedonce06:10:32UTC; trainingchild3979537thenindependentvalidator3979671bothcompleted0. All10fits/80orders/14080steps/fixedrange-bias-commandexports/2millionselectedvalidationlogitvaluescheckedexactly. MaxF32trainingprobabilityvsnominalF64lawdifference1.4869397613193058e-7retained. No independentfullretrainingornative/testclaims. Fullpreparationreport/completionseals/rootreview/documenthistoriesverified; progressreceipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T061354Z_probability_training_progress.json`. Wordpipeline/producerexactidentitieslivewith55/150cases; noGPUworkadded/restart/existingfrozencontentchanged. Training50MBfutureallowancecompleted0, globalinventoryremainsstaleuntilwordandanynewwritersclose. Nextrequiredstepislearned-commandnativecorrectnessbeforefulltaskcohort.

## 2026-10-28T06:18:16.939336+00:00 — All30trained RGB native correctness cases pass

Previous/currentgoalturns: progress. Sourcec97d9abec06bdd7385ecd4bad2a59d192d7bb071d82229a7ffdacf2f1aaafdea archivedbeforeCPUexecution; uses unchangedpureNumPylearnedadapter and separatepublicrecoveryfromFashionvalidatedmodules, samegenericallocatednativeoperator, correctRGBshape2x8x3x16x16. Newkey1847520/model/method/T/bank stream. All10trainedexports/threefullbudgetsT64,256,1024checkedon16validationinputs; no labels/taskscoring. Everyoriginalrange/biasandlatent-commandfield independentlyreconstructed; neverrefitminmaxtoexpectedweights.

Native/publicexact2955960integercounters/840bankarrays includingcodes/masks/intermediates. All3928weightnodesremainfree, noactivationpbits. EachCPU/nativepublicpass105584640retainedplus235680warmupupdates. OfficialTHRML0.1.4cleanpin/CPUJAXF32sampling andexternalF64readout, no physicalcostclaims. Execsession63687completed0, fullsavedcase/source/input/completionhashes/schema/costsreviewedbyarchivedrootreviewsource3af527486dcd38864876c116e7baf73ceff26e0cfb75ad0faf017782fdc09064; reviewexited0. ReportCIFAR_CNN_PROBABILITY_NATIVE_WITNESS_REPORT.mdsealed. FullGPUpreflight/fulltask/public/control/statisticalconfirmation remainrequired; originalfrozenRGBqualitytargetmiss/trainingconfound/biasedsurrogatesretained.

Wordpipeline3967754/start47852613andactualproducer3977809/start48195137livewith76/150nativecases, unchangedprotocol. NoGPUworkadded/restart/frozenexistingcontentchanged. Newnativelearnedfullcohortnotyetqueued; existingcapacityhelpercoverswordonlyuntilfuturecohortexplicitlyincluded. Documenthistories/rootreceipts in `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T061816Z_learned_native_progress.json`; globalinventoryremainshistorical.

## 2026-10-28T06:29:12.714476+00:00 — Full300case learned RGB cohort source-sealed/queued

Previous/currentgoalturns: progress. Added separatelyarchivedprobability_cohort modules derivedfromunchangedFashioncohort: common1d57a45868205d451feaad8058c67635a3dc802cc4ea3cebc33f4feef907a6c0,run919885a7b07737c871d136ab7165dab44c48dd8417958ffe90d0df6ef360fdea,verifya0e9ebc37fdd7beb217c1f2cec0f28befc5270a040d565a0970559932eca75bc,bootstrap_probe4b6c289eb2e134307f44baf5a898df5d07fb10da3c1eeb6f2f4022cb0cd82bae,pipelinefinal3f5d9b09005f42b5c6be310d99d874e824bf3735db1bf2deb747860fc420e0d2. Pre-freezepipelinefe92cb0a659498784060e17dad964215fd7832152fc2e2d768c88cde6a16a74barchivedtoo; finalversionrequiresstatisticalfixture. RGBshapes/costs/payload/fullGPUpreflight/receiptcoverage/33contrastcoverageandwarmup/trainingconfoundaccountingfixedbeforeexecution, originalsourcesunchanged.

All10meancontrols/64valinputs eachpassindependentNumPy/Torchgraphs; bootstrap1877520indicesmatchactualNumPy2.5.3/2.5.4environments. Newsynthetic-onlyfixture358a6d99530eb4fb3bed92f354a1bb5457931f9d447e9d470f28a4e6a7111ab6exited0afterindependentlyenumerating330000bootstrapscalars/all33contrasts/300caseand10meanmetrics. Fixtureexplicitlyretainsdummy nativeflags/accountinganddoesnotprovephysical/native/taskbehavior. Rootreviewhashesallprerequisitecompletionboundartifacts.

ProtocolSHAe89eaac772cf2baefae6507055efbf4a5a180a8bdcf22c84a56bf6d9de35c40bsealedbeforetrainedtestscoring. All5models/2methods/T64,256,1024/10MCseeds,5000pairedmodelunitbootstrapdraws/newmethodindices, all33comparisonsretained. Newnativekey1857520/GPUvalidation1867520/bootstrap1877520. Eachfullnative/publicpass659904000000retainedplus1473000000warmupupdates/18474750000counters. NewGPUpreflight30cases105584640retained+235680warmup/pass; no actualGPUlearnedworkyet. Training8extraepochsconfoundcomparedtooriginalFP32/frozen/word; original2.6936ppmissretained, no outcome-selectedgate/freshconfirmation/physicalwin.

Actualqueue3981233/start48338739launchedonce06:26UTCandconfirmedwaitingforwordfullaudits. StageorderGPUpreflight→controls→300native→fullpublic→stats→independentverification. Currentwordpipeline3967754/start47852613andproducer3977809/start48195137livewith119/150cases. Newcapacityhelper49cd66d6361e7d3b7c1a169e51202a01515c9a8148d911adfc3ed4652f7e9422archived/executed;68source/55inputhashesactualqueueandoneRGBGPUworkerpass;830MBcombinedboundfits~1.2TBfree. No restart/concurrentGPUwork/sourceororiginalresultmutation. Newreport/protocolreviewsealed; documenthistories/progressreceipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T062912Z_learned_cohort_queue_progress.json`. Globalinventoryremainshistoricaluntilallwritersclose.

## 2026-10-28T06:34:26.073291+00:00 — All10RGBprobabilityfits byte-reproduced

Previous/currentgoalturns: progress. Newarchivedsource1eb9de125f68bb873a9e80c3e75947dfa1a688dec6b9a1a8d306f4205c722f2breuses unchangedRGBtrainingmodule in freshCPUprocess, redirectingonlytrain.OUTtoisolatedfolder. ReproductionprotocolSHAb7eca64ac281058ad4a6b359069efcdea6caab31515147601a6abd2fdfc195eafrozenbeforeexecution, actualprocess3981675/start48368426launchedonce06:30:57UTC. Itcompletedallreproductionchecksby06:32:31UTC; completion/pipeline/logfinalsuccesspresent. No independentexitcodewrapperwasused, so acceptanceisboundtoactualcompletionchecks/terminalpipelineandlog, notanunobservedexitcoderesult. Originalsources/outputsneveroverwritten.

All10fits selectedepochs/CE and fullepoch-historybytes match exactly;240arrays/2158200values including2millionselectedvalidationlogits dtype/shape/rawbytesidentical. ZIPcontainerbytesmaydiffer. Original14080optimizersteps+additional14080=28160;0newTHRMLdraws/notesttaskscores. Same-code reproducibility, notindependentlyimplementedgradients/optimizer/newseeds/statisticalreplicas. Source/input/completionsealsandall10exactnessflagscheckedinrootreview/reportCIFAR_CNN_PROBABILITY_TRAINING_REPRODUCTION_REPORT.md. Extra20MBfutureallowancecompleted0; originalfrozen2.6936pointqualitymiss/trainingconfound/biasedsurrogatecaveatsretained.

Actualwordpipeline3967754/start47852613producer3977809/start48195137andnewlearnedqueue3981233/start48338739livewithword144/150cases. NoGPUworkadded/restart/conditionchanges. Documenthistories andprogressreceipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T063426Z_training_reproduction_progress.json`; globalinventoryremainshistoricaluntilactivewritersclose.

## 2026-10-28T06:37:38.592121+00:00 — All150matchedRGBwordnativecases produced

Previous/currentgoalturns: progress. Existingwordproducer3977809/start48195137finishedall150cases; wrapper3977808productionstageexit0at2026-10-28T06:36:04.176092+00:00. No restart/conditionchanges. Manifestcomplete/source-input/caseSHA256seals/exact5x3x10coverage/matchedsingleT=4wordT and329952000000retained-update/9237375000scheduled-counter totalschecked. Wordwarmup2946000000updatesperpassseparatefromsingle736500000; physicalflips/energy/latencyunknown. No partialwordtaskmetricscomputed/released.

Actualindependentpublicreplay3982087/start48399118nowlivebehindwrapper3982086; wordparent3967754/start47852613andnewlearnedqueue3981233/start48338739stilllive/correctlygated. Currentpublicprogress8/150. Combinedcapacityhelpercovers330MBword+500MBlearnedfutureallowancesandonlyoneactualRGBGPUmodule;~1.2TBfree. Allfullwordpublic/control/statisticalconfirmationremainrequired. Productionreview/docarchives/progressreceipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T063738Z_word_production_complete.json`; originalfrozencontentunchanged/globalinventoryremainshistorical.

## 2026-10-28T06:43:46.051230+00:00 — FrozenRGBsampling figure exported/verified

Previous/currentgoalturns: progress. Source5803bd37fbbdb0683fbdca956c77c5e0c8fba387a1a9579cc18c4a1c802aed6a and independentchecker2d3d5d623b321bf1a5af42a8a6f1cb8e0f7c24055d0cb6d959facf6644602027 archivedbeforeexecution. Bothcompleted0. GeneratedstandaloneSVG/PNG/CSVwithall3nativeTpointsandall14conventionalcontrolsretained; plotshowspreselectedFP32/minmaxINT4/INT8levels. Independentscalarenumerationchecksall170000bootstrapvalues,point/intervalunits/CSVcoverage. No nativecalls orneuralcasesadded. Nativepointsandcontrolabsoluteintervalsonsamefixedconsumedimages/parentbootstrapunitsnotpairedtests/generalization/equivalence. Samples/stochasticnodecountsnotphysicalenergy/latency, conventionalcontrolsnotzeroarithmeticcost. Frozen2.6936pointmissvisibleandretained.

PNGvisuallyinspectedviaimageviewer; bothpanels/labels/legend/footerlegible,nocrop. FP32/INT8linesnearlyoverlapasactualdatadictate, notaformattingfailure. Completionhashes/rendersourceunitsreviewsealed; reportCIFAR_CNN_FROZEN_SAMPLING_FIGURE.md providesvector/raster/CSVdownloads. Existingfrozencompletionboundreportsunchanged. Actualwordpublicreplay3982087/start48399118andparent3967754/start47852613/waitinglearnedqueue3981233/start48338739live, wordaudit37/150. No restart/conditionschanged. Documenthistories/progressreceipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T064346Z_frozen_figure_progress.json`; globalinventoryremainshistoricaluntilallwritersclose.

## 2026-10-28T07:00:56.723649+00:00 — NominalRGBstatechange versus scheduled-update diagnostic

Previousgoalturn: verifiedwait. Currentturn: progress. Newsourcea746effab81b9f922ad7d7b599fda19ed881388369e043868c6a487d3d16c4f8archivedbeforeexecution; programcompleted0. UsespreviouslyverifiedfrozenRGBprobabilitiesonly, nolabels/newnativecalls/conditions. Enumeratesfourprevious/currentstatepossibilitiesforall98200deviceconditions (5x3928single+5x15712word) tocheckiid2p(1-p)law, all15fixedmatchedmodel/budgetrows. RetainedTupdatesafterstationarywarmup, T-1betweenretainedsamples, andsimulatorinitialzero-resetwarmupactivityseparate.

AtmatchedretainedU, idealexpectedretainedsingle/wordstatechangesratio~627.237934forthesefrozenencodings, becauseinternalwordfieldsareallnearendpoints. This isnotmeasurednativeF32flipactivity,physicalswitches,energyfactororExtropicarchitectureclaim; especiallysoftwarecoldresetwarmupandretainedphysicalstatesdiffer. Deviceupdate/change/holding/program/readoutcostcoefficientsandadditivemodelneedhardwarevalidation. Nophysicalwinner/samplequalityclaim. AllrowsretainedinCIFAR_CNN_NOMINAL_ACTIVITY_REPORT.md/activity.json; existingfrozenquality/inferenceoutputsunchanged.

Completionhashes/15coverage/resourceunitsreviewedandsealed, documenthistories/progressreceipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T070056Z_nominal_activity_progress.json`. Actualwordauditor3982087/start48399118andparent3967754/start47852613/learnedwaiter3981233/start48338739live, wordpublic119/150. Combinedcapacity830MBallowanceclears~1.2TBfree, soleGPUwordaudit. No restarts/conditionschanged; globalinventoryremainshistoricaluntilallwritersclose.

## 2026-10-28T07:14:24.554630+00:00 — Wordfullqualitycrossoverclosed; learnedGPU300caseproductionstarted

Previous/currentgoalturns: progress. Samewordpublicauditor3982087/start48399118completedall150cases; subsequentstats/independentcontrolsallstageexit0andcompletionsealsverified. All9237375000counters/15millionlogits/150metrics/500000meancontrollogits/19640weightmeanlawcoords/120000bootstrapvalueschecked. WordaccuracyT16/64/256=51.6144/51.7096/51.7068%,versusmatchedsingleT64/256/1024=34.5380/47.5452/53.5644%. Single−wordaccuracydifferences−17.0764/−4.1644/+1.8576pp. Largestdescriptive95%interval[+0.027990,+3.681615]pp, CEbetterby0.0587596481nats. Allthreebudgets/12contrastsfixedbeforewordscoringandretained; no multiplicityadjustment/freshgen/equivalence/novelpostoutcomegate. BothfrozenrepresentationstargetmissFP3256.2580%. Boundedcrossovernotphysicalsystemswin;4timeslocalrounds/fewerweightnodes, command/metadatapayloadandexpectedactivitycostsseparate.

Sourcefrozenrootreviewc0ca7da331ff4cc9f152233885911b59c92a2c090ef63de134ac264cd904a86earchivedbeforeexecution; source/input/completion/stagelogseals/all12contrastdirections/unitschecked, reportCIFAR_CNN_MATCHED_WORD_REVIEW.mdsealed. Originalcompletion-boundreport/sourceunchanged. Formerwordparent3967754/auditor3982087terminalsuccessfulwithoutrestart.

Learnedqueue3981233/start48338739automaticallyadvancedonlyafterfullwordclosure. All30GPUfullbudgetvalidationcases/nativepublic2955960counters/840bankarrayspassed;105584640retained+235680warmupupdatesperGPUvalidationpass. TenfullF64nominalcontrolssavedwithfinite1millionlogitvalues; independentfullcontrolverificationlater. Actualproducer3986451/start48591250livewith26/300cases. Training8extraepochsconfound/originalfrozenmissretained, no trainedtestqualityrelease. NewGPUprereviewboundallsource/input/prereq/case/loghashes; currentmonitor68source/55inputchecks/oneactualGPUworkerpass, futureoutputbound500MBfits~1.2TBfree. No restartedjobs/newconditions/frozenexistingcontentchanges. Documentarchives/progressreceipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T071424Z_word_closure_learned_start.json`; globalinventoryremainshistoricaluntilactivewritersclose.

## 2026-10-28T07:19:51.357760+00:00 — MatchedRGBsampling figure exported/verified

Previous/currentgoalturns: progress. Newmatched_figuresderivativesarchivedbeforeexecution:plot4e0bb93ba74b6367063dfd3a39ca235ac91517130d532961875bb6a270985e29,verifye23efba86bb7c1e70463942eb5c263993149e34c0762c2da68e34b54c7baa52a. Prepare/run/independentverifycompleted0; sourceparentsealschecked. Allsixpredeterminedaccuracybudgetpointsand30000absolute bootstrapaccuracyvalues/CSVresourcefieldsverifiedwithseparatemodelunitsummation. SVG/PNGtworesourcesslices K*TandTpreservealllower-budgetlosses/crossover; FP32/INT4referencelevels andlimitationfooterlegible,nocropverifiedimageviewer. Absoluteintervalsnotpairedtests; pairedCI/preregisteredcontrasts/consumeddata/frozenqualitytargetmiss/nominalcontrolpayload/physicalcostcaveatsretained. No newnativecalls/qualityreplicas; trainedresultsexcluded.

ReportCIFAR_CNN_MATCHED_RESOURCE_FIGURE.md/vector/raster/CSVandrootreviewcompletionsealed. Actuallearnedproducer3986451/start48591250andpipeline3981233/start48338739liveat50/300cases. No restart/conditionchanges/originalfrozencontentmutation. Documenthistories/progressreceipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261028T071951Z_matched_figure_progress.json`; globalinventoryremainshistoricaluntilallwritersclose.


## 2026-10-28T18:00:00.152155+00:00 — Full learned RGB closure and direct BF16 contrasts

Previous goal turn: no progress (conversation clarified BF16 target without execution). Revalidated the same actual run: all six stages completed0, native300cases finished08:25:24UTC, publicreplay11:28:13UTC and independentstatistics11:28:24UTC; no actual RGB GPU process remains. No restart or scientific source/output edits. Rootreview initially asserted mutable manifest auditflag; preserved archivedfailedsource3d7e063358fab09206572fdd3ab2a48d9c49f1f565be351b6d08f1b73a47c57f and correctionrecord. Producer manifest remains immutable with independently_audited=False; passed separate audit binds its exact SHA. Correctednewsource797797832d30994d7100a9518f65e13e4f91464c9575c45e946359648d56b8f7archivedbeforeexecute, rootreviewcompleted0. All300cases/sixlogs/68source55inputhashes/33contrastunits and330000bootstrapunits verified, reportCIFAR_CNN_PROBABILITY_FINETUNE_REVIEW.mdsealed. All18,474,750,000counters/30millionlogits/1millionmean-controllogits audited. T1024expected55.0924%,sampleaware55.0146%vsFP3256.2580%; losses1.1656/1.2434pp, below frozenloss2.6936pp. All lowerbudgets retained, extratrainingconfound/biasedsurrogate/consumeddata remain. Native/publiceach659904000000retained+1473000000warmup schedulednodeupdates; no physicalcostclaim.

User clarified desired BF16-like taskquality with onepbitperweight. Newretrospectivederivativesbf16_comparison/run.pySHAf256ea57830c147d4a309e5073c27edae5f0576d26f2bfc0bc7248c66c8084c9,verify.pySHA79f9cfbd5afb70957c669cb80429c6f735ddf94190a53507a10275f33257feb2archivedbeforeexecution. Prepare/run/independentverifyeachcompleted0. All24contrasts(two datasets,two methods,three budgets,two BF16paths),600raw nativecase metrics+20BF16paths/62millionstoredlogits and240000bootstrapscalarvalues independentlychecked. Zero newnative draws/no extra statisticalreplicas. AtT1024sampleawareFashion85.0262vsCPU_BF1685.1100%,delta-0.0838ppCI[-0.442480,+0.262810]; RGB55.0146vs56.2940%,delta-1.2794ppCI[-1.577410,-1.015395]. Allretrospective/descriptive/unadjusted; no formal equivalencemargin/test orfreshconfirmation. Exactweightprecisionandtotalcontroller/bias/rangecostsremainseparate. ReportSINGLE_PBIT_VS_BF16_REPORT.mdandcompletionsealed.

Fullgoalactive. Documentationpriorbytesarchived; historicalglobalinventorynotrefreshedyet, nowstaledueto postclosurederivative/docwrites, notliveRGBjobs.


## 2026-10-28T18:03:43.551936+00:00 — Larger temporal budget validated before BF16 task follow-up

Previous/currentgoalturns: progress. Remaining RGB gap suggests testing higherT while preserving one freepbitperweight. Existing expectednominalmean57.22%versusT1024native55.0924%; sampleawaremean55.84%versus55.0146%. These controls are nonlinear mean-weight graphs, not E[finite-sampleCNN] or guarantee of improved quality. T4096 chosen after original consumeddata; no freshconfirmation/equivalenceclaim. Both original methods and alloldbudgets retained.

Newoperator248e228c99e61a3e6a7482c55174c3e6646679806515882ec01aa45d7c4fd4ed exactlymatches priorforward except cap1024→4096/docstring/unusedrecoveromission; integernativegateoverflowguard unchanged. Witness e7e9c3f9012e5ef2dffc07cad7edfc14e915b2d5b7a3501ed477caf164554dfc and rootreview05115b9e28665c6a486436966001cd58039888aeb39e67bdb8517e2456b9d51d archivedbeforeexecute. Prepare/nativeGPU/rootrevieweachcompleted0; actualGPUhandleexecsession77404closedexit0. All10learnedmodelT4096cases/16validationimages/140arrays/985320counter entries native-public exact. Perpass321781760retained+78560warmup updates. Worstcaseint32gatebounds28200960/75202560/267386880safe, allcounterdtype/finiteF64outputs checked. No labels/taskaccuracy or probability-controllerchange/extraweightnodes.

ReportCIFAR_CNN_HIGH_BUDGET_NATIVE_WITNESS_REPORT.md andcompletion/rootreviewsealed. Plannedfull follow-up100conditions (two methods,fivemodels,tenMCseeds,T4096), all10000historicallyconsumedtestimages,2011136000000retained+491000000warmup schedulednodeupdates pernative/publicpass,6158250000counter entries. Designfixedinplanned_task_followup.json; sourcefreeze/fullpipeline andstatisticscode still pending beforelaunch/taskscoring. BF16/FP32/ownmean/oldT1024/methodcomparisonsallrequired; no outcome-selectedequivalencemargin. No newtaskresultorhardwareclaim. Fullgoalactive; globalinventoryhistoricaluntil newwritesend.


## 2026-10-28T18:10:25.010632+00:00 — FullT4096 native RGB follow-up launched

Previous/currentgoalturns: progress. Fixed100cases after all10validationcasespassed: samefiveoriginalfits,two unchangedtrainingmethods,ten inferenceMCseeds,all10000historicallyconsumedtestimages,T4096onefreepbitperweight. Sourceprotocol312045d604ab78a48cce66eae78193855a3605ad3d3fa92066a5d66d15a80ae0sealedbefore new task scores;80source/104inputhashes. Newlyarchivedcommonc75575455a55b9318c2799b1651c93bc470114817f5c1ea1ceba0f2f72239b5c,run2099c10fae9170a42a753f25bde278d5de81a8729b99bb62577487825b8c4bde,statistics5b59cf44fa742b83d7542184a7f28af89ca0b7087ad1fe440d5d8c7228db63db,verifya9d625368595902f6b50ae81e067dc0ea6659b5f316eab7185f34b1ffb069eed,pipeline99dbef7100163cdcef8d801ee0ab04d3e55b071596eeaa8158af6341c5ce3a28. Contrastindexing testedwith scalar syntheticdata for all15rules/210values beforefreeze; no fake native success fixtures. All15comparisons (bothBF16paths/FP32/ownmean/ownT1024/frozensingleT1024/wordT256 per-method plus methodcross) predeclared. Priorcomparisonrefs not matchedT4096work, explicitcosts.

Actualdetachedpipeline4045681/start52550488 launchedonce18:07:~UTC; controls_copycompleted0, originalnominalcontrolsbyteidentical. Nativeproducer4045712/start52550574 live with2/100cases. No partialaccuracyreleased. Eachnative/publicpass2011136000000retained+491000000warmup schedulednodeupdates,6158250000counter entries/10millionfinal logits; fullpublicaudit beforestats, independent fullmetrics/10BF16paths/150000bootstrapvalues beforeacceptance. Combinedmonitor5408b52df2d23f2d4b279399267114e97ffcf6c7c78778b5250fe7005a181c59archivedbeforeexecution verifiesactualPID/startticks/argv/schedule/hashseals/allpriorclosedcohorts/future250MBbound~1.2TBfree. Oldmonitoralone excludesnewpipeline. No originalscientificcontentmutation/restart/physicalcostclaim. Newprotocolreport/preparationreviewsealed. Goalactive/globalinventoryhistoricalwhilewriterlive.


## 2026-10-28T18:18:24.199644+00:00 — Direct BF16 sampling curves exported and verified

Previousgoalturnverifiedwait; currentprogress. Newfigure sourceseebb94f3a5297e5e9bac0e93bbd2e6b3fb6dd8112be83de35dc952c55918306e and independentchecker4f3bfab20b300d78920da3e0bcef20209e2750b3098096df3880b1e8fabd6b48 archivedbeforeexecution; prepare/run/verifycompleted0. All12CPU_BF16accuracy-differencepoints(two methods,threebudgets,twotasks) rendered; all24bothBF16pathcontrastsinCSV. Coverage/metricunits/pairedintervals/120000existingaccuracybootstrapvalues checked fromclosedBF16evidence. StandaloneSVG/PNG/CSVandreportSINGLE_PBIT_BF16_SAMPLING_FIGURE.mdsealed. PNGinspectedviaimageviewer: labels/errorbars/legend/footernocrop,allnegativelowerbudgetsvisible. FormalBF16equivalence,numericalprecisionandphysicalcostclaimsabsent; extra training/confounds/consumeddata retained. No newnativeorbootstrapreplicas.

High-budget actualpipeline4045681/start52550488 andproducer4045712/start52550574 remainlive,11/100cases at this receipt; fullmonitor80source/104inputchecks/250MBallowancepass~1.2TBfree. No partialT4096metrics. Source/input/protocolsunchanged,norestarts. Goalactive/globalinventoryhistoricalwhilewriterlive.


## 2026-10-28T19:36:39.569281+00:00 — All100T4096native cases complete; full public replay live

Previousgoalturnverifiedwait/currentprogress. Originalproducer4045712/start52550574andwrapper4045711/start52550572completedproduction0at19:34:45.799464UTC. Samepipeline4045681/start52550488automaticallyadvancedpublicreplay at19:34:45.952799UTC, wrapper4055523/start53071280actualGPUauditor4055524/start53071283verifiedlivewithliteral.run auditargv. No restart/conditionschanged. Onecasepubliclyverifiedat19:36:13UTC. Prior45secondwaitreceiptclassifiedobservationasverifiedwait; this subsequentcompletedproductionreviewchangescurrentturnclassificationtoprogress.

Rootproductionsource93239bdcd1c486f0a2816c2ebe28e4601a01694192d36757bbd1375c6e6e271barchivedbeforeexecution, completed0. Full100caseSHA/receipt(1250,14,32)U8/logits(10000,10)F64finite/clippedcoordinates/exactcoverage and2011136000000scheduledretainedupdates checked withoutlabels/taskmetrics. Reusedcontrolsbyteidentical,80source/104input/prereqcompletionseals unchanged. Storeduncompressedcasearrays136002400B=136000000receipt/logitbytes+2400clipped-countbytes; smallsubtotalomissionexplicitlycorrectedinnewreceipt/report, originalprotocolretained and137MBcase/250MBtotalboundsstillcoverit. No scientificoutput/sourceedits.

Publicreplaymustregenerate6158250000counterentries/10millionlogits; eachnative/publicpass2011136000000retained+491000000warmupupdates. Full replay/BF16/metrics/150000bootstrapchecksrequired beforetaskqualityacceptance; none claimed. ProductionreportCIFAR_CNN_HIGH_BUDGET_PRODUCTION_REVIEW.mdandreviewcompletionsealed. Fullgoalactive/globalinventoryhistoricalwhilewriterlive.


## 2026-10-28T21:39:49.151113+00:00 — T4096 RGB cohort independently closed

Previous goal turn verified wait; current turn progress. Original pipeline automatically finished all five stages with exit code zero. Public replay ended 21:37:33.311369 UTC; statistics ended 21:37:34.452556 UTC; independent statistics ended 21:37:36.658526 UTC. No restarts or frozen experimental source/output edits. All100cases/6,158,250,000counters/10millionnative-logits/10BF16paths/1millioncontrol-logits/150000bootstrap scalar values pass. Root closure source aab406a5db1ba8066e9bb8beced23f4741b38668003cf59032be0d61b8b592d5 archived before execution; completed zero; CIFAR_CNN_HIGH_BUDGET_REVIEW.md, root_review and review_completion sealed. All15contrast directions/units/intervals, five stage logs,80source/104inputhashes and completion-bound artifacts reviewed. Terminal monitor confirms no actual scientific GPU workers; old pipeline4045681/start52550488 and auditor4055524/start53071283 terminal.

Expected-weight accuracy56.5842% vs originalFP3256.2580%/CPUBF1656.2940%, difference+0.3262/+0.2902pp, BF16 descriptive95%[-0.072005,+0.662010]pp. Sample-aware55.6122%,BF16difference-0.6818pp[-0.969005,-0.437990]. Bothmethods/all15contrasts/lowerbudgets retained. Extra8trainingepochs, fixedhistoricallyconsumedimages, retrospectiveT4096selection and absent equivalence margin/freshdata remain limits. Comparable observed task quality supported; numerical BF16 precision/formal equivalence/physical advantage unproven. Fourfoldlocalrounds versusT1024, eachnative/public2011136000000retained+491000000warmup scheduled evaluations.3928weightnodes plusdeterministicmetadata/commands/bias/ACT8/counters/F64readout. Expectedmethod aggregateclipped[0,0,19],sampled[0,0,0] retained. Fullgoalactive/globalinventoryhistorical pending post-write refresh.

Status/requirements prior bytes archived verbatim; edits recorded in execution receipt.


## 2026-10-28T21:44:12.065690+00:00 — Same-fit BF16 controls remove training-history mismatch

Previous goal turn progress; current progress. New run841218fb9b9f702ecb013954599bf3f13a00f49fe5a9d6fa953507c655ee5313 and independentverifya76b594aafc8f5e8a424eacbbdd801c2258f6f9d9af8a97c0d1954a98fd5d6c8 sources archived before execution. Validation-only30paths×64inputs with independent BF16 RNE bit rounding/fullfunctionalCPUgraph/NumPyF64spatialcontractions passed. Protocol frozen before new BF16 scoring. Prepare/evaluate/verify all completed0; sessions4966/72607/20206closed, selfcheck54784closed. Full30controlgraphs/3millionlogits and all24contrasts/240000bootstrap scalarvalues independently checked. Reuses all2methods×4budgets×5models×10MC parent outputs and model/MCbootstrapindices; zero new stochastic draws/training/bootstrapreplicas. Full samefit report/evidencecompletion sealed. Rootreviewsource7ad27bfbc89ce895ba39e9ab3c0d55be593d5630b232d873f48bdf58f7f4674b archivedbeforeexecute/completed0, root_review/review_completion sealed.

AtT4096expected56.5842%vs samefitCPU_BF1657.1700%, delta-0.5858pp[-0.703400,-0.464800]. Sample-aware55.6122%vs55.8400%, delta-0.2278pp[-0.278605,-0.179795]. BF16weightACT8F64 controls57.1520/55.8420%, nativegaps-0.5678/-0.2298pp. Threecontrols preserve learnedweights/traininghistory; ACT8path additionally holdsactivationdiscretization/span/F64biasreadoutfixed. F64exportmeanscastFP32thenBF16explicit; CPUaccumulationwidthunclaimed. EarlieroutperformanceoforiginalFP32remainsend-to-endtrainingcomparison, notencodingadvantage. Bothmethods/allnegativebudgetsretained. Historicaltestdata/descriptiveintervals/formalequivalencemarginabsent/physicalcostsremainlimits. Status/requirements snapshots archivedverbatim beforeprepend; globalinventoryhistorical/stalependingrefresh. Fullgoalactive.


## 2026-10-28T21:50:25.552404+00:00 — Same-fit four-node comparator validation complete

Previousgoalturnprogress/currentprogress. Original learned_word adapter661767243807679879ddd1e2268cc598096275b8e58d8569e1e9f51aaa9b1018/replayd8a0f028d40dcba725fdac0e2bba40aa0ae6758fe4ed15cb5ef3eebcf9459dd2/witness6a42b089dc9e1a2f8ab0030c704985df716c3cef3581d2ff523f1c3bdc67bec0 archivedbeforeexecute. Firstattemptsession37652terminal1 at meanidentitycheck; reciprocaltimes versus divisionexport differed1.11e-16. Source/protocol/run.log retainedunchanged. Nativefirstcase502784retained+31424warmup; public0. Explicitfailure receipt preserved. Corrected namespacelearned_word_v2 changesdivisionorder/importnamespaceonly; no tolerance relaxed. Sourcesadapterunchanged,replaybc9d7fdaa6f9f30f55f3a80f9f2ad33c600b7522e44549847e012133a5963b6f,witnesse2854aec4b1c51e536ee621e9fedbe403e2b1ff8dad89531da4bfa984c93e364archivedbeforeprepare/run.

Session12288completed0. All10fits×4wordbudgets16/64/256/1024×16validationinputs(two8banks), no labels/accuracy. OriginalrangesindependentlyreconstructedfromoriginalFP32checkpoints, learnedmeansfromcommands, fourcodesfromprobabilities; nativeadapterusesexportedmeans/originalranges. ExistingTHRMLfour-free-nodeobserver unchanged. All40cases/1120bankarrays/3941280counters native/public exact, each427366400retained+1256960warmup scheduledupdates. Headbound1002700800<2^31. Matchesone-node64/256/1024/4096retainedwork; fourfoldweightnodes and largerwarmupcharged. Rootreview8d6946f731cc962f14ae00315f8ac40dafde44780dade000fb5591c34ac3e08aarchivedbeforeexecution/completed0, allcaseSHA/schema/bounds/39280weightcodecoordschecked, report/reviewsealed. Fulltestmatchedworkcomparison/nominalcontrols stillpending, no hardwareformat/energy/qualityadvantageclaimed. Status/requirementshistorypreserved. Fullgoalactive/globalinventoryhistorical; no actualscientificGPUworker remains.


## 2026-10-28T21:56:54.683552+00:00 — Full same-fit matched-work400case study launched

Previousgoalturnprogress/currentprogress. Sourcescommon d2da4868dcb2fc3b42d94808369fd6bd5a00be7ca3c317ce7d9e17cd26c45b64,pipeline8b66f96f51fa153591d3d25fcfb89ae8b446e4791a88b7ead1ed9436458b6dbe,run c2535b8884e28755c98bd6bce9165a944fe8677886cfeb83aa4297524031bae9,statistics4f35a402f1e0dae12462f54735d94070ac5ed11ce258d7b3b2ec419ab980ae10,verifyf33fa55f7c28ba6927d309216fcedcad2b324fb7fd1774359caaa006cc873e6f archivedbeforeexecution. Nominalwordmean10fits/39280coords/6400validationlogits independentbitprobabilitymeanlaw+NumPy/Torchgraphchecks passed(session60571completed0). Syntheticscalarbootstrapindexing28values independentMC/deterministicreference passed; no fake nativeevidence. Frozenprotocol98e651dd81854de1cbf239468a355c02c578bd35452f438159cd6938a55a958c andpreparationreport reviewed92source/177inputhashes/prereqseals; all400cases/24contrasts/budgets fixed before newtaskscoring.

Onepipelineonlysession31977, PID4071240/start53912453 started21:54:58UTC. Nominalcontrolscompleted0at21:55:01UTC. Producerwrapper4071270/start53912887 actualGPU4071271/start53912889 live,productionstart21:55:02UTC. Latestmonitor3/400cases; no taskmetricsaccepted. Same10fits/ranges/biases/ACT8spans,4freepositionalpbits weightT16/64/256/1024 against unchanged1freepbitT64/256/1024/4096. Eachnative/public2671040000000retained+7856000000warmup schedulednodeupdates;24,633,000,000counters/40mlogits;1mnominalcontrol logits/39280weights/240000bootstrap scalarvalues require independentverification.8single-minus-word+8word-minus-mean+8word-minus-samefitCPU_BF16. Fivepairedmodels,independentwordMCresamplingroot1937520/nativekey1927520,originalsingleMCindicesretained; consumeddata/descriptiveonly. Equalretainedwork,4xwordwarmupcharged. Bounds544009600rawcasebytes<600MB+200MBsupport800MB,totalfree~1.2TB;int32max1002700800<2^31.

Newactualprocess/source/input/prefix/capacitymonitor02049532a9248ed5b8ae53dffa7eb5a63d43abf9efc0a294b850df10ad004f58archivedbeforeexecution; olderhighbudgetmonitor excludesnewstudy. Fullpublicreplay/statistics/independentaudit queuedsequentially, noautomaticresume/restart. Documentationoldbytesarchivedverbatim. Globalinventoryhistoricalwhilewriterlive; goalactive.


LFM3 adaptation requested and research resumed 2026-10-29T18:13:46.244200+00:00. Pinned LiquidAI/LFM3-1.7B at ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7. Local meta architecture construction succeeds: 1,728,801,536 unique parameters / 286 tensors; 262,144,000 tied embedding weights, 1,466,269,696 projection weights, 186,368 depthwise convolution weights, 201,472 norm parameters. Checkpoint HTTP range accessible (3,457,635,448 bytes); original HF download session 85853 live without completed weight artifact. Preparation source archived before execution; deterministic smoke and stochastic conversion pending. Full source/status: results/thrml_spw/lfm3_1p7b/preparation_status.json. No LFM3 task accuracy claim.


LFM3 progress 2026-10-29T18:18:14.994245+00:00: checkpoint downloaded and content hash verified; full BF16 GPU forward smoke completed (session 89078 exit0). First FP32 operator pilot session17912 exit1 at T256 due reduction-order replay discrepancy; failure preserved. Derivative v2 archived SHA5afe0f7c53f9c24ef8e896d4fdd75f85b864b958fe191ef484b63d90a8f588dc executed session63620 exit0; all40 ACT8/int32 THRML linear/convolution cases independently replay exactly via public states/int64. No full stochastic model inference or language-quality claim. See LFM3_ADAPTATION_PROGRESS.md.


LFM3 full operator progress 2026-10-29T18:21:14.735083+00:00: full projection12,582,912 and depthwise14,336 weights executed atT1/16/64 using dynamic finite8bit fields/bounded banks. Session79719 exit0; all384 tile replay checks exact. Independent audit source27485b2a037149a4d3c8a992d57de6c0df3029f1aab4e8e4d734122cd79f48a8 archived before execution; auditexit0 recomputed18metrics and12,597,248 commandcoordinates. No full stochastic neural forward/task-quality claim. Sources/exports/protocols/results preserved.


LFM3 end-to-end decoder conversion 2026-10-29T18:28:21.331443+00:00: bridge session43784 exit0; all166 native decoder projections/depthwise convolutions,1,466,456,064coordinates84.825%,T16oneconsumed16tokenprompt. StableFP64diagnosticCE BF163.69359826 / mean4.01102160 / native11.33334650; substantialdegradationretained.23,480,598,528retained+1,467,537,408warmupincludingpadding. IdentitybridgeexactBF16. FirstauditfailedF32vsF64CE comparison; derivative686ec5a51de2f84a9d2db0ab7faf3a82190aea594a2dfd6de1cc27dbf9a6e996 archivedbeforeexecution passed independentlyrecomputedlosses/coverage. Embedding/head/normconversion and languagebenchmarksremainpending.


LFM3 all-parameter progress 2026-10-29T18:38:42.438225+00:00: preflight89968 exit0 all8 embedding/norm checks passed; fullmodel79585 exit0 all1,728,801,536uniqueparameters/287modules covered,T64oneconsumedprompt. Independentverifiercb2b3c4677e248d98e8f370b28da4f4e1d26116482c1442ce73531e678f7fb4c archived-before-run/exit0 exactparametercoverage andstableFP64losses. CE BF163.69359826/mean3.94107528/native11.64965896 (degradation retained).110,714,601,472retained+1,729,915,648warmupinclpadding. Frozen8WikiTextwindows512predictedtokens prep23260exit0. Resourceinventoryaccountedstates/8bitcontrols/groupmetadata; nohardwareadvantage. Fullqualityadaptationremainsopen.


LFM3 sampling/ranges 2026-10-29T18:46:52.724340+00:00: fullT256session25482exit0, audit93813exit0 verifiesall1.729Bcoverage/287modules/exactBF16andmeancontrols. CE6.91211737 improvesT6411.64965896 butfarfromBF163.69359826.442,858,405,888retained+1,729,915,648warmupinclpadding. Row/block128finitecontrolmoments session86163exit0; independentendpointaudit71717exit0 validates2288tensorstatchecks/full1.729Bcoordinatespermethod. Varianceratio0.38997248; metadata5,884,216→108,251,440FP32bytes. Analyticalonly; groupedneuralexecutionnotyetimplemented.


LFM3 group128 2026-10-29T18:57:21.602411+00:00: nativegroupoperator8d9d3c09bb625419552aa7ea2c0b8ecc9558206aa4918444fcc65cebdce5377b archivedbeforeexecution. Preflight74629exit0 all16cases; offlinee658a2f4d0e8e136ff6068d6d560a8f64bc407208ec1719a0bf1acc46ac87a22 verifies200publictiles/28672exactintegerentries/79616commandcoordinates. Fullmodel95548exit0; audit40f8c2dcddb0834f2f2fc80dbe80c3c70a9469b3efc19768ab6d613be87f4e4dexit0 full1.729Bcoverage287modules/matchedrowT256work. CEgroup4.51365599 vsrow6.91211737/BF163.69359826; mean3.91324381. Singleprompt/seed; noequivalence/benchmark/hardwareclaim.


LFM3 precisioncontrols 2026-10-29T19:08:20.537902+00:00: preflight10141exit0 all32cases; offline90e8282eef097265c3d9ddc7c7cec2fbb69ec782f2cfa1defc7231e4b64b56fd verifies400publictiles/57344integerentries/159232codes. Controlgrid8062exit0 and independent1c24f91da99f6ac9f337ee29df74010cfc928f49df838065eb955bd7ed11d602 verifiedzero nativework/fullcoverage1148calls. CE(f8a8/f12a8/f8a12/f12a12)=3.91324/4.43746/3.66172/3.67006vsBF163.69360; allretained singleprompt. New nativeprecision_T256 session28546 launchedsource8e148030536b8f06250563318ef3ca80e0f32f336f24fdadff12ced88282c4b8; awaitingresult/audit. Rootstatus/evidence histories archivedverbatim beforeprepend; source/date receipts retained.


LFM3 nativeprecision 2026-10-29T19:15:21.856422+00:00: T256session28546exit0; independent844ddae82eb354304cedcb8032c0cab1e2a08a4e0da1c684a9026bcbe9d9b8f6exit0 verifiesfull1.729Bcoverage287modules/exact12bitmeancontrol/exactpriorgroupT256nodework. NativeCE4.21117023vsold4.51365599/BF163.69359826. Singlepromptseedlimitsretained. NewfixedprecisionT1024session86404 source6eb3d369393ae81c4f05c3c4ef4f6bb949c46307bd40e37f1fbf99b511dfeaec launched; futureverifierfd9a92a935014945d4cf1db72385923d623516eee22f9905ed65888e4ccbcaa0 archivedbeforeresult. Rootstatus/evidencehistorypreservedbeforeupdate.


### 20261029T192230Z — LFM3 full native T1024

Full LFM3 native execution at fixed block128 ranges, 12-bit controls and ACT12 completed and passed independent verification at T1024. All 1,728,801,536 unique learned parameters / 287 operator modules covered. Diagnostic cross-entropy 3.846787889 versus T256 4.211170230 and BF16 3.693598262; top-prediction agreement 15/16, logit MSE 0.2927633523. Nominal control 3.670062474 and BF16 arrays match prior controls exactly. Retained scheduled node evaluations 1,771,433,623,552 plus warmup 1,729,915,648 including padding: retained work exactly four times T256, warmup unchanged. Single consumed prompt / one seed, not benchmark perplexity or BF16 equivalence. Full frozen 512-token multi-seed language evaluation and fair conventional quantization baselines remain pending. No physical advantage inferred.

Native session86404 and independent audit session96986 both exited0. Source snapshots archived before execution. No failed runs removed or restarted; broad goal remains active.


### 20261029T193030Z — Language controls verified; corrected five-seed native cohort launched

Frozen 512-token LFM3 language controls completed and independently audited across all 1,728,801,536 unique parameters. Matched batch8 token NLL: BF16 4.635723773 (subset PPL103.102514); affine INT2/g128 10.961896503 (57635.645635); INT4 4.973005522 (144.460415); INT8 4.615521421 (101.040499). Quantized weights execute through BF16 operators; no integer-kernel performance claim. All512 saved token losses, tensor names/shapes/group counts and artifact hashes checked. Separate batch preflight found BF16 batch-vs-individual logit MSE0.0074080914 (maxabsolute3.3671875); preserved individual controls and used independently matched batch controls for native comparison. Batch BF16 arrays exactly match preflight. No native512-token quality result yet.

Five-seed T1024/fullparameter/block128/12-bit-control/ACT12 native cohort frozen before native quality scoring: seeds202610300..202610304, all eight65-token contexts in one batch,512 predicted tokens. Shared stochastic draws across batch/context positions; independent seeds are MC clusters. Report all results and paired tokenNLL differences; MC95%t4 interval across5 seeds, no confirmatory equivalence claim. First driver failed in nominal first-embedding setup (CLI args shadowed by hook tuple); no native scoring completed, all failure sources/protocol/logs preserved. Corrected only CLI variable name in separate v2 source/output namespace, with identical seeds/inputs/budget/analysis. V2 protocol09d83eb3a431096cd7c8fca652d12c014cd15ccd0a19cf4f943485c82b253696. Sequential native/audit pipeline session50670, parent42522/child42523/worker42626 confirmed live, progressing nominal layer3 and BF16/identity exact. No automatic restart policy. Broader benchmark quality, matched spatial baseline, training/nonideality/hardware work remains incomplete.

Previous goal turn progress (full T1024 diagnostic completed/audited). This turn progress: completed full language controls/audits, identified numerical batch discrepancy, built matched controls, preserved setup failure and launched corrected predeclared cohort. Goal active. Global evidence inventory remains stale; not refreshed during live writers.


### 20261029T193420Z — Conditional MC analysis frozen before native scores

Five-seed language cohort remains live (native session50670, parent42522/child42523 identities verified); no completed native seed yet. First seed nominal-control pass logged tokenNLL4.638117449 versus matched BF164.635723773; these are provisional log metrics pending final saved-output audit, not native task quality or a hard performance lower bound. Native sampling has begun after full287-module nominal pass. All512-token inputs/five seeds/budget unchanged. Previous goal turn progress; this turn progress: froze and launched complete four-control paired NLL analysis and independent Decimal arithmetic audit. Analysis session63085 watches only the existing cohort, cannot restart inference, and runs after all5native seed audits pass. Reports seed-level MC Student-t4 intervals conditional on fixed512tokens, explicitly excluding unseen-text/equivalence/hardware inference. Statistics are not yet computed or verified.

Sources archived before execution: language_statistics.py0ca158c367514983f84d7a8b5c414cdcfe6f412c2510763c663d0026ee2598f6; verify_language_statistics.py e641981721d24b0315946e673d409bda5b7b70b53c910a0ac52bda06ba5aed41; watcher f6a12fdc58fa9412903d39d8dddb9dfe5bc4b1fc7f1730f61439406454e262a8. Analysis protocol1db4339ec1457b271b182f802be61f3a8a506b55aa12bc8b058cc8f4ccb63520. Global inventory remains stale during live writers.


### 20261029T193816Z — Full-scale control/state/metadata bookkeeping

LFM3 block128 language resource accounting independently verified: N1,728,801,536,13,531,430groups,108,251,440bytes for two FP32 range values/group. One packed state bit plus packed12-bitcontrols plus ranges =2,917,553,936bytes (13.50093bits/parameter,84.3808% of BF16 weight payload3,457,603,072bytes). With16-bit command storage total3,781,954,704bytes (109.3808% of BF16 weight payload). These are conditional digital payloads, exclude checkpoint/framework/node/activation/accumulator/compiler/IO overhead and do not convert analog controls into measured area. No actual memory or physical improvement claim.

For frozen eight-context language mapping,299unique lookup rows: predicted per-seed retained1,772,027,117,568 plus warmup1,730,495,232 scheduled node evaluations including padding. Verified by independent dimension formulas and exact lookup-row difference from earlier audited T1024 prompt; completed language-run receipts still pending. Fully resident logical device count1.729B differs from time-multiplexed banks: maximum491,520nodes, sum623,104across cached fan-in shapes. Under equal per-node sampling rate/energy and full parallelism, singleT1024 versus four-nodeT256 gives equal retained work but4times sampling latency; requires matched quality before any advantage conclusion. No empirical four-node LFM3 quality result yet.

Five-seed native session50670 and analysis session63085 confirmed live with process identities; first seed progress {"stage": "native", "operator_calls": 348, "last_module": "model.layers.5.conv.in_proj"}. No native512-token quality score accepted yet. Goal active; prior turn progress, current turn progress (new verified resource accounting).

Sources archived before execution: language_resource_model.py086e56530cddad64a809d183cca695a00bb3b3f1284a06a5569edbf7daa23477 and verify_language_resource_model.py f2fefede8a5b7b1c13b953017b6e096b5409b346b56303d844d96d86bacac612. Both executions0. No hardware energy constants invented; global inventory stale during live writers.


### 20261029T194420Z — Four-node positional LFM3 comparison

Internal four-free-p-bit positional LFM3 operator baseline implemented and independently validated on16saved-input cases (linear/convolution/embedding/norm x T1/T16 x mean/native). CPU officialTHRML;800preserved public bit-trajectory tiles,114,688exact integer accumulator entries,318,464programmed sign coordinates including padding verified offline. Finite fields+/-4, coefficients1/15,2/15,4/15,8/15, block128 ranges, ACT12 inputs, direct group output accumulation. Two programmable field levels/node. Not an asserted Extropic hardware format or full-model quality result.

Full1.7288Bparameter/five-seed/frozen512-token four-node cohort frozen and queued atT256, matching singleT1024 retained node work including padding; warmup4times separately reported. Queue session48884/process45530 confirms existing singlecohortprocess42522 identity and waits for all single-seed/statistics audits; no inference restart. Same originalcheckpoint, inputs, ranges and ACT12; every learned tensor, including norm gamma and tied embedding/head, converted. Each word bit is simulated sequentially through time-multiplexed banks with independent keys; no physical6.915B-node residency claim. No four-node language quality score yet.

Control information must count the actual encoding: four-node endpoint-word needs4sign bits per weight plus shared endpoint values, not12independent command-information bits per node. Under a packed digital state+command convention its4statebits+4signbits+FP32groupmetadata total1,837,052,976bytes; this is a hypothetical payload, not measured p-bit circuit area or storage. Single method uses12controlbits+1statebit+same metadata2,917,553,936bytes. Device-count, control-information, latency and energy tradeoffs therefore require separate analysis; no systems advantage inferred.

Single language native session50670 remains live; latest {"stage": "native", "operator_calls": 497, "last_module": "model.layers.17.feed_forward.w1"}. No completed seed quality accepted yet. Analysis session63085 remains associated with the unchanged singlecohort. Previous goal turn progress, this turn progress (four-node validation and concrete frozen queued comparison); broad goal active. Global evidence inventory stale while writers run.

All new drivers archived before execution. Preflight session52856 completed0 and independent audit completed0. Protocolbd0262d4c3fef8ef0e7d89c7d9ef9a62e59553e2c1f7356faf0a36ae46f3f135. No existing cohort restarted or changed.


### 20261029T195314Z — First512-token native LFM3 seed accepted after audit

First frozen-language native seed202610300 completed and passed independent fullcoverage/saved-logit audit: all1,728,801,536uniqueparameters/287modules and all512token losses verified. NativeT1024 tokenNLL4.883533826/subsetPPL132.096647 versus matched BF164.635723773/PPL103.102514; nominal control4.638117449/PPL103.349603. This seed has about28.12% higher subsetPPL thanBF16, while loss is below deterministic INT4/g1284.973005522/PPL144.460415. One of five seeds only: no cohort interval, standard fullWikiText quality or BF16 equivalence established. No selection/removal of seeds. Secondseed202610301 started in unchanged pipeline session50670.

Full-language retained1,772,027,117,568 plus warmup1,730,495,232scheduled node evaluations includingpadding exactly match independently predicted resource totals, now checked against completed audited receipts. All BF16 control arrays exactly match matched batched baseline. Operator trajectory replay is inherited from public preflights, not fullmodel raw trajectories. Original BF16 kernels execute before native hooks and simulator retains checkpoint/working buffers; no hardware performance claim.

Paired single-minus-four-node analysis machinery frozen and queued (session4809/process46274) before first full-language native score. Primaryfive-paired-seed NLL difference plus all8 deterministic control contrasts, MC95%t4 intervals conditional on fixed512tokens; source/Decimal arithmetic audit still pending actual finaldata. Four-node inference queue session48884 waits for fullsinglecohort and statistics audits. Protocol d5f9351b9088d4e9fb49ff8fa2f8699a0e3e05d51589d76e41c80917be5d3e98 binds three new archived sources. No automatic restart. Prior/current goal turns progress; broad goal active.

Fullsource/analysis artifacts preserved. New paired-analysis sources: word_language_statistics.py1a32b6cbde840e7088877d7e066f61d6c4ab6e4236643351569f84bc232f3aec; verify_word_language_statistics.py57d436d542d504fbe1a717fe25145805cd63c3d8674838dac6e40478c6ce13b5; watcher9c0b117b02bcd35c1935a5e08105e7a1085b769913f448c024e3101d92e6586c. Global inventory stale during live writers.


### 20261029T200125Z — Full-checkpoint scalar bias/variance decomposition

Full-checkpoint scalar moment comparison independently verified over all1,728,801,536uniqueweights and13,531,430shared block128groups:2288tensor-moment checks. Single finite12-bit-field marginal squared-bias7.752955213e-10, one-sample variance0.002637572757; internal four-endpoint-node word squared-bias4.552020890e-6, variance per4-node-realization1.582739514e-6. At equal retained evaluations C, MSE_single=b1^2+v1/C and MSE_word=b4^2+4v4/C. Formal scalar MSE crossingC578.136632. C1024: single2.576529941e-6 vsword4.558203467e-6; C4096: single6.447139570e-7 vsword4.553566535e-6. All eight budget points4..65536retained, including budgets where word wins.

Independent verifier used different tensor chunking, two-endpoint scalar laws and complete16-state bit-error channel enumeration; source/checkpoint hashes verified. These are nominal independent stationary sigmoid laws in float64, not empirical full-model Monte Carlo or exact float32 probability arithmetic. Excludes input quantization, nonlinear propagation, correlations/calibration/drift, warmup, hardware cost and task quality. Does not prove BF16 equivalence or predict the neural quality crossing. Current singleT1024 scalar error is dominated by temporal variance under these assumptions; increasingT has a clear mathematical scalar effect but unproven application benefit.

Analytical scalar curves exported as PNG/SVG and tableCSV, visually inspected. V2 caption explicitly states single12-bit control versus two bias levels/node for word; initial figure/manifest preserved. Figure uses verified result/audit hashes and separateCPU environment matplotlib without mutating active experiment environments.

Native secondseed202610301 continues in session50670; latest {"stage": "native", "operator_calls": 367, "last_module": "model.layers.6.feed_forward.w1"}. Firstseed202610300 remains the only accepted language result (NLL4.88353vsBF164.63572); four seeds still require completion for MC cohort statistics. Word comparison/analysis queues unchanged. Prior/current goal turns progress; broad goal active; global inventory remains stale while writers run.

Sources archived before execution: word_moments.py726115a53fb6dd1d88a2e1a197815146b7804c245be3d27f002a90200edfd596; verifier864929e95c34d5ab85fbf6f4ad9ed23a92ed033e8c9b10655f54ecfb9bc868af; figurev2 source2212891d4f882e072d330b1a163891db2b800e33ff43f793390e64d11cf5836d. Producer session2098/audit53402 exited0. Matplotlib absent in Torch environment, found existing CPU THRML environment; no installation needed.


### 20261029T200859Z — Expanded spatial reference reveals strongK8 baseline

Expanded full-checkpoint scalar comparison to allK2/4/8 spatial words; independently verified4,576tensor-moment checks over1,728,801,536uniqueweights/method and complete4/16/256-state error channels. Prior single12/K4 boundary moments match unchanged. At C1024 retained evaluations/weight: singleT1024 MSE2.576529941e-6; K2T512 1.149034470e-4; K4T256 4.558203467e-6; K8T128 2.696399151e-8. K8 has about95.55times lower nominal scalar MSE than single at thisbudget. This adverse result is retained and qualifies the favorableK4 comparison. No universal precision or systems dominance claim.

Formal aggregate scalar crossings single vsK2/K4/K8 are C22.9152/578.1366/172866.8147, respectively. Under this nominal stationary independent model, single beatsK4 atC1024 but remains worse thanK8 throughC65536; its lower asymptotic finite12-bit-control bias eventually crosses theK8 curve. This is marginal reconstruction theory on real checkpoint weights, not empirical neural quality, BF16 equivalence, hardware performance, autocorrelation or calibration measurement. Single uses12controlbits plus one state/weight; each spatial node selects two endpoint bias levels. Device-count/control-information/latency costs remain distinct. No K8 full-language native run claimed.

All formats included in PNG/SVG/CSV artifacts, linked to verified results/audit hashes and visually inspected. V2 only adjusts text placement; initial figure/source/manifest preserved. The figure clearly labels12-bit singlecontrols vs two biaslevels/spatialnode and excludes neural quality/hardware costs.

Second language native seed202610301 remains live under session50670; latest {"stage": "native", "operator_calls": 546, "last_module": "model.layers.21.feed_forward.w3"}. One of five seeds accepted; no cohort interval yet. Single and matchedword neural pipelines/protocols unchanged. Prior/current goal turns progress; broad goal active; global evidence inventory stale while writers run.

Sources archived before execution: spatial_moments.py8896d0540a35197ae120c3b34c94ef6ff22493b9ac71f0421f3c8ba6ff2c6f22; verifier3808288f4f19a66740c979b1a06ee7a9fb78309a1d0fa39b5f4ba4badd018e30; figurev2 source41f18f0fb09468570b1ff0e19ef618a8a8bce98ca27ceaf421531e99d2893222. Producer89478/audit74363 exited0. No existing treatment, seed, dataset or stopping criterion changed.


### 20261029T201547Z — Second native language seed and separate adaptation data

Second frozen-language native seed202610301 completed and passed independent all512token-loss/full1,728,801,536parameter/287module audit: nativeT1024 tokenNLL4.973846934/PPL144.582016, BF164.635723773/PPL103.102514, nominalcontrol4.638117449/PPL103.349603. Firstseed4.883533826/PPL132.096647 remains retained. Both BF16 and nominalcontrol arrays match firstseed exactly. Retained1,772,027,117,568+warmup1,730,495,232 scheduled evaluations/seed includingpadding unchanged. Two of five audited, third202610302 running in original session50670; no cohort interval or equivalence conclusion yet.

Future probability-aware adaptation inputs now frozen and independently verified, with no training/model scores:256train windows x257tokens (65,536predictedtargets),32validation windows x65tokens (2,048targets). Official separate WikiText-2raw splits at pinned datasetrevisionb08601e04326c79dfdd32d625aee71d232d685c3, modeltokenizerrevisionac2586a34f1a1c03329dbdcdefabd8b6478ddaa7. Trainparquet newly downloaded/preserved locally SHAe83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7; validation cached SHA204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c. Exact65-token context exclusions verified against selectedtraining/frozen8test contexts;49,408training65contexts/32validation contexts. Every reserved window, offset, seeded selection, source hash and pinnedtest parquet/token identity reconstructed independently. This does not exclude shorter shared phrases or originalpretraining overlap. Existing512-token test is already consumed exploratory evidence, not a new confirmatorytest. No training starts while current GPU cohorts run.

Read-only data inspection found existing data/wikitext/wikitext-2-raw-v1.zip is487bytes and not a validzip (SHA29a8972ba186bcd1efe4d46096ee841b982a8b11ba1e8b4a6f697c8f261b98b6); left untouched. Used pinned official parquet splits instead. Currenttest.parquet checksum resolves exactly to the pinned dataset test split. No credentials requested or used (public HF download token=False).

Single/word inference and analysis queues unchanged; process identities rechecked. Prior/current goal turns progress. Broad goal active and incomplete: remaining3single seeds, matchedwordcohort, fullquality statistics, adaptation/nonidealities/physical-cost evidence. Global evidence inventory remains stale while writers run.

Preparation38134/audit12473 exited0. Archived sources: prepare_adaptation_data.py b62e903a3d54d94c48acab21bde2c369110a370fba80660a7452b5a0a62f9951; verifier eaa27686f457bfab4463562a26d4296ddd16dc707858298f635141eb35679531. No existing treatment or stopping criterion changed.


### 20261029T202544Z — Fixed-range probability-training CPU preflight

Expected-operator probability-training component validated on a controlled1024-weight LFM3 projection slice with16fixed captured input positions, ACT12 input quantization, fixed block128ranges, finite12-bitfields+/-4. Predeclared Adam200steps/lr0.03 optimizes exact nominal marginal squared-output error (squared mean error+variance/T) atT16 via biased control-rounding STE. Initial mean operator exactly matches validated precision_worker within strict floating tolerance. Native officialCPU THRML before/after across5x100pairedtrials, all16,384,000public binary observations and128,000integer accumulator entries checked. Empirical MSE0.0009293362625 ->0.0001674753075 (81.98% reduction); expected nominal MSE0.0009438328462 ->0.0001683412828. MC-theory zscores-2.15345/-0.51968 within retained6SE validation tolerance. No full-model training or language improvement demonstrated.

First gradient audit failed because the checker omitted the clamp derivative at exact fieldlimits. Independent local Torch2.14.1+cu130 probe showed clamp gradients[0,0,1,0,0] at[-4.00001,-4,0,4,4.00001]. Corrected strict-interior mask in separate verifier_v2; all original failures/source/logs preserved, no tolerance relaxation and no training/native rerun. Sixteen of1024initialendpoint parameters have zero gradient under this clamp convention. This is an optimization limitation to address before full-model adaptation (e.g. explicit projected-gradient convention), not a sampler-gradient proof. The STE is intentionally biased and controlled input positions were used in its objective; no unseen-activation generalization, neural quality, BF16 equivalence or physical benefit claim.

Third frozenlanguage seed202610302 remains live in unchanged session50670; latest {"stage": "native", "operator_calls": 401, "last_module": "model.layers.9.feed_forward.w1"}. First two accepted losses4.88353/4.97385 vsBF164.63572 retained; all5 required before cohort statistics. Word inference/analysis queues unchanged. Prior/current goal turns progress. Goal active, global inventory stale during writers.

Producer63386 exited0; original audit exited1 (preserved), derivative v2 audit exited0. Sources archived before execution: probability_training.py b26a69c8a14bf9529843765aec3ec65ce9c357a9ead3a1fc8d2da725319c0d78; worker c43b2d95ad158b57d72ef7ad2b6d94ee73acfa6058a3a7755b9cff5e2f2b696c; runner8a6f4a5e00dcac6891f59942ed15a424e2adda9e510e8ab9595ba66d8240c756; v2verifier f9bc16ca3ebda058c5e4beaabeed19b161867a2d350aea27ecd7539883d2effb. No changes to frozen language inference treatments or seed choices.


### 20261029T203141Z — Explicit projected-gradient boundary convention

Explicit projected-gradient probability component implemented and independently verified. Forward still uses clipped finite12-bit commands in[-4,4]; backward uses declared identity STE through clipping/rounding, with explicit parameter projection after every update. Separate controlled boundary objective and8SGDsteps proves both lower/upper endpoint commands move inward when required. All16initialendpoint gradient coordinates in the real1024-weight pilot are now nonzero. Analytic gradient, feasible commands, before/after mean laws and128000integer accumulator entries/16.384Mpublic binary observations verified. No reliance on Torch clamp subgradient at the endpoints.

Matched before-condition parameters, codes, inputs, means/variances and500native trajectories/outputs exactly reproduce prior clampedpilot. Same fixed1024realprojectionweights/16capturedpositions/ACT12/block128ranges/200Adamsteps/lr0.03/T16/five100-trial streams. New expectedMSE0.0009438328462 ->0.0001683918097; empirical0.0009293362625 ->0.0001676315096 (~81.96%reduction). Slightly higher final error than prior clampedpilot0.0001674753075 retained; no additional quality benefit asserted. Treatment chosen to permit inward optimization at limits, not for favorable task score. This resolves the component boundary-freeze issue but is not full-model training or inference improvement.

Controlled surrogate uses a biased STE and fixed input positions appearing in its objective. Full-model training integration still requires convolution/normalization/tied-embedding parameterization, finite-command export with fixed range metadata, separate train/validation selection and native evaluation; these are not implemented by this1024-weight proof. No BF16 equivalence, unseen-input generalization, unbiased sampler-gradient or physical benefit claim. Prior clamped component, its initial audit failure and corrected audit remain unchanged.

Full frozen-language cohort remains live in original session50670; latest thirdseed {"stage": "native", "operator_calls": 546, "last_module": "model.layers.21.feed_forward.w3"}. Two accepted native losses4.88353/4.97385 vsBF164.63572; all5 still required for cohort statistics. Word comparison/analysis queues unchanged. Prior/current goal turns progress; broad goal active and global inventory stale during writers.

Runner93899 and independent audit exited0. Archived source hashes: component7d6f2381f8c5a65bfe17ad8f09193b31777462c5563088fc6d1ba69b3d5e70c7; runnerabd1cb2a222c5fd199f5ae19da93ca9e87aa74e2cc2480fef348e9b74bf9d781; auditor74a82225159a2db034f096764c0dbdc9c1fe7fe5de7915945672c113c31477da. No changes to frozen neural-cohort treatments/seed selection.


### 20261029T220907Z — Fullsingle language cohort accepted and fulltensor mapping audited

All five single-p-bit LFM3 language seeds independently verified and Decimal statistical audit passed. All1,728,801,536uniqueparameters/287modules per seed, same frozen8contexts/512predictedtokens, block128ranges/12-bitcontrols/ACT12/T1024. SeedNLLs4.883533826,4.973846934,5.202486257,4.861113882,4.719998370; mean4.928195854, MC95%t4 interval[4.706810152,5.149581556]. BF16NLL4.635723773; pairedmeanNLLgap0.292472081, MCinterval[0.071086379,0.513857783]. SubsetPPLexp(meanNLL)=138.130081 vsBF16103.102514; ratio1.339735, MCinterval[1.073674,1.671728]. Current setting falls short ofBF16 on this fixed subset; no equivalence or unseen-text generalization claim.

All four deterministic controls retained: INT2 NLL10.961896503; INT44.973005522; INT84.615521421; BF164.635723773. Single-minus-INT4meanNLL-0.044809668 withMCinterval[-0.266195370,0.176576034], so no established INT4 advantage from thiscohort. Single-minus-INT8mean0.312674432, interval[0.091288731,0.534060134]. Nominalmean-operator control4.638117449 is close toBF16 but is not expectation of the full sampled nonlinear network or a hard performance bound. Sampling/adaptation remains the main research target.

Each seed processes one batch of8contexts sharing stochastic draws per operator: retained1,772,027,117,568+warmup1,730,495,232scheduled node evaluations includingpadding. Cohort totals8,860,135,587,840+8,652,476,160. These are scheduled evaluations, not measured physical flips or hardware energy; activation accumulation and framework costs excluded. All19statistics input hashes match; all seed outputs/coverage and stableF64tokenlosses independently checked upstream. MCintervals are conditional on512fixedtokens and5seedclusters, not a standardWikiText benchmark or distribution-free generalization interval.

Matched internal four-node positional cohort started automatically after single statistics audit. First seed202610300 verified: nativeT256 NLL4.982921110/PPL145.899950; nominal4.957582526. Retained node work exactly equals pairedsingleT1024; warmup4times separately charged. One of5wordseeds accepted, second202610301 live in session48884; no pairedcohort conclusion yet. Format is internal, not asserted Extropic physical weight representation.

Full-checkpoint probability-training tensor mapping completed and independently audited. CPU parametrization has1,728,801,536FP32trainable controlslots plus frozen originalBF16reference tensors,13,531,430fixedFP32rangegroups,286uniqueencoders/287moduleuses, one shared embedding/head encoder. Every coordinate initial12-bit command and metadata fingerprint checked independently against originalcheckpoint, and all four operator anchors checked against preserved public-command data/meanoutputs. Representative actual parametrized embedding/head gradients both contribute and add correctly. CPU full16-token mean-weight surrogate forward finite; no full-model backward/training/native sampling/ACT12 fullmodel operator parity or systems improvement claimed. Fixedzero-span groups=3kernels/21weights, hence those weightvalues cannotmove with fixedranges. Actual software storage includes originalBF16references,FP32latents andworking buffers; this is not modelcompression.

Prior goal turn continued full tensor implementation and completed CPUforward while cohort progressed; classify as progress. This turn progress: completed independentall-coordinate mapping audit and accepted fullfive-seed/native/statistical evidence plus firstmatchedword result. Broad goal active: finishwordcohort/pairedstats, implement faithful trained-command export and full training/backward/validation, then further sampling/nonidealities/hardware-cost work. Global inventory remains stale while live writers run.

CPU integration72124/audit37912 exited0. New archived sources: probability_tensor.py17bdb55b09855348cf4ab4f895d0e38be0215eb18150a698b7b912c08fd02e03; integrationrunnerb6b3f15ca0ba5aad1accc8d09d7b35d5244431885a203dbe19c01669da27861f; auditor9b3e17749a5b19907674edcb54217d98f00a221d0e52187ac4d7ff4dc1c5ee6a. Frozen neural treatments, all5seeds and controls preserved.


### 20261029T221725Z — Finite-program serialization and directTHRML execution

Finite-command export and direct native execution path implemented and independently verified. Canonical program files contain UINT16codes with12informationbits, fixedFP32sharedlower/span metadata, originalshape andgroupsize. No reconstructed scalarweights or trainablelatent values stored in deployableprograms. Runtime worker accepts codes plus fixedmetadata and explicitly rejects a referenceweight array; native branch accumulates sampled integer outputs before group affine readout, without re-estimating ranges or re-encoding learned commands. StoredUINT16arrays are16storagebits, not packed12bit; no memoryadvantage inferred.

All34CPUofficialTHRMLcases passed and were independently checked offline: linear/convolution/embedding/norm x initial/perturbed controls x T1/T16 x mean/native, plus actual trained1024-weight projection controls atT16. Nine canonical programs,401preserved public trajectorytiles,58,368exactinteger accumulator entries and1,484,544binary observations. Every program code, fixedrange metadata, operatorlayout (including embedding/head views), inputquantization, native/public counter, affine readout and scheduled evaluation receipt checked. Actual trainedprojection codes exactly match the exported trainingpilot aftercommands; fixedranges remain identical. All cases retained. No full-model trained checkpoint export or task-quality claim.

This closes the operator-level serialized-control-to-native path. Full-model training still needs a memory-efficient backward implementation, integration with the intended operator/input-quantization surrogate, optimizer/validation protocol, fulltrained export and independent native quality evaluation. Existing mean-weight Torch parametrization is a surrogate and not native sampling; the512-token T1024cohort remains a negative BF16-quality result (meanNLL4.92820vs4.63572, subsetPPL ratio1.3397). No full-model fine-tuning has occurred.

Matchedfour-node cohort unchanged in session48884; acceptedwordseeds [202610300], currentseed 202610301, latest {"stage": "native", "operator_calls": 568, "last_module": "model.layers.23.ffn_norm"}. Wordanalysis session4809 waits for all5audits. Prior/current goal turns progress; broadgoal active. Global inventory stale while writers run.

Producer83274 and independent audit exited0. Sources archived before execution: program_export.py15adc6485602f8ccbdfc250ef7932265e2f807a3f454e4fad16260531611042f; programmed_worker.py5b2c63ee4a2209a3054d02ae334f4fdd1333b5d5a98d6ed3a44362fa3b0a9642; preflight5f8667524728e702fc7e5528cd52b6cca0dbc8068ae7c7b822b09fc37ae8c94e; auditor251124a5cd1b797658e04f4418c26ff26ecd076eb4064c046113ee3fea767b26. No changes to frozen neuralcohort sources/treatments/seeds or thresholds.


### 20261029T224008Z — Recomputed control gradients and full LFM3 backward

The full LFM3 mean-weight training surrogate completed a CPU BF16 forward/backward probe on 16 tokens from the reserved training split (15 targets). All 1,728,801,536 probability-control slots received finite FP32 gradient tensors; 1,728,678,465 coordinates were nonzero. All 286 unique encoders and 287 weighted modules were covered, with one shared embedding/head encoder. No optimizer step was performed and no fine-tuning quality result is claimed.

A new first-order custom backward recomputes programmed probabilities instead of retaining FP64 probability tensors. It preserves the declared projected identity STE and the finite 12-bit command semantics. Fifteen cases across real tensor kinds, zero ranges, perturbed/end/outside controls and gathered embedding rows passed reference forward/gradient and saved-storage checks; 73,725 gradient coordinates were independently checked by algebra. On the tested mean-weight path, saved state references existing FP32 controls/ranges and adds no transient probability storage. The full CPU probe saved zero FP64 tensor storage for backward. Its recorded saved storage excluding control/range references was 3,523,374,024 bytes; process peak RSS was 28,732,698,624 bytes and runtime was 105.08 seconds. These are software diagnostics for this short probe, excluding optimizer state and longer-context training, not a hardware memory advantage or a measured full-training peak.

The independent probe audit checked canonical coverage, per-parameter finite-gradient records/digests, data/source hashes, saved loss arithmetic and memory-record sums. It does not independently rerun the full backward or reconstruct the unpersisted 6.9 GB gradient arrays. Component gradient algebra was audited separately. The diagnostic training loss reproduces as FP32 7.311960697 and recomputes as FP64 7.312086535; the precision distinction is retained. This uses different train data from the 512-token inference subset and is not comparable as a quality result. Full-model ACT12 operator training parity, stochastic training and native-trained quality remain unproven.

The matched internal four-node cohort now has two independently audited seeds: NLL 4.982921110 and 4.960669046 at T256. Each exactly matches single T1024 retained scheduled work, with four times the warmup separately charged. Third word seed 202610302 is running under the original process 45530; the old tool handle is unavailable, so OS identities are used without restart. All five single-p-bit seeds remain closed and audited (mean NLL 4.92820 versus BF16 4.63572, subset perplexity ratio 1.3397). No paired cohort conclusion before all five word seeds.

The backward integration gate has passed for the mean-weight surrogate. Next: optimizer/finite-command update checks, a prospective probability-aware training and validation protocol with same-fit deterministic controls, and full trained-program native evaluation. No frozen inference treatments or seeds changed. Previous and current goal turns are progress. The broad research goal remains active; the global inventory is stale while writers run.

Component runner 28834 and its audit exited 0. Full CPU probe 58986 and audit 14983 exited 0. Sources archived before execution: recomputed_probability.py 0de26f5f48db9e3fbdbdbf004d5dbfe14e2db3a73f4470382d4d8246d6e87241; component runner a97ef689787ff5bb1c7a22ede61a6c5439a1a44aa209ded6481c6253ced2c74a; component auditor 371bfa65c859ff0e648e801acc4f82a96b86b5d9563eaec110c9dfa9014278f5; full probe 2825e33341fdf3edbf172873a779709e44018fea7c83b5c6a96760e2f841574d; full auditor 0c36fb360faa738ed698ee3e74eeb82271dfba875bcf6699ea473059d87877fe.


## Full-model optimizer and finite-program export verified — 20261029T225806Z

The full LFM3-1.7B mean-weight surrogate completed one actual CPU Adam update on 15 reserved training targets, followed by export of all 286 unique canonical finite-command programs. The independent verifier replayed first-step FP32 Adam arithmetic on all 1,728,801,536 saved raw gradient coordinates from checkpoint-derived initialization. Every updated latent hash, exported command, fixed range value, gradient/file digest and shared embedding/head alias matched. All 286 programs changed; 1534266482 individual commands changed. Exported programs retain UINT16 command arrays with 12 information bits and FP32 shared lower/span metadata. This is an engineering update/export result, not meaningful fine-tuning or quality evidence. The snapshot is deployable commands, not a resumable optimizer/latent checkpoint.

The 15-target diagnostic loss fell from FP32 7.311960697 to 2.005464315 on the same training example; this does not establish unseen-text improvement. Stable FP64 loss checks are retained separately in the audit. Observed CPU process peak RSS was 39,642,394,624 bytes. The surrogate uses BF16 mean weights, declared projected first-order STE and fixed ranges; full-model ACT12 training parity and native trained quality remain unproven. Raw gradients are retained, enabling the independent all-coordinate update replay without rerunning the full backward.

The matched four-node T256 comparison has three independently audited seeds (native NLL 4.982921110, 4.960669046, 4.986387788), with seed 202610303 running under the original pipeline. No five-seed paired conclusion yet. The closed single-p-bit T1024 cohort remains mean NLL 4.92820 versus BF16 4.63572 (subset perplexity 138.13 versus 103.10); current frozen conversion does not match BF16 quality. Those five runs cover all learned parameters on a fixed 512-target subset, not a full benchmark. No physical device residency, hardware advantage or BF16 equivalence claim.

Next: a prospective probability-aware train/validation protocol with matched same-fit controls, followed by full native execution of adapted programs. Existing inference artifacts remain unchanged. The broad goal remains active; this turn completed the full update/export engineering gate. The global inventory remains stale while live experiment writers run.

[Independent optimizer/export audit](lfm3_1p7b/optimizer_export_probe/independent_audit.json).


## ACT12 probability operators verified; full-model adaptation running — 20261029T231358Z

A new differentiable operator path retains FP64 probability-derived affine values inside each operator, applies the native ACT12 input quantizer, and rounds only operator outputs to BF16 before original nonlinearities. This resolves the earlier training-surrogate mismatch of rounding mean weights to BF16 before multiplication. Expected execution and nominal IID binomial aggregate counts are supported; quantized controls use the declared projected expectation STE and activation quantization uses an identity STE. These are biased training surrogates, not differentiable calls to THRML.

All 34 operator cases passed independent verification against preserved official THRML public trajectories/mean outputs, including actual trained projection commands. Seventeen cases reuse native public counts; 198,480 control-gradient coordinates and input gradients checked by independent NumPy algebra. ACT12 integers match exactly; FP64 affine readouts meet the predeclared 2e-12 tolerance. Nominal binomial moments at T1/16/1024, 5,000 trials per probability, agree with theory under predeclared six-standard-error bounds (maximum observed mean 2.004 SE, variance 2.314 SE). This establishes component forward/declared-gradient laws, not unbiased gradients, full-model native equivalence or language improvement. A small integration fixture also exercised tied embedding/head, convolution, normalization and linear modules at mean/T16/T1024 with finite full gradients.

The full-parameter prospective adaptation pilot is now running under frozen protocol SHA256 481843a72f7d4736be09fa51d8c17a1cfcbd1f22564397f023bfcfe1a814d900. Two trained arms: expected operators and binomial T1024; each gets 16 Adam updates at lr0.0001 on the same first16 reserved training windows truncated to65tokens (1,024 targets per arm). All1,728,801,536controlslots/286encoders/287modules included, tied controls shared. Validation uses all32 separate65-token windows (2,048 targets), four batches of8, mean control plus five fixed binomial seeds. Original unadapted BF16 and untrained probability baseline are retained as separate candidates. Selection uses only the predeclared validation scores and fixed endpoints; no test-based tuning or adaptive training-budget selection. This is a small adaptation pilot, not convergence or a full benchmark.

Every full FP32 update gradient is retained. Final controls and Adam moments/steps/RNG state will accompany deployable finite-code/fixed-range programs, allowing independent full-coordinate multi-step replay. The chunked audit arithmetic was independently checked bitwise against actual Torch Adam for16syntheticsteps including zeros, small gradients, scale changes and boundary projection. A separate live audit queue verifies producer PID/starttime/argv while waiting and audits each completed arm. Producer completion and audit acceptance are tracked separately. No automatic retry. Native trained-program quality evaluation is still a subsequent implementation/execution step; it is not queued by this producer/audit scheduler.

Current producer arm is mean; 4 optimizer-update records are complete and current stage is {"stage": "training_forward_backward", "elapsed_seconds": 379.366185421939, "step": 5, "samples": 0}. The initial expected-operator validation loss is4.985681076 on2,048reservedtargets; BF16 and stochastic validation controls for this new split are not yet complete, so no quality comparison is accepted. Full-model forward coverage assertions passed across all287weightedmodules. Original simulator word cohort remains live, three audited seeds, fourth running. Frozen singleT1024 five-seed512-target loss4.92820 vsBF164.63572 remains negative for BF16 quality; no prior treatment changed.

Previous and current goal turns are progress. Broad goal active, with probability adaptation/native validation, wider model/data assessment, nonidealities and system-cost conclusions still outstanding. One logical p-bit per learned weight does not imply onebit total information or simulator residency. Actual software retains original weights, FP32 latents/optimizer/gradients and finite controls/ranges. Global inventory remains stale while writers run.

[Operator audit](../../results/thrml_spw/lfm3_1p7b/probability_operator_preflight/independent_audit.json) · [Prospective protocol](../../results/thrml_spw/lfm3_1p7b/adaptation_pilot_protocol.json).


## Native adapted-program validation frozen and queued — 20261029T232917Z

The mean-operator full LFM3 adaptation arm now has 16 completed update records of its fixed16-stepbudget. Current stage: {"stage": "validation", "elapsed_seconds": 1297.1164249340072, "label": "after_mean", "batch": 1}. All sixteenfullrawgradients will support independent optimization replay; final286canonicalprograms plus resumable FP32control/Adam state are being exported. Validation and independent whole-arm acceptance remain incomplete, so no trained quality improvement is accepted. The binomialT1024training arm and untrained/BF16validation producer remain queued by the original pilot scheduler.

The matched internal four-node cohort has four independently accepted seeds. Fourthseed202610303 native NLL4.971377337715307/PPL144.225398, with retained work exactly matched to paired singleT1024 and fourtimeswarmup separately charged. Fifthseed202610304 is running under the existing original pipeline. No five-seed paired conclusion until its audit and statistics audit complete.

A separate direct-program native validation study is implemented, prospectively frozen and actually queued. Protocol SHA256 a4f847aa5453c94768c74b288b62d68f31e5bc4c3b899ee431cf6d18f9897fe6 was frozen before any adapted validation result. All three candidates (unadapted, expected-trained, binomial-trained) will be reported atT1024 on all32reservedvalidationcontexts/2048targets, with five new nativeMCseeds202610350..354. It preserves all candidate/seed failures and negatives. Candidate selection is made from the prior five-seed Torch-binomial validation scores before native evaluation, with baseline winning exact ties; native results for all arms remain mandatory. This uses the same validation data for transfer assessment, not an independent confirmatory test.

Queue gates are explicit: verify the existing word process identity while waiting for that GPUcohort; run34direct-code GPUoperator/public-trajectory cases and independent audit; verify the adaptation-audit process identity until all three producer/audit arms pass; require bitwise reproduction of allBF16/nominal/native logits from the original unadapted202610300512-target run through the new serialized-program path; then evaluate/audit all15nativecandidate/seed combinations. Full learnedparameter coverage, shared embedding/head commands, every program hash, all peroperator node-work/padding/warmup receipts, and independent saved-token loss arithmetic are required. Direct programmed_worker receives codes/fixedranges and no referenceweight payload; ranges are not refitted. Fixed call/tile keys couple common uniforms across candidates. Original deterministic GPUoperators also execute before hooks; extra work and simulation time are not hardware-performance evidence.

The exploratory quality gate is an upper two-sided95percentt4 MC bound on native-minus-BF16 NLL no larger than log(1.02), corresponding to a2percentperplexity margin on this validation subset. All per-arm native-vs-BF16, native-vs-ownexpectedoperator and pairedtrained-vs-unadapted contrasts will be retained. Statistics have independent Decimal arithmetic and input-hash verification code. This gate does not establish numerical BF16 equivalence, data-generalization coverage, a standard benchmark, multiple-comparison-confirmatory significance or hardware advantage. A fresh-data study remains necessary if it passes. The runtime/metadata/controller costs are still explicit; logical one-p-bit assignment is not physical simulator residency or onebit total storage.

The native scheduler is live and waiting on verified upstream processes; no native adaptation quality run has started yet. Source snapshots, pinned data/model/source hashes, clean official THRMLcommit2d54cd8404df85bb65703d7e8b4e84e48c1b72a9 and frozen protocols are retained. GPUpreflight/initialreplay are new gates, not assumed passed. Original frozen five-seed singleT1024 loss4.92820versusBF164.63572 remains unchanged and does not matchBF16 quality. Prior and current goal turns are progress; broad goal remains active. Global evidence inventory remains stale during live writers.

[Native protocol](../../results/thrml_spw/lfm3_1p7b/adaptation_native_protocol.json).


## Same-fit BF16 comparison strengthened before native inference — 20261029T233851Z

All16full-model mean-operator training updates and all286finite-command/fixed-range programs plus resumable FP32control/Adam states are produced. Stochastic validation is still running and independent full-arm optimization/quality acceptance is pending. Preliminary expected-operator CPU validation NLL on2048separate targets is4.985681076before vs4.302017009aftertraining (PPL146.303185vs73.848597); these are producer-only values, not yet audited and not native sampling. This is ordinary adaptation evidence pending audit, not representation equivalence.

The preliminary mean improvement makes an exact same-fit BF16 reference essential. The earlier V1native protocol reported same-fit expected operators but placed its2percentquality gate against the original unadapted BF16 model. That gate could mix the benefit of extra training with representation accuracy. Before any GPUpreflight/native inference began, the waiting V1scheduler was deliberately terminated (verified PID143537/startticks63090515, tool exit143); its sources, protocol, waiting-state snapshot and explicit supersession receipt remain unchanged. No training run was interrupted, and no candidate, seed, sample count, optimizer setting or quality tolerance was changed.

Native V2protocol SHA256 ce5aa89eb8d447097b0ab05b4b49fa27c73284950327e20922712b51fe41f286 adds a conventional BF16 control using each candidate's exact learned expected weights. Decode the12-bitfinitefields, compute fixedlower+span*sigmoid(2field) inFP64, round each weight toTorchBF16, and use the exact original neural operators without ACT12quantization or sampling. Original frozen weight pointers are temporarily exchanged only for these control forwards and restored after every weighted operator; tied embedding/head use the same cached control weight. Every BF16 control weight hash is checked independently. The new quality gate is native minus this SAME-FITBF16 NLL upperMCbound<=log(1.02). Comparisons with original unadapted BF16 and with the same-fit ACT12expectedoperator remain separately reported. Passing this conditional validation gate still does not prove numerical BF16 equivalence or fresh-data generalization.

The control implementation passed a CPU BF16 integration fixture: fiveactualweightedmodules/fouruniquetensors, tied embedding/head, actualpaddingindex31, normalization, convolution and linear projection. Two same-fit forwards bitwise match a separately weight-assigned reference model, original parameters are bitwise restored, and interleaved original forwards remain identical. This verifies the pointer-swap/control mechanism on a small fixture; fullLFM3/GPU control validation is pending. No custom reimplementation of the backbone operator kernels is substituted for the BF16 control.

The V2scheduler is actually live, waiting on verified word-cohort and adaptation-audit process identities. It retains all3candidates x5newnativeMCseeds x32validationcontexts/2048targets, the34-caseGPUpublic-trajectory preflight, strict original unadapted-run bitwise replay, all-coordinate optimization audit gates, direct-code native worker, exact metadata/program hashes, four conditions (originalBF16/samefitBF16/nominalACT12/nativeACT12), and independent loss/resource/Decimalstatistics audits. All negative outcomes and candidate results retained. The V1plan is superseded before inference; V2is the authoritative active native study. No adapted native quality result has been observed or accepted.

The original matched four-node cohort has four accepted seeds, fifth202610304running. Original five-seed singleT1024cohort loss4.92820 vsBF164.63572 remains unchanged. Both previous and current goal turns are progress. Fullgoal active; native transfer, same-fit quality, fresh-data assessment, wider nonidealities and system-cost conclusions remain outstanding. Global inventory stays stale during live writers.

[V2nativeprotocol](../../results/thrml_spw/lfm3_1p7b/adaptation_native_v2_protocol.json) · [Control fixture](../../results/thrml_spw/lfm3_1p7b/same_fit_control_preflight.json) · [V1supersession](../../results/thrml_spw/lfm3_1p7b/adaptation_native/superseded_before_inference.json).


## Matched spatial comparison closed; GPU gates and unscored holdout verified — 20261029T235904Z

The full five-seed matched retained-work LFM3 comparison is complete and independently audited. All1,728,801,536uniquelearnedweights/287weightedmoduleuses per run, same512frozenpredictiontargets, singleT1024 versus internal positional four-nodeT256. Single meanNLL4.928195854/PPL138.130081; four-node meanNLL4.976253122/PPL144.930327. Paired single-minus-four NLL=-0.048057268 withMC95%t4interval[-0.267479699,+0.171365162]; PPLratio0.953079 withinterval[0.765306,1.186924]. The interval permits either method to be better; no superiority or equivalence established. Against originalBF16NLL4.635723773/PPL103.102514, single PPLratio1.339735[1.073674,1.671728] and four-node1.405691[1.387769,1.423845]. Both current frozen conversions fall short ofBF16 on this subset.

AllfivewordseedNLLs retained:4.982921110,4.960669046,4.986387788,4.971377338,4.979910330. Allninecontrasts and resource totals checked by independent Decimal arithmetic; all statistics input hashes rechecked now. Each method uses8,860,135,587,840retainedschedulednodeevaluations overfive runs, includingpadding; warmup single8,652,476,160 versusword34,609,904,640. These are evaluation counts, not measured flips/energy. Logical fullyresident assignmentsN versus4N do not describe actual simulator residency. Control payloads differ: single12-bitfieldcommands versusfourtwo-levelsignchoices, so conditional packedstate+control+FP32range payloads are2,917,553,936B versus1,837,052,976B. Fixed sampling latency under equalrate/fullparallelism would be4timeslonger forsingle; no systems advantage inferred. The four-nodeformat is internal, not an asserted Extropic hardware weight encoding. Prior K8scalar negativecontrol (95.55timeslowerMSE than single atC1024) remains intact.

A PNG/SVG comparison now shows every paired seed, audited absolute and pairedMCintervals, BF16/INT4references, retained-work normalization, control precision and conditionalscope. PNG visually inspected: all points/labels/intervals/caveats visible without observed clipping/overlap. This is a single-budget comparison, not a Pareto frontier. Intervals condition on512fixedtargets, fiveMCclusters and fixedcheckpoint/settings; they do not describe unseen-data or training-fit uncertainty.

The direct serialized-program GPU preflight passed all34cases and independent audit:9programs,401publictrajectorytiles,58,368integercounterentries and1,484,544publicbinaryobservations. Four tensor kinds, initial/perturbedcontrols and actual trainedprojectioncommands; exact integer/codes/layout/readout/resource checks. The same-fit BF16control mechanism additionally passed a GPU fixture on NVIDIA GB10: fiveweightedmodules/fouruniquetensors, tiedembedding/head, paddingindex31, normalization/convolution/linear, two repetitions, bitwise agreement to a separately weight-assigned BF16model and bitwise restoration of original weights. These are component/gate proofs, not fullLFM3trained quality.

A larger unscored LFM3 holdout was reserved and independently verified:10,240predictiontargets across40articles. Short cohort:96windows of65tokens, fourwindows each from24articles,6144targets. Long cohort:16windows of257tokens from16differentarticles,4096targets. Both exclude all7articles touched by the8consumedLFM3testwindows, and any exact65-token context from all256reserved257-tokentrainingwindows,32validationwindows,legacytestwindows or prior selectedholdoutwindows. Cohorts have disjointarticles and contained nonoverlapping windows. Global pinnedtesttokenstream/offsets, articleboundaries, source/tokenizer hashes and complete seeded reservation replay independently checked. No model scores computed on these heldout windows. Articlebalanced sampling is not fullWikiText perplexity, and the same corpus was used in other-model studies; no pretraining/phrase-level decontamination or pristine-corpus claim. Cohorts are separate robustness assessments, not a controlled causal context-length ablation. Evaluation treatment/statistics must be frozen before consuming their scores.

Current adaptation producer/audit/native queues remain live and verified byPID/startticks/argv. Mean arm has all16updates/286exports produced; current validation stage {"stage": "validation", "elapsed_seconds": 3029.7004681410035, "label": "after_binomial_202610344", "batch": 3}. Whole-arm acceptance and binomial-training arm remain pending. NativeV2scheduler passed both GPUpreflight stages and now waits for all three adaptation audits before strict original-run replay and all3candidate/five-seed2048-target nativevalidation. No adapted native quality result or BF16same-fit equivalence is accepted. Previous and current goal turns are progress; full research goal active. Global evidence inventory remains stale while large writers run.

[Paired audit](../../results/thrml_spw/lfm3_1p7b/word_language/statistics_audit.json) · [Comparison figure](../../results/thrml_spw/lfm3_1p7b/word_language/figure/matched_work_language.png) · [Holdout audit](../../results/thrml_spw/lfm3_1p7b/holdout_data/independent_audit.json) · [GPU direct-program audit](../../results/thrml_spw/lfm3_1p7b/program_export_gpu_preflight/independent_audit.json).


## Full expected-training audit passed; stronger K8 neural baseline live — 20261030T001409Z

The full LFM3 expected-operator adaptation arm has passed independent audit. Every one of1,728,801,536control coordinates was replayed through all16CPUFP32Adam updates from pinnedcheckpoint initialization and retainedrawgradients:27,660,824,576coordinate-updates total. Every final latent/firstmoment/secondmoment/code/fixedrange and sharedalias matched. Changed finitecommands=52,610,250 (~3.04percent); all286programs/finalcontroloptimizercheckpoints retained. All savedbefore/after/5sampled validation logit losses independently recomputed. This does not rerun the fullmodelbackward, trainingdraws or nativeforward; component STE laws checked separately.

On the separate2048-targetreservedvalidation split, expected-operator NLL4.985681076 before vs4.302017009 after16updates/1024trainingtargets is now audited. Post-fit nominalTorchIIDbinomialT1024 validation NLLs4.395606292,4.387999000,4.416233808,4.388075420,4.378880478; mean4.393358999635/exp(mean)perplexity80.911745556. These are surrogate results, not officialTHRML native scores or matchedBF16 equivalence. Unadapted stochastic/BF16controls on this validation split are still queued, and learnedsamefitBF16/nativecomparisons remain outstanding. Do not compare these scores with the older512-targettestscores as a quality change. Extra-training benefits must be distinguished from representation error.

The binomialT1024 training arm is live under the original unchangedpilotprotocol; current stage {"stage": "training_forward_backward", "elapsed_seconds": 821.9836512201, "step": 7, "samples": 1024}. Meanarm accepted, otherarms pending. NativeV2scheduler has completed directprogramGPUpreflight/audit and waits allthreeadaptationaudits before strict originalunadaptedreplay and all3candidate/five-seednativevalidation against SAME-FITBF16. No native adapted quality outcome yet. Originalfrozencohort singleT1024NLL4.92820 vsBF164.63572 and single-versus-fourpairedCIcrossingzero remain unchanged.

A stronger spatial baseline is now running, prospectively frozen before any K8fullmodel quality score (protocolSHA2566718330634708838ab525bdeb30989a9689e4bd41673c4a104e28465b6e1279c). Internal eight-free-p-bit positionalword withcoefficients2**k/255,nearestaffineINT8target,fieldlevels+4/-4 only, sameACT12/block128/FP64readout/BF16nonlinearities. K8T128 matches K1T1024 and K4T256 retainedschedulednodework exactly; warmup1:4:8 separatelycharged. All1728801536uniquelearnedweights/287moduleuses, sharedembedding/head, all5pairedMCseeds202610300..304 andsameconsumed512-targetsubset. No10,240-targetholdout scores consumed. The preexisting scalar result favorsK8 by95.55timesMSE atC1024; neuralquality is not assumed. This control tests whether that strong precision advantage survives fullneuralexecution; it is not an assertedExtropicphysicalweightformat.

K8GPUpreflight passed all16operatorcases with independent offline sign/count/int32accumulator/affinereadout audit:1,600publicbittrajectorytiles,229,376integerentries,636,928programmedsigncoordinatesincludingpadding. Fields remain+/-4, twolevels/node; no hiddenhigherprecisionperbit. Fullcohort producer/audit pairs now run sequentially, followed by all3methodstatistics:3pairwisecontrasts plus12deterministiccontrolcontrasts and independentDecimal/resourceaudit. Equalretainedwork, logicalassignments1:4:8, controllerinformation andsameFP32rangemetadata explicitlyreported. Singlecommands12bits/weight versusK4four/K8eight signbits; onepbit is not onebit totalinformation. Conditionalpackedstate+command+metadata budgets are included. IndependentGPUjobs mayoverlap lateradaptedvalidation; simulatorwalltimes are not compared as hardwarelatency/energy.

Current K8pipeline {"status": "running", "pid": 194819, "protocol_sha256": "6718330634708838ab525bdeb30989a9689e4bd41673c4a104e28465b6e1279c", "completed_seeds": [], "current_seed": 202610300, "stage": "202610300_word8_language", "child_pid": 194820}. All failures/negatives retained, no automaticretry or score-basedstopping. Frozen originalsources/protocols untouched; newworker/producers/verifiers archived beforeexecution. Previous and current goal turns are progress. Fullresearchgoal active; BF16equivalence, nativeadaptationtransfer, freshheldoutassessment and broadsystem/nonidealityconclusions remain unproven. Global evidence inventory stale during livewriters.

[Meanarm audit](../../results/thrml_spw/lfm3_1p7b/adaptation_pilot/mean/independent_audit.json) · [K8protocol](../../results/thrml_spw/lfm3_1p7b/word8_language_protocol.json) · [K8public audit](../../results/thrml_spw/lfm3_1p7b/word8_preflight/independent_audit.json).


## Layer calibration and sample-budget candidates verified; T8192 counter edge passed — 20261030T010845Z

Full-model layer calibration passed independent audit at the audited mean-trained LFM3point. Two batchesof8 from reserved TRAINwindows16..31, first65tokens (1024targets), outside fittingwindows0..15 and outsidevalidation/test. No model update. All287weightedmoduleuses perbatch captured separately, including independentlydrawn embeddinglookup andheaduses ofthesame canonicalparameter. All3,458,854,400savedFP64weightgradient values were checked against finiteprogrammetadata/probabilities for the scalarvariance algebra, file/content hashes, input/outputlosses, node/paddinggeometry andcandidatebudget. Originalcontrol/command hashes remained unchanged. Smallinstrumentationfixture alreadyproved bitwiseunchanged outputs/allcontrolgradients and independent NumPyvariance laws.

The proxy is V_l=sum_i(g_w_i**2*span_i**2*p_i*(1-p_i)); sum_lV_l/T_l estimates FIRST-ORDER STEbatchloss variance underindependentdraws. It is not the actual nonlinear lossvariance, curvature-aware expectedcrossentropy increase, accuracy or representationequivalence. Fixedlookup/max4096greedycandidate proxy ratio0.614537248 vsuniformT1024 (~38.55percentvariance reduction), with cap1,772,040,749,056nodeevaluations andcandidate1,772,024,233,984; unusedcap16,515,072 (0.00093198percent). It does not establish a sampledqualitygain orglobaloptimality.

Calibration assigns27.25percentofproxycoefficient toembeddinglookup, which was deliberatelyfixed1024in thefirstcandidate to simplify input-dependentbudget accounting. A separate postprocessing candidate now allowslookup andsensitivelayers upto8192samples, lookupminimum1024/otherminimum16. It uses a conservative lookupupperboundM=1028tokens*2048weights/token; allotherpadding-inclusive nodecosts fixed. Given TE>=1024 and m<=M, the conservativebudget guarantees actual_alloc-actual_uniform <=(m-M)*(TE-1024)<=0. Independentgreedy/proxyreplay andeverydistincttokencount0..1028checked. Thus8x65 or4x257inputbatches staywithin theuniform1024node-work cap. This is a budget inequality, not exactmatchedwork. Alluniform/fixedlookup/extendedcandidates retained; lowest calibrationproxy chosen withoutvalidation/native/testscore selection. Existinguniformstudies unchanged.

Selected boundedlookup/max8192candidate proxy ratio0.347871781 (~65.21percentlowervarianceproxy), lookup8192samples, modulehistogram{"256": 1, "512": 55, "1024": 42, "2048": 41, "4096": 16, "8192": 132}. This still predicts only a first-order sensitivity quantity. Serialized-operator samplecountsum is1302272versus293888uniform (~4.431times); no measured latencyimprovement orphysicalsystemsadvantage claimed. More samples on cheap/sensitive modules can trade longer sampling latency for lower modelednoise atsimilar totalnodework. Nativequality/Paretoevaluation remains necessary, with the same learnedweights/metadata and SAME-FITBF16reference. No adaptive-nativequality experiment has been executed or scheduled yet; next step is to freeze that evaluation and test the allocation natively.

The T8192direct-program GPUintegeredge prerequisite passed independently: two all-positive/all-negative ACT12conditions, fanin128, eightreal/64paddedoutputs, finite+4fields, samekey/publicstates.134,217,728publicbinaryobservations and128integercounterentries reconstructed with independentint64arithmetic. Countermagnitudes reached2,145,753,421, belowtheoreticalACT12bound2,146,435,072 andint32max2,147,483,647; readouts/padding/work/counterdtypes andexactsignmirrorschecked. This validates the genericworker at8192sample integeredge, not fullneural allocation quality orhardware performance.

Both16-update LFM3trainingarms have completed their optimizer updates andprogram/stateexport. Meanarm fullyaccepted; stochasticarm currently{"stage": "complete", "elapsed_seconds": 4121.487785391044}andwholearmauditpending. NativeV2queue stillwaitsallthreeproducer/auditconditions beforestrictunadaptedreplay andthreecandidatefive-seed2048-targetnativeevaluation. K8cohort hasoneacceptedseed, secondrunning; allfive remainmandatorybeforeprimarythree-methodclaims. FirstacceptedK8nativeNLL4.613043742/PPL100.790463 versusitsnominal4.608252133 is an individualdescriptive resultonly; no superiority/equivalenceconclusion fromone run. Allpairedscheduledwork matches priorK1/K4, warmup8timesK1 separatelycharged. Existingfive-seedK1missvsBF16andK1/K4paireduncertaintyunchanged. Ten-thousand-target article-separatedholdout remainsunscored.

Currentturn andpreviousgoalturn areprogress. Broad researchgoalactive: actualnativeallocatedquality, fulladaptationnativecomparisons, freshdataassessment andphysical/system/nonidealityconclusions remainopen. No BF16equivalence orwholeprojectcompletion asserted. Globalinventory staysstale duringlarge livewriters.

[Calibration audit](../../results/thrml_spw/lfm3_1p7b/layer_calibration/independent_audit.json) · [Budget audit](../../results/thrml_spw/lfm3_1p7b/layer_allocation_candidates/independent_audit.json) · [T8192native audit](../../results/thrml_spw/lfm3_1p7b/sample8192_preflight/independent_audit.json).


## Allocated native comparison frozen and queued — 20261030T012144Z

Both 16-step adaptation arms now passed independent all-coordinate Adam/program/validation audits. The binomial arm's saved Torch surrogate validation NLLs are [4.446066180831667, 4.439288519015782, 4.497245812009234, 4.408095450464138, 4.468698233133838]. These are not native THRML quality or same-fit BF16 comparisons. The baseline producer remains live, and its audit is still required by the existing uniform V2 pipeline. No adapted native quality conclusion yet.

A separate allocated-native study is prospectively frozen (protocol SHA256 516a1844c4aea81681dabebbae0b0e6478da41806940ba8052b88e9e661c02f3) and queued behind the strict uniform V2 original replay audit. Mean-trained program point and all fixed commands/ranges unchanged. Exactly the audited training-only bounded_lookup_max8192 sample policy is used, irrespective of native outcomes. All 32 reserved validation contexts / 2048 targets, five paired seeds 202610350..354, all four controls and all 287 operator uses / 1,728,801,536 unique parameters are retained. Native per-module budgets use 256..8192 in this selected map; nominal controls and the uniform comparator use 1024. T8192 public integer-edge audit is a mandatory prerequisite.

Per-seed independent checks reconstruct every saved token loss, command/reference hashes, tied aliases, same-fit BF16 weight hashes and module/padding work. Final paired comparison waits for all five uniform mean-arm audits, then checks exact saved input/BF16/same-fit-BF16/nominal arrays across methods, identical commands/call identities and warmup, allocated actual retained node work <= corresponding uniform work for every batch. Five-seed paired t4 NLL interval and same-fit BF16 upper-bound gate <=log(1.02) are reported for both methods without score-based stopping. This is exploratory conditional validation, not fresh-data generalization or numerical BF16 equivalence. The 10,240-target article-separated holdout remains unscored. The allocation's serialized-operator sample-cycle proxy is 1,302,272 vs uniform 293,888 (4.431x); no latency improvement is claimed, and concurrent simulator job wall times are not hardware latency.

The paired-analysis fixture passed in isolated temporary directories: known positive and negative quality-gate outcomes, five-seed paired arithmetic, rejection of altered summary arithmetic, and rejection of mismatched deterministic control arrays despite rehashed input receipts. Synthetic fixture data are not model evidence. New source files archived by content hash before execution; existing uniform/training/K8 sources unchanged. Pipeline PID 640365, start_ticks 63787251, confirmed live, waiting for strict replay. Tool session 83856. No automatic restart. Current goal turn is progress: new tested, frozen and running scheduler/state. Previous user-facing status turn inspected newly accepted binomial audit; no new model execution was launched in that status turn. Full research goal remains active and incomplete. Global evidence inventory remains stale during live writers.

[Allocated protocol](../../results/thrml_spw/lfm3_1p7b/allocated_native_protocol.json) · [Synthetic analysis fixture](../../results/thrml_spw/lfm3_1p7b/allocated_statistics_preflight.json) · [Binomial training audit](../../results/thrml_spw/lfm3_1p7b/adaptation_pilot/binomial/independent_audit.json).


Verified wait 20261030T012412Z: OS identities and CPU/progress state rechecked for adaptation producer139744/baseline child621078, audit queue140819, uniform native queue166038, K8queue194819/second seed frontend564415, allocated queue640365. Baseline validation advanced from seed341 batch1 to batch2 during this turn; K8secondseed native last operator lm_head completed, results/audit still pending at observation. No restart or protocol/source mutation. Prior goal turn was progress (new frozen tested allocated study); this turn is a verified wait on these specific live processes, not a blocker. Full research goal remains active. Receipt: results/thrml_spw/cifar_cnn_native/execution_checks/20261030T012412Z_LFM3_verified_wait.json.


## Second K8 control accepted; one-p-bit target unchanged — 20261030T012554Z

Second K8seed202610301 passed independent audit and root seal review: all1,728,801,536uniquelearnedweights/287modules, nativeNLL4.600881865781805 onthefixed512targets, exactBF16/firstseednominalcontrols, retainednodework1,772,027,117,568 exactlypairedsingleT1024/fourT256; K8warmup13,843,961,856 chargedseparately. Rootchecked frozensource/prerequisitehashes,audit/resultsbindings,coverage,tokenmetricagreementandpairedworkreceipts. Savedlogits/fulltokenlossesalreadycheckedbyindependentverifier; no duplicatefullnative/publictrajectoryreplay. Thirdseed202610302child648344/start63809261 confirmedlive. Only2of5accepted; no comparativecohortorBF16qualityclaim.

Theuserreconfirmedonepbitperparameter. Itremains theprimarytarget for everyuniqueLFM3parameter, includingembedding/headsharedcoordinatesandnormweights. Mean/stochasticadaptationadjustprobabilities,allocationadjustssamplebudgets; neitheraddsweightpbits. K4/K8arecomparisoncontrolsonly. Existingtraining/uniform/allocatedprotocolsandjobsunchanged. Bothfittedtrainingarmsfullyaudited;baselinevalidationanditsauditstillprerequisitesforuniformnative/replay;allocatedstudywaitsstrictreplay. Preparationstatus stale nextsteps/limitations correctedandpriorstatepreservedinstatus_history. Fullresearchgoalactive; noBF16equivalenceproved.

Previoususerclarificationturnwasanswer-only(nooperationalprogress); thisturnrevalidatedspecificliveprocessesandacceptedthecompletedsecondcontrolaudit, updatingauthoritativeevidenceandnextactions. Receipt:results/thrml_spw/cifar_cnn_native/execution_checks/20261030T012554Z_LFM3_second_K8_seed_review.json. Globalinventoryremainshistorical/stalewhilewriterslive.


## Article and MC uncertainty component verified without holdout scoring — 20261030T012927Z

The current one-p-bit LFM3 native studies remain unchanged and live. Baselinevalidationadvanced toTorchMCseed342/batch2duringthisturn; remaining baselineaudit, strictnativeoriginalreplay and adaptednativecohorts are pending. K8controlcohort has2of5acceptedseeds andthirdlive; no newcohortqualityclaim. Previousgoalturn wasprogress(secondK8audit/rootreview); thisturnisprogress: implemented/validated the uncertaintycomponent needed before later article-separated assessment, whilepreservingcurrentstudies.

New pure arithmetic module holdout_uncertainty.py groups five inference-seed clusters,24shortarticleclusters and16longarticleclusters. Shortarticles containfour64-targetwindows; longarticles one256-targetwindow. Eacharticlecontributes256predictiontargets, so pooledtokenweightis60percentshort/40percentlong. Allwindowswithinanarticle staytogether; bootstrap resamplesarticleswithinlengthstrata, andoneglobal seed-row sample issharedacrossallarticles/bothstrata. Paired candidate-minus-reference tokenNLLs mustbeformedbeforeaggregation. Itreportsthestrata/pooledgap,conditionalMCt4interval, crossedstratifiedpercentilebootstrapintervalandexp(NLLgap)PPLratio. No treatmentselection, pass/failgate or fullassessmentprotocol chosen. Intervalsareapproximate diagnostics, notformalgeneralization/coverageguarantees; lengthgroupsare differentarticles, notacausalcontextlengthcomparison.

Syntheticfixture independentlyreplayedall64replicateindexsets/clusterarithmetic andtokenaggregation; testedcommonglobalMCnoise, article-onlyvariance, zero-gaps, invalidshapes/nonfiniteinputs. Commonseedvariance observed0.3971745711 vstheory0.4; article-onlypooledvariance0.933210891 vstheory0.93125. Thesearearithmeticchecks, notstatisticalcoveragecalibration. Separatemetadata-onlygeometryauditverifiesthereservedmanifest/IDs/offsets,theexactconsecutivefour-short-windows grouping, withinarticlebounds, disjointarticleIDs and40x256=10240targets. No tokenarrays,modelforward,predictionsorreservedtaskscoresloaded; archivebyteshashedonly. Holdout remainsunscored. Allthree newsourcesarchivedbeforeexecution; existingfrozenprotocols/sourcesuntouched.

[Uncertainty fixture](../../results/thrml_spw/lfm3_1p7b/holdout_uncertainty_preflight.json) · [Actual geometry audit](../../results/thrml_spw/lfm3_1p7b/holdout_uncertainty_geometry_audit.json). Fullresearchgoalactive; onepbit-per-parameterBF16qualityremainsunproven andnativeadaptedtransferpending. Globalinventoryremainshistorical/staleduringlivewriters.


Verified wait 20261030T013112Z: specific OS identities139744/621078/140819/166038/194819/648344/640365 checked. Baseline control {'stage': 'validation', 'elapsed_seconds': 1354.823249727022, 'label': 'after_binomial_202610343', 'batch': 0}; K8thirdseed {'stage': 'nominal_controls', 'operator_calls': 287, 'last_module': 'lm_head'}. Mean/stochastic training audits accepted; baselineaudit andstrictreplay pending before adaptednativeexecution; allocatedqueue preserved. Previousturnprogress(uncertaintymath/geometrycomponent), currentturnverifiedwait. No new nativequality result, no BF16equivalence, no restart. Goalactive; receipt results/thrml_spw/cifar_cnn_native/execution_checks/20261030T013112Z_LFM3_verified_wait.json.


Bounded verified wait 20261030T013450Z: seven30-second observations over180seconds verify stable parentprocess identities139744/140819/166038/194819/640365. Baseline control advanced {'stage': 'validation', 'elapsed_seconds': 1354.823249727022, 'label': 'after_binomial_202610343', 'batch': 0} -> {'stage': 'validation', 'elapsed_seconds': 1590.3546611659694, 'label': 'after_binomial_202610343', 'batch': 3}; currentseed343of340..344, no acceptedfinalresultyet. K8thirdseed frontend648344 andsamplingwrapper648411live; progressreceiptstillnominalheadwhilewaitingnextnativeworkerrequest, notaterminalstate. Existinguniform/allocatedpipelineswaitsameaudits; no scientificsettingschanged/restarts. Previousgoalturnverifiedwait,currentturnverifiedwait; no genuineexternalblocker. Fullonepbit-per-parametergoalactive, BF16qualityunproven. Receipt:results/thrml_spw/cifar_cnn_native/execution_checks/20261030T013450Z_LFM3_bounded_live_monitor.json.


## GPU parallel bank component passed serial/public fidelity checks — 20261030T013932Z

The primary target remains one logical p-bit per unique learned parameter. To reduce Python/GPU dispatch overhead in future large-model inference, added a new component that vectorizes independent official THRML bank executions with JAX vmap. Each lane retains its own original folded PRNG key, field array, observer input and bank state. It does not replicate precision bits for one parameter. Existing programmed_worker.py, all scientific drivers/protocols and all running/queued full-model studies are unchanged. No simulator timing, physical latency, energy or model-quality benefit measured yet.

All six GPU component cases passed serial-versus-vmap comparison for every sampled public state and int32 accumulator. Variants include fanin1/7/128/256, depthwise and linear observer layouts, two/three/four independent lanes, distinct original tile IDs, finite12bit endpoint/interior fields, signedACT12 inputs, T1/16/1024/8192 and int32 overflow guards. 135,812,288 public binary values and 2,816 integer entries per execution path checked. Independent offline NumPy/int64 reconstruction used public per-weight counts and original inputs to reproduce each serial/vmap integer accumulator. No full neural operator readout, software speed, full-model graph or quality parity is inferred from primitive equality. Saved compressed raw arrays retained for all six cases. Sources archived before execution; verifier hash bound prospectively in verification_protocol.json and root hash/seal review passed. Producer tool session21619 and offline verifier both exited0.

This component needs a separate prospective worker integration (original per-tile FP64 readout, unchanged resources/padding/random keys), operator-level exact controls/public checks and a full original model replay before adoption in later studies. It is not inserted into the frozen V2 or allocated protocols. Current one-p-bit evaluations remain behind baseline validation/audit and strict original replay. Baseline reached final Torch MCseed344/batch2; K8thirdseed native advanced to layer3. All processes retained without restart. Previous goal turn verified wait; this turn progress from a new executed and independently verified GPU primitive. Full research goal active, BF16 task quality unproven.

[GPU component audit](../../results/thrml_spw/lfm3_1p7b/batched_bank_preflight/independent_audit.json) · [Prospective verifier binding](../../results/thrml_spw/lfm3_1p7b/batched_bank_preflight/verification_protocol.json). Global evidence inventory remains historical during ongoing writes.


## All adaptation audits passed; original native replay running — 20261030T014503Z

The original adaptation producer (tool session94018) and audit queue (session86048) both exited0. Baseline control validation completed all2048reservedtargets/fiveTorchMCseeds; independent audit accepted every1,728,801,536initial control/finiteprogram coordinate, zerochangedcommands andall savedvalidationlosses. Mean andbinomial arms previouslypassed16-stepall-coordinateAdam/program/validation audits. Allthreeaudits nowaccepted; rootchecked verifier/source bindings, result/package hashes, step/coordinate counts andnative selection inputs. No optimizer/validation setting changed.

The prospectively defined native scheduler selected mean-trained program using prior Torch surrogate averagevalidationNLL: {'baseline': 5.19254277148887, 'mean': 4.393358999634852, 'binomial': 4.4518788390909325}. Allthree nativecandidates remainmandatory regardlessofselection/outcomes. Surrogategains includeextra adaptation, so they do not establish representationquality against SAME-FITBF16 or nativeTHRMLtransfer. OriginalunadaptedBF16 onthese2048validationtargets is4.975995543278368, distinctfromtheearlier512-targettestsubset; do notcomparethosebaselines acrossdatasets.

NativeV2pipeline166038 is nowrunning its strict original replay child664915/start63925791, seed202610300/unadaptedprogram/originalconsumed512targets. It must reproduce all originalBF16/nominal/native logits exactly underunchangedrandomkeys beforetrainedcandidate evaluation. Theallocatedsamplingstudy640365 waits for this acceptedreplayaudit, then runsall5meanpointcases. First replay execution is not completion and no exactreplayresult acceptedyet. Fullnative adaptedquality remainsunproven. Current onepbit-per-parameter target includesalluniquelearnedweights,sharedembedding/head andnorms; K4/K8strictlycontrols.

Previousgoalturn progress (parallelbankcomponent), currentturnprogress fromallthreeaudit acceptance/native replay start androotsealreview; boundedmonitorobservedtransitionfromliveproducerthroughlastindependent audit. No retries, queuedsource changes orhardwareclaims. Allfailures/negativespreserved; 10240-targetholdoutunscored. Fullgoal active.


## Parallel finite-program worker passed exact operator/readout audit — 20261030T014941Z

Implemented a new separate batched_programmed_worker.py using the previously verified vmap bank component. It batches up to four independent64-channel tiles, keeps each original key=fold_in(fold_in(seed,call),tile), and retains original per-tile FP64 affine readout/order after int32 group accumulation. No continuous weight reconstruction or range refit. Final chunks use their actual number of tiles, with no extra dummy lanes. Padding/state/retained/warmup work and maximum simultaneous simulator slots are explicit. Four tile lanes represent different parameters, not multiple precision p-bits for one parameter. Existing programmed_worker, nativeV2 and allocated protocol/source files remain unchanged and all current model processes keep the original worker.

New worker outputs matched every one of the34 previously accepted GPU direct-program cases (four kinds, initial/perturbed/trained controls, mean/native,T1/16). Four additional native cases test fanin128/channels323/four+two lane chunks,T16; fanin7680/channels129/three lanes,T16; fanin128/channels197/four lanes,T1024; and fanin1/channels323,T8192. These include output padding, repeated chunk boundaries and actual large LFM3 matrix fanin. All sampled public state arrays, integer totals, original codes and quantized inputs match serial reference exactly. Independent offline int64 reconstruction and FP64 readout reproduced every native output. In total62,564,096 public binary values and96,000 counter entries per path verified; all38 outputarrays exact. Fixed original raw/source/output hashes and new inputs/raw/output hashes retained. Additional public replay draws are audit work, not free production samples; public bit counts report that per-path work. Producer session31864 and independent verifier exited0. New source hashes archived before execution, verifier included in prospective protocol. Root hash/resource-binding review passed.

This establishes component/worker fidelity, not full-model graph fidelity, measured simulator speed, hardware latency/energy or quality. More simultaneous simulator slots (up to4*fanin*64 vs one tile in original worker) are reported and must not be mistaken for reduced simulation residency. Next permissible adoption step is a separate prospectively frozen full original replay under this worker with exact original logits/resource comparison. No current model study is changed or replaced. Main strict nativeV2 original replay664915 passed all287 nominal operator executions and is now native atoperator0; acceptance requires final originalBF16/nominal/native arrays bitwise equal, still pending. K8thirdseed remains running; all5required.

Previousgoalturnprogress(alladaptationaudits/native replaystart); currentturnprogress(newworker+38GPUcases+independentpublic/readoutaudit). Primarytarget remains1logicalpbitpereachof1,728,801,536uniqueparameters. Fullgoal active, nativeadapted/SAME-FITBF16qualityunproven,10240-targetholdoutunscored.

[Worker audit](../../results/thrml_spw/lfm3_1p7b/batched_worker_preflight/independent_audit.json). Global evidence inventory remains historical during live writes.


## Separate full-model replay of the parallel worker frozen and launched — 20261030T015332Z

After the38-case operator/public/count/readout audit, froze a separate replay-only protocol (6e394dab4041ab4bb3b30c089196479c796cf4af71814ba126858ff30a7f1269) and launched batched_native_replay_pipeline. It executes the identical unadapted286canonicalprograms/287moduleuses/1,728,801,536uniqueparameter controls, samefinite12bitfields/fixedFP32ranges/ACT12, sameT1024 andoriginalseed202610300/call+tilekeys onthealreadyconsumed8x65contexts/512targets. Fourconditions(originalBF16,same-fitBF16,nominalACT12expected,native) retained. Only the sampling worker and output/protocol namespaces change from the V2 driver; replay-only argument guards and38-case prerequisite added. Source diff reviewed/saved. No training, freshholdout consumption or nativecandidate selection.

Independent verifier requires every saved originalBF16/nominal/native logit bitwiseequal toimmutablelanguage_native_v2/202610300outputs, exact pertokenlosses and allcanonicalprogram/same-fitBF16weight hashes/tiedaliases/operatororder. It checks identical retained/warmup/paddingwork, plus perrequestfour-lane/tilecount/state-slotbounds. Extra simultaneous tile lanes process differentparameters ratherthanreplicatingprecisionbits. Claimed samplerstate-slotcount is a structural bound, not measuredpeak RAM or physicaldevicearea. The pipeline cannot automatically adopt the new worker or change any current model study. Failure will be retained without gate relaxation. Allnewdrivers/protocols archived/frozen beforeexecution; originalV2 andallocated sources/protocolsunchanged.

New pipeline PID699682, child699683, toolsession90658, identityconfirmedlive. OriginalV2replay664915 remainslive (native layer1 at observation); K8thirdseed648344 remainslive (native layer17); allocatedqueue640365 stillwaitsacceptedoriginalV2replay. No timing comparison can be inferred from concurrently running simulatorjobs. Even exact fullimplementationreplay wouldnot establish adaptedquality or BF16equivalence. This turn isprogress(newfrozen/fullmodelimplementationtest launched); previousturnprogress(worker38-caseproof). Full onepbit-per-parameter researchgoalactive/incomplete. Reserved10240-targettext remainsunscored; globalinventory remainshistorical/stale whilewriterslive.

[Replay protocol](../../results/thrml_spw/lfm3_1p7b/batched_native_replay_protocol.json) · [Source derivation review](../../results/thrml_spw/lfm3_1p7b/batched_native_replay/source_derivation_review.diff).


Verified wait 20261030T015904Z: nine identity-bound observations over four minutes. Original V2 replay advanced from native operator70/layer6 to129/layer10; parallel replay advanced from nominal operator180/layer15 into native operator7/layer0. K8thirdseed advanced521->545calls/layer21. All seven recorded frontend/scheduler OS identities remained stable; actual original sampler665067 and batched sampler713531 confirmedlive separately, with activeCPUtime and no worker-logerrors. Host reported47GiBavailablememory. No timing/hardware advantage inferred fromconcurrentjobs. No finalreplayaudit ornewnativequality acceptance yet. All source/protocols andqueuedconditions unchanged; no restarts. Previousgoalturnprogress(fullparallelreplaylaunch), currentturnverifiedwait. Fullonepbit-per-parametertarget andgoalactive; BF16qualityunproven, reservedtextunscored. Receipt:results/thrml_spw/cifar_cnn_native/execution_checks/20261030T015904Z_LFM3_two_replays_bounded_monitor.json.


## Repository entry documentation now includes current LFM3 scope and evidence — 20261030T020635Z

Updated existing root and research README files to lead with the actual1,728,801,536parameter/onepbit target, signed-range normalization, accepted frozenfive-seed missvsBF16 (NLL4.9282vs4.6357,PPL138.13vs103.10 on512fixedtargets), adaptationauditclosure and same-fitBF16reason. Added currentprotocol/evidence/pipeline pointers, reservedtextstatus, resource/trajectoryscope, environment/source-snapshot reproduction notes andread-only current-state commands. Eachof30newlocal links checkedexists. AllolderREADME bytespreserved in README_ROOT_HISTORY_20261030T020327Z.md andREADME_RESEARCH_HISTORY_20261030T020327Z.md underresults/thrml_spw; older milestones explicitlyhistorical. Frozen scientific files and modeloutputs unchanged. Source/input protocol hashes provide exact reproducibility; existingexclusiveoutputnamespace must bepreserved.

Currentreplays/comparisonarelive withnofinalnewacceptance: originalnativeV2replay reachedlayer14, batchedreplaylayer1, K8thirdseedlayer23 atlastobservedprogress. Per-operator progress ofnewbatchedworker doesnot establish asimulator-speedgain, especiallywhileinitialcompiledshapes andconcurrentjobsdiffer; onlyfidelity is beingtested. No new quality/speed/hardwareclaim. Primaryonepbit-per-parametertargetunchanged. Previousgoalturnverifiedwait; currentturnprogress(entrydocumentation/reproducibility corrected) withspecificliveOSidentitiesrevalidated. Nextscientificstep stillcompletingexactreplays andactualtrainednativequalityevaluation. Goalactive,incomplete;10240reservedtargets remainunscored.

Documentation receipt: results/thrml_spw/cifar_cnn_native/execution_checks/20261030T020327Z_LFM3_entry_documentation_progress.json. Globalinventoryremainshistorical/stale duringlivewriters.


## Exact count factorization and worker passed GPU/public/readout audits — 20261030T021650Z

Current model jobs remain unchanged. The separate four-lane vmap replay is advancing more slowly than the original replay atthisobservation; no speedgainhasbeen established. Different compiledshapes andconcurrentworkloads prevent a fair elapsed-time comparison. To address the expensive observer arithmetic, added a separate exact software simulation factorization for a fixed integer input: sum_t(X_int Z_t)=X_int(sum_t Z_t). It uses the official THRML sampler with the original fields/keys/schedule and an int32 count observer, then one grouped integer contraction outside the temporal loop. Existing ACT12/group128/T<=8192 overflow guard ensures all rearranged integer operations are exact. No approximate Bernoulli replacement or changed stochastic budget.

This uses one temporary integer count per weight in each simulated bank. It therefore is NOT evidence that the original physical direct-output-counter datapath has no count-storage cost, nor does it improve statistical precision or establish a hardware advantage. Representation/reconstruction/execution regimes remain distinct. This is an accelerated arithmetic evaluation of the same fixed-input observable in software; all existing direct-accumulation experiments and resource claims are preserved. Count workspace can be larger or smaller than output accumulators depending on the number of input positions; all recorded byte costs are retained. It requires full-model exact replay before adoption in any later quality evaluation.

Six GPU primitive cases passed independent public-count/int64 contraction audit, including520positions andfanin7680, signedACT12 inputs, two/three/fourlanes, depthwise/linear observers andT1/16/1024/8192. Publicbits175,658,176; output integer entries4,327,232perpath; weightcount entries1,058,304perpath. Returned serial/vmap counts equal independently summed publicstates, and both factored outputs match the original per-sample GroupSum observer exactly. Large520-position shapes explicitly exercise actualLFM3observer geometry absentfromtheearliersmall-inputtimingassumptions. No timing result claimed.

A derivative counted_programmed_worker retains original finitecodes/sharedranges/keyfolds/per-tileF64readout, explicitly returns observer identity and temporarycountworkspace bytes. All34accepted originalGPUcases plusfour wide/tail cases matched savedserial outputs; independent audit checks publicstate/code/input/total equality and every returned per-weightcount, then reconstructs int64 totals/F64readout and sampling/padding/warmup receipts. Worker publicbits62,564,096 and96,000integerentriesperpath. Producer sessions14604/98343 and both independentverifiers exited0; allnewsourcesarchivedandprotocol-bound beforeexecution. OriginalfullmodelV2/allocated/K8 andvmapreplay sources/conditions remain untouched.

Mainoriginalreplay664915 reached nativeoperator281/layer23 atlastobservation; finalhead/results/audit stillpending. K8thirdseed648344 remainsatfinalheadpendingcompletion. Vmapfullreplay699683 remainslive; no exactfullmodel orqualityresultacceptedyet. Nextstep for the counts worker is a separate frozen fullmodel original-logit replay, not adoption based onlyonprimitive tests. Thisturnprogress(twoexecutedGPUproofs/workers/independentaudits); previousturnprogress(entrydocumentation). Fullonepbit-per-parametergoalactive, SAME-FITBF16qualityunproven,10240reservedtargets unscored.

[Count identity audit](../../results/thrml_spw/lfm3_1p7b/counted_bank_preflight/independent_audit.json) · [Count-worker audit](../../results/thrml_spw/lfm3_1p7b/counted_worker_preflight/independent_audit.json). Globalinventory remains historical duringlivewrites.


## Strict full original replay accepted; allocated trained evaluation released — 20261030T022747Z

The original V2 full-model replay passed independent audit and root seal review: all1,728,801,536uniqueparametercontrols/287operatoruses covered. Every BF16/nominal/native logit is bitwise equal to the original language_native_v2/202610300 run, with the same original512targets/seed300 and direct finite-command packages. Same-fit BF16 weight hashes/allcanonicalaliases checked, saved tokenlosses independently recomputed, result/output/source hashes rootchecked. Retainedschedulednodeevaluations1,772,027,117,568 andwarmup1,730,495,232 exactlymatchoriginal. NativeNLL4.883533826061303 is an exact implementation replay, NOT a new independent qualityrun orimprovement. The original five-seed miss vsBF16 remains unchanged.

This accepted gate released both planned pipelines: originalV2 nowevaluates baseline seed350 on32validationcontexts/2048targets, and allocated_native nowevaluates the actual mean-trained program atseed350 underfixedtraining-selectedT_module. Thus the trained one-p-bit model's fullnativeevaluation is in progress; no acceptedqualityresult yet. Allfive seeds and same-fitBF16controls mandatory, no scorestopping. K8thirdseed302 independentlyaccepted: nativeNLL4.620849931010616, alluniqueweights, exactdeterministiccontrols, retainedworkmatchedK1/K4 andwarmup8xK1; 3of5accepted, fourthrunning. No cohortadvantage/equivalenceclaim beforeallfive.

A separate count-factorized fullmodel original-replay protocol was prospectively frozen (bb2b178ea2e2af435cb8bab9a2469fa141e30199ad7e8b54adfc23c46324ccb6), after accepted6primitive/38worker audits. New replay-only source changes worker/namespace/observerworkspacechecks whilekeeping all originalfields/keys/T1024/ACT12/F64readout. It explicitly stores per-weightint32 temporarysoftware counts, not afreephysicalR2counterdatapath. Scheduler761986/tool12421 waitedforacceptedoriginalstrictreplay beforelaunchingchild762294; nowrunning. Acceptance requires exact originalBF16/nominal/native arrays andnodework, allparametercoverage/controlhashes andevery count-workspace/state-slotreceipt. Noautomaticadoption orcurrentstudyreplacement. Sourcesarchivedbeforeexecution; sharedprotocolandupstreamaudithashbound.

Operational scheduling attempted a bounded180-second holdofsecondaryvmapfrontend699683 whileitsworkerfinishesthecurrentrequest. The originalauditwasalreadyavailable, so finally resumedthefrontend after0.000659seconds, reasonoriginal_replay_audit_available. Identitycheckedandfrontendconfirmednonstopped; no restart, source/data/settingschange, goalpauseorperformanceclaim. This near-instant schedulingevent did not cause the earlier replaycompletion. Vmapfullreplay remainslive andnotaccepted. Host29GiBavailable atobservationafternewrunsstarted.

Currentturnprogress(frozen/countfullreplaylaunched plus originalstrictgateacceptance andtrainedallocatedrelease); previousturnprogress(countproofs). Allrequirednegatives/failures preserved. Fullgoal active/incomplete, BF16task-equivalence notproved,10240reservedtargets unscored.

[Strict replay audit](../../results/thrml_spw/lfm3_1p7b/adaptation_native_v2/initial_replay/independent_audit.json) · [Count replay protocol](../../results/thrml_spw/lfm3_1p7b/counted_native_replay_protocol.json) · [Third K8 audit](../../results/thrml_spw/lfm3_1p7b/word8_language/202610302/independent_audit.json).


## Counted full replay accepted; selected mean quality cohort started — 20261030T025752Z

Count-factorized software full original replay completed and passed independent audit. All1,728,801,536uniquecontrols/287uses, exactoriginalBF16/nominal/native outputs, original512targets/seed300; retainednodework1,772,027,117,568 andwarmup1,730,495,232 unchanged. Maximumconcurrentbankstateslots1,966,080 andtemporaryint32weight-countworkspace7,864,320bytes explicit. This doesnot establishphysicaldirect-output-counterdatapathcost equivalence, numericalBF16precisionornewquality. Root additionallycompared allfour outputstages(includingsame-fitBF16) asuint32bitpatternsagainstoriginalV2, covering266,240,000F32valuesand signedzeros; compared574operatorreceipts/canonicalweight hashes exactly. Thelegacythree-stage bitpatterns matched too. Rootcrosschecksealed separately; implementationfidelity gateclosedforthispoint.

A separate prior-selectedmean qualitycohort is prospectivelyfrozen(protocol053c020bfb7907023edc49540a6105f61bfe9512fc9db7603ea8a5a2687c48ee) and running. Exactlytheauditedmean-trainedcommandpackage, all32validationcontexts/2048targets, fivefixedMCseeds350..354,T1024 andthepriorV2same-fitBF16upperMCt4NLLmarginlog1.02. Fourconditions, everyprogram/alias/target/loss, temporarycountworkspaces andnodeworkcheckedforeachseed; independentDecimalstatisticsafterallfive. ThemeanpointwaschosenbypriorTorchsurrogateselectionbeforeanynativevalidationoutcome. Syntheticstatisticsfixturetestedpassingandfailingthresholds andrejectedalteredgate/CI values inisolateddirectories. Allnewsources archivedbeforeexecution. Originalthree-armV2/allocated/K8/vmapstudiescontinueunchanged. Futureoriginalmeanoutputswiththesamefivekeysareimplementationduplicates, notadditionalindependentseeds; do notpoolas10MCruns. Onepbitperuniqueparameter targetunchanged.

Selectedmeanpipeline924513/child924514, toolsession38784, confirmedlive and atnativebatch0. Countreplayoriginalpipelinecompleted0; nocoordinatorautomaticadoptionoccurred: newcohortexplicitlyfrozenandreviewed afterproof. Fullquality andconditionalBF16gatepending; no reservedtext scoring. Thisturnprogress(fullcountproofacceptance,newfrozenqualitycohort,testedstats,andcost-evidenceupdate); previousturnprogress(originalstrictgaterelease).

## Public hardware planning inputs separated from measurements — 20261030T025752Z

Extropic's [Summer2026 announcement](https://extropic.ai/writing/from-one-to-one-billion) lists269,568pbitsinFigure05,16chipspercard,andasamplingrateover50MHz. Its [hardwarepage](https://extropic.ai/hardware) lists2027earlyaccess. Thesearecompany-publisheddesign/roadmapclaims, not measurementsfromthisproject. URLs/HTTPresponsefingerprints andtranscribednumericclaims retained; copyrightedfullpagebodiesnotredistributed. Theconsultedpagesdonotverifyour12bitbiascontrolprecision,independentsamplerate,readoutenergy orapplicationinferenceenergy.

Owncapacity-onlyarithmeticusingthosenominalfigures gives6,414chips/401cards foronefullyresident1.729B-nodeassignment, beforewiring/controllers/ranges/readout/feasibility; not anactualdeploymentdesignoravailabilityclaim. Aseparateindependentarithmeticverifiercheckedceilcapacityinequalities, localcontrol/rangepayloads, andhypotheticalserializedsample-roundtimesat1/10/50millionINDEPENDENTobservations/sec. Publicsamplingclockdoesnotproveindependence. At50millionindependentobservations/sec asanassumption,sumT/fis5.87776msuniformand26.04544msallocated; theseomitoperatorcompute/program/readoutandDAGoverlap andareNOT measuredhardwarelatencyorlowerbounds. Energy/latencymeasurementfields remainnull; publicchip-powerclaimnotconvertedtoenergypertransition. K2/K4/K8 remaininternalencodings,notclaimedExtropicformats. Thiscostmodeldoesnotproveahardwaredensity/energyadvantage.

[Fullcountreplayaudit](../../results/thrml_spw/lfm3_1p7b/counted_native_replay/initial_replay/independent_audit.json) · [Rootbitpatterncrosscheck](../../results/thrml_spw/lfm3_1p7b/counted_native_replay/root_replay_crosscheck.json) · [Meanvalidationprotocol](../../results/thrml_spw/lfm3_1p7b/counted_mean_validation_protocol.json) · [Hardwarecontextarithmetic](../../results/thrml_spw/lfm3_1p7b/published_hardware_context/independent_audit.json). Fullgoalactive/incomplete; BF16qualitynotproved; globalinventoryhistoricalduringlivewriters.


## First completed native adapted validation batch: diagnostic only — 20261030T031105Z

The first fixedordinalseed350/batch0completedandwasindependentlyscoredfromsavedlogits. Itcovers8of32validationcontexts/512of2048predictiontargetsandoneMCseed, atT1024usingtheauditedmean-trainedpackage(all1.729Buniqueparametercontrols). Everytokenloss checkedtwoways, NumPyFP64logsumexpandTorchFP64crossentropy, rtol/atol1e-12; inputs/outputs/protocol/programsourcehashesbound. No othermodelrun, newtraining,samplerchange orfreshholdoutscoring. Rootcheckedtokenloss/artifactbindings. Thediagnosticisnotafinalfullcase/program/resourceaudit; crosscheckwiththefinalauditedseed350recordwhenavailable.

Firstbatch NLL/PPL: originalunadaptedBF164.571355349816873/96.67504952888822; SAME-FITadaptedBF164.067842146170707/58.430741458287244; adaptedACT12expected4.050323897164371/57.41605093011602; actualnativeonepbit4.112129491644858/61.076641382207406. Native-minus-SAME-FITBF16NLLgap0.044287345474149795,PPLratio1.045282668983569 (4.53percentworse). Native-minusexpectedoperatorgap0.061805594480486115, ratio1.0637555246797945. NativeimprovesovertheUNADAPTEDBF16baselineonthisbatchbecauseextralearningchangedthemodel; thatcomparisondoesnotestablishrepresentationequivalence. Thesevalidationtargetsare differentfromtheold512testsubset; do notcompareresultPPL61withtheearlierfrozenPPL138/103 asifmatcheddata/model.

Noqualitygateorconfidenceintervalcomputed:512correlatedtargetsareoneMCcluster,not512independentstochasticruns. Thewhole2048target/five-seedtwo-sidedMCt4gate<=log1.02vsSAME-FITBF16stillrequiresallplannedcases. Thefirstbatchisdescriptiveevidenceofremainingdeploymentnoise, notaformalcohortfailureorsuccess. Allcurrentruns/budgets/seedsretainedwithoutscore-basedstopping. Nextpointdesigncanbeconsideredafterfullcohortanalysis; noredefinedthresholdorselectedfavorablebatch. Maincountedseed350hasadvancedtobatch1. OriginalV2/allocated/K8/vmapstudiescontinue.

Previousgoalturnprogress(fullcountproof/meanstudy/hardwarecontext); currentturnprogress(newactualadapted-nativebatchloss evidence independentlycomputed). Fullonepbit-per-parametertarget/goalactive,incomplete; noBF16equivalenceclaim;10240reservedtargets unscored. [Diagnostic](../../results/thrml_spw/lfm3_1p7b/counted_mean_diagnostics/seed202610350_batch0/metrics.json). Globalinventoryremainshistoricalduringlivewriters.


## First complete adapted-native validation seed accepted — 20261030T034024Z

Seed202610350 completed all32reservedvalidationcontexts/2048targets andpassed independentfullcase audit: all1,728,801,536uniquecontrols/286programs/287moduleuses, everycommand/hash/tiedalias/same-fitBF16weight hash, every savedtokenloss, all2,296nominal/nativeoperatorreceipts andexplicitcounterworkspaces. Retainedschedulednodeevaluations7,088,068,624,384 pluswarmup6,921,942,016 acrossfourbatches. Maxsimultaneoussamplerstates1,966,080 andtemporaryper-weightcountworkspace7,864,320bytes, notphysicalSRAM/area measurements. Rootreviewverifiedsource/result/audit bindings, allfourbatchloss means/resources andall512earlierdiagnostic tokenlosses againstthefinalrecord. No neural source/seed/budget change.

Full2048targetFIRSTSEED NLL/PPL: originalunadaptedBF164.978250316314612/145.22007004870076; SAME-FITadaptedBF164.321675645982948/75.31472343088639; ACT12expected4.29973876220307/73.68054304246463; nativeonepbit4.429380291303697/83.87941999445101. Pairednative-minus-SAME-FITBF16NLLgap0.107704645320749, PPLratio1.1137187547588109360233273226300427288135917407192 (about11.37percentworse). Thisreplaces the4.53percent FIRSTBATCH diagnostic as theavailable full-seed descriptive score; thefirstbatchwaseasieranddidnotrepresentallvalidationtargets. Native improvementversusunadaptedBF16includesextra learninganddoesnotestablishrepresentationprecision.

Stilloneof5MCseeds; noMCconfidenceinterval orcohortpass/failgateapplied. All2048targetseeds350..354 remainmandatoryunderfixedT1024/program/2percentPPLmargin. Secondseed351started, child952784/start64581517, identityconfirmedlive. Fullfinalstatisticalauditneedsallfive; no scorestoppingorparameter/budgetchanges. Ifcohortmissesthecriterion, furtherresearch can adjust precision/training under a new frozen protocol ratherthanrelaxingthecriterion. OriginalV2/allocation/K8/vmapstudiescontinueunchanged.

Previousgoalturnprogress(firstbatchdiagnostic); currentturnprogress(firstcompletecase audit accepted, diagnosticcrosschecked, nextseedconfirmedlive). Two boundedidentitymonitorreceipts preserve six- and seven-minute observations throughfinalbatch/verification. Fullonepbit-per-parameterresearchgoalactive,incomplete; BF16qualityequivalenceunproved,10240reservedtargets unscored. [Firstseed audit](../../results/thrml_spw/lfm3_1p7b/counted_mean_validation/mean/202610350/independent_audit.json) · [Rootreview](../../results/thrml_spw/lfm3_1p7b/counted_mean_validation/mean/202610350/root_review.json). Globalinventoryremainshistoricalduringlivewriters.


## Conditional fourfold temporal-precision extension frozen and queued — 20261030T035810Z

CurrentT1024allfive-seedmeanstudy continuesunchanged; onlyfirstcompletedseedaccepted(11.37percentPPLgapvssame-fitBF16), sofullcohortqualitygatehasNOTbeenclassified. Secondseed351isprocessingitsvalidationcontexts. AnewconditionalT4096extensionisnowfrozen(protocol9a9c91c10911629e0676febaf2d7c8e18a17a75b83bb4feebe81a4fc060a76ff) andscheduler967036/start64682856/tool1410confirmedlive. ItwaitscompleteallfiveT1024casesANDindependentlyauditedstatistics. Ifthatfixedsame-fitBF16upperMCt4gatepasses,itrecordsnot-neededanddoesnotrunsampling. Ifqualitygatefails,itexecutesallfiveT4096seeds420..424,thenindependentcase/statisticalaudits. Lowerimplementation/auditfailurestopstheextensionwithoutautoretry. No conclusionfromonlythefirstseed andno scorestoppingoncehighTstarts.

Weights/groupmetadata/controllerprecision/ACT12/F64readout/32validationcontexts/2048targets/2percentPPLmargin stayfixed. T4096isfourtimesretainedstochasticnodeworkwithunchangedwarmupandonepbitperuniqueparameter. FreshMCkeysavoidcouplingtothetriggeringlow-budgetnoise; cross-budgetseedresultsarenotpairedorpooledassame-conditionruns. NonlinearNLLneednotfollowthescalar1/Tvariance law; effectunproven. Everyhigh-budgetdeterministiccontrolbitpatternmustmatchtheacceptedlower-budgetseed350 controlsacrossallfourbatches. Runtimeint32guardmax2047*128*4096=1,073,217,536<2^31; existingT8192public/countproofcovershigherintegeredge. Temporarycountworkspaceexplicit; nofreephysicalstorageorenergyclaim. No newtraining orfreshholdoutscoring.

Continuationpredicatewasindependentlyexercisedonsyntheticpass/fail andsixincomplete/alteredrecords, rejectingpartialcohorts andunauditedhashes. Newstatisticsmodulesarenamespace-onlyderivativesofthealreadytestedNumPy/Decimalmath; byte-derivationidentitychecked. SourcesarchivedbeforeexecutionandprotocolfrozenbeforeanyT4096outcomes; originalT1024/V2/allocated/K8sources/jobsuntouched. High-budgetmodelrunshavenotstarted. Fullqualitygate stillrequiresallfiveMCclustersonfixedvalidationdata; notformalBF16numericalequivalence, generalizationormultiplicity-controlledproof.

Theseparatefour-laneGroupSumfulloriginalreplayalsofinishedandpasseditsindependentaudit: allNcontrols,originalBF16/nominal/native arraysandoriginalnodeworkexact. Thisisduplicateimplementationfidelityontheconsumed512targets/seed300,notnewqualityevidenceornewindependentMCsample. No causalsimulator-speedgaininferredfromconcurrentwalltimes; countfactorizationremainsseparatelydocumentedsoftwareobserver.

Previousgoalturnprogress(firstfullmeanseedaccepted); currentturnprogress(conditionalprecisionstudyfrozen,testedandactualschedulerwaiting, secondaryreplayauditclosed). Fullonepbit-per-parametertarget/goalactive,incomplete; BF16qualityunproved and10240reservedtargets unscored. [Conditionalprotocol](../../results/thrml_spw/lfm3_1p7b/counted_mean_t4096_protocol.json) · [Continuationpredicatefixture](../../results/thrml_spw/lfm3_1p7b/count_budget_continuation_preflight.json). Globalinventory remainshistoricalduringlivewriters.


### 2026-10-30T04:02:55.900477+00:00 — live one-p-bit cohort verified wait

Verified the current OS identities for the selected mean cohort, its second-seed child, conditional T4096 scheduler, original uniform cohort, allocated cohort, and K8 comparison. The selected second seed is on validation batch 3; original uniform is on batch 1, and allocated on batch 1. The earlier user-facing status answer changed no scientific state; this continuation verifies actual live work. No new treatment, restart, stopping rule, or scientific source change. One-p-bit-per-parameter scope is unchanged. The first completed seed remains insufficient for BF16 equivalence. Receipt: `results/thrml_spw/cifar_cnn_native/execution_checks/20261030T040255Z_LFM3_live_cohort_wait.json`.


### 2026-10-30T04:03:36.239544+00:00 — conditional high-sampling scheduler integrity check

Recomputed all 53 frozen source and input hashes for the queued T4096 continuation. All matched, including the scheduler source and its archived bytes; launch receipt matches the frozen protocol. Rechecked lower pipeline PID/start ticks against its launch binding. No continuation decision has been written and the scheduler still waits for all five T1024 seeds and accepted statistics. This is a verified wait plus integrity evidence, not another quality run or treatment change. Receipt: `results/thrml_spw/cifar_cnn_native/execution_checks/20261030T040336Z_LFM3_conditional_scheduler_integrity.json`.


### 2026-10-30T04:12:26.833360+00:00 — second full selected-mean native seed accepted

Seed 202610351 completed all 2,048 validation targets and independently passed full program/alias/operator/output-loss/resource checks. Native NLL 4.392363728321433, perplexity 80.83125647781789, versus same-fit BF16 NLL 4.321675645982948, perplexity 75.31472343088639. Root recomputed every stage mean from all saved token losses using Decimal; all three control metric records and resource receipts exactly match seed 202610350. This seed's PPL ratio is 1.0732464091432735720217172628531484198274040587922 (approximately 7.325% worse). First seed was 11.372% worse. Two of five fixed-protocol MC seeds are accepted; no interim cohort CI or gate decision. Source/result bindings and verifier archive checked in `results/thrml_spw/lfm3_1p7b/counted_mean_validation/mean/202610351/root_review.json`. Run 3 (202610352) is live as PID 999260, start ticks 64813325. One logical p-bit per each of 1,728,801,536 unique parameters is preserved; control/counter workspace is additional storage, and physical BF16 equivalence is not established.


### 2026-10-30T04:15:03.911342+00:00 — accepted-case deterministic versus sampled loss accounting

Archived and executed `research/thrml_spw/lfm3/accepted_case_loss_decomposition.py` (SHA256 54b480bf206661f634e205a50cc1bfc6205f64e7779f5548df0fbad2b711f282) on accepted seeds 202610350 and 202610351 only. All 2,048 saved token losses per stage/case and all 32 validation windows are retained; source/audit/result bindings and identical controls checked. An independent math.fsum check verified 456 scalar outputs within 1e-12. Native-minus-nominal NLL is 0.1296415291006259555582265625 and 0.092624966118358720184626953125; nominal-minus-same-fit-BF16 NLL is -0.021936883779877547147052734375 in both. Thus the deterministic programmed-weight/operator baseline is slightly better on this subset while sampled runs are worse. This is descriptive algebra, not independent causal contributions: stochastic trajectories also change propagated activation quantization. No assumed 1/T NLL law or success guarantee for T4096; no cohort inference, modified rule, new treatment or holdout score. Outputs: `results/thrml_spw/lfm3_1p7b/counted_mean_diagnostics/accepted_two_seed_loss_decomposition.json`, `results/thrml_spw/lfm3_1p7b/counted_mean_diagnostics/accepted_two_seed_loss_decomposition_root_audit.json`.


### 2026-10-30T04:19:37.438542+00:00 — live cohort and software resource snapshot

Verified live process identities for all five schedulers and their observed descendants. Captured Linux smaps_rollup PSS/RSS/swap counters, host MemAvailable, and free disk without modifying running jobs. Observed 17 processes in these trees, summed PSS 26.49 GiB, host MemAvailable 24.80 GiB, disk free 381.72 GiB. These measure concurrent simulator software and do not describe packed control size or physical p-bit hardware. GPU/driver-managed pages may not be fully attributed. Third selected-mean seed remains live; two accepted seeds, no new quality conclusion. Receipt `results/thrml_spw/cifar_cnn_native/execution_checks/20261030T041937Z_LFM3_runtime_memory_observation.json`.


### 2026-10-30T04:27:34.551882+00:00 — matched-resource comparison scope audit

Read and fingerprinted frozen original K8 and both selected-mean quality protocols. Confirmed the existing equal-work K1/K4/K8 experiment covers original checkpoint weights and 512 fixed targets, while adapted one-p-bit quality covers 2,048 targets at a trained point. These must not be merged into an adapted-model Pareto claim. For matched geometry, T4096 requires K8T512 (or K4T1024), not the current K8T128; actual node padding, warmup/control/metadata/workspace remain separately verified. A matched adapted spatial-baseline experiment is still an explicit incomplete systems requirement, to be separately frozen/executed before such claims. No running sources, treatment, stopping rule or holdout changed. Review: `results/thrml_spw/lfm3_1p7b/counted_mean_diagnostics/matched_resource_scope_review_20261030T042734Z.json`.


### 2026-10-30T04:29:41.898558+00:00 — adapted positional-word conversion law validated

Added and archived `adapted_word_conversion.py` and `preflight_adapted_word_conversion.py` before execution. The CPU component targets probabilities from the exact float32 field map used by programmed_worker, preserves lower/span ranges by changing only command arrays, and calibrates positional-word levels to the finite +/-4 sign probabilities. Unlike simply quantizing p and then applying finite sign noise, it inverts that finite endpoint contraction before nearest-word rounding. Independent 70-digit Decimal brute-force nearest-lattice enumeration matches all 4,096 commands for K=2/4/8 (12,288 conversions), with endpoint, monotonicity, error-bound and five invalid-input checks. No new GPU workload or change to the primary one-p-bit campaign. This is an internal candidate baseline format, not an Extropic hardware-format assertion. Actual adapted-package export, native trajectory/operator checks and a separately frozen full matched study remain incomplete; no neural-quality or hardware claim follows from this CPU component. Result `results/thrml_spw/lfm3_1p7b/adapted_word_conversion_preflight.json` (SHA256 38afc6d3b25ad39b11eb8ec858612c7fb6cb434dcc0113676c0241c97b6cd024).


### 2026-10-30T04:36:13.882975+00:00 — full adapted K4/K8 comparison package exported and independently accepted

CPU exporter session 24113 completed successfully, then independent verifier session 99967 completed successfully. Both were archived before execution. Exported all 286 canonical programs for each width, targeting exactly the mean-adapted finite-control expected weights using the calibrated converter, with unchanged range bytes and tied aliases. Independent 70-digit Decimal LUT checks validated every one of 3,457,603,072 output word coordinates across K4 and K8, all source/output seals, and every metadata byte. Metadata remains 108,251,440 bytes per width; actual saved word arrays use uint8 (1,728,801,536 raw word bytes per width), including K4, and must not be described as physically packed 4-bit memory. No full FP weight reconstruction was needed for conversion, and no GPU sampling job was added. This closes package preparation only: new-format native/operator validation and a frozen matched neural evaluation remain open. Main one-p-bit seed 202610352 is live as PID 999260/start 64813325, batch 3 of 4. No BF16-quality, spatial Pareto or hardware-performance conclusion. Package `results/thrml_spw/lfm3_1p7b/adapted_word_package/package.json`; accepted audit `results/thrml_spw/lfm3_1p7b/adapted_word_package/independent_audit.json`; root scope review `results/thrml_spw/lfm3_1p7b/adapted_word_package/root_review.json`.


### 2026-10-30T04:41:07.998869+00:00 — official THRML CPU adapted-word operator and raw evidence accepted

Added and archived `adapted_word_operator.py`, CPU preflight, raw-preserving preflight derivative, and separate NumPy-only raw verifier before their executions. Six cases span K4/K8, linear/depthwise layouts, fan-in 7/16/128/256, multiple groups, signed ACT12 extrema, and K4T1024/K8T512 matched high budgets. All 68,841,472 raw public sampled states are saved with integer inputs, words, ranges, per-plane counts/totals and outputs. Separate auditor checks every count and public-state int64 contraction exactly; a differently ordered temporal mean-weight readout has max absolute error 2.444721758365631e-09 (tolerance 1e-8). Raw-preserving repeats match every initial component output hash, so they are duplicate validation, not new quality seeds. Sampler/vendor commit checked (2d54cd8404df85bb65703d7e8b4e84e48c1b72a9). Observer carry is 4*bank_nodes bytes per active bitplane; replay additionally retains raw states/counts, and that carry size is not peak software memory. No full-model frontend, tiling/ACT scaling, native GPU or quality claim; these remain prerequisites for the matched adapted spatial study. No heavy GPU job added. Sources and raw evidence/audits are under `research/thrml_spw/lfm3` and `results/thrml_spw/lfm3_1p7b/adapted_word_operator_cpu*`; root review `results/thrml_spw/lfm3_1p7b/adapted_word_operator_cpu_root_review.json`.


### 2026-10-30T04:42:20.848495+00:00 — fourth original matched K8 seed accepted; fifth live

Original K8T128 seed 202610303 independently passed all1.729B unique-weight coverage, 512 saved-token metrics, deterministic controls and paired work accounting. Native NLL 4.620267493896344, perplexity 101.52118479507989; BF16 control NLL 4.6357237726129945. Retained work exactly matches paired K1T1024/K4T256 (1,772,027,117,568 evaluations), with eightfold single warmup separately charged. Root source/result and all saved metric-mean bindings checked. Four of five original K8 seeds accepted; final seed202610304 is live PID1014947/start64990670. No cohort contrast until all five and independent statistics; this is the original checkpoint/512-target experiment, not an adapted-model comparator. Root review `results/thrml_spw/lfm3_1p7b/word8_language/202610303/root_review.json`.


### 2026-10-30T04:48:09.384627+00:00 — tiled calibrated spatial worker CPU audit accepted

Added and archived `adapted_word_worker.py`, fixture driver and independent auditor before execution. The new worker directly consumes uint8 calibrated words and fixed FP32 lower/span metadata, caches official THRML count kernels, executes bitplanes sequentially and up to four independent 64-column tile lanes. CPU fixture session86863 and separate auditor completed successfully:16 mean/native requests, all raw trajectories preserved, 48,734,208 public sampled states checked, actual7680fanin plus257-channel multi-chunk/tail, conv7, norm and embedding layouts, zero input/range, K4T1024 and K8T512 component budgets. All counts/int64 contractions, ACT12 input scaling, tail padding, field law, output reconstruction and padding-inclusive retained/warmup/carry bookkeeping independently checked. Reordered readout max error 5.002220859751105e-12, tolerance1e-9. Core observer carry is explicitly distinguished from total runtime and replay-retention memory. This completes CPU worker integration only: native GPU preflight and separately frozen full adapted matched quality study remain open. Original full-model studies/one-p-bit hypothesis unchanged. Artifacts `results/thrml_spw/lfm3_1p7b/adapted_word_worker_cpu_preflight`, root binding review `results/thrml_spw/lfm3_1p7b/adapted_word_worker_cpu_preflight/root_review.json`.


### 2026-10-30T04:50:51.693311+00:00 — third full selected-mean native seed accepted

Seed202610352 independently accepted all1,728,801,536 unique programmed weights/all2,048 validation targets, canonical aliases, every operator/resource/output-loss receipt. Native NLL4.411901722461526/PPL82.42606605093889 versus same-fit BF16 NLL4.321675645982948/PPL75.31472343088639: PPL ratio 1.09442167873826689642228521727077638372145108045365879049088 (9.442% worse). Root recomputed all stage means from every saved token loss with Decimal; all three control records and resource receipts exactly match first seed. Three of five MC seeds accepted, no interim cohort CI or gate. Fourth seed202610353 is live PID1028478, start65043965. Root review `results/thrml_spw/lfm3_1p7b/counted_mean_validation/mean/202610352/root_review.json`. One logical p-bit per parameter unchanged, additional controls/ranges/counters explicit; holdout unscored and goal active.


### 2026-10-30T04:53:39.803378+00:00 — GPU tiled adapted-word worker independently verified

Archived GPU fixture/auditor derivatives before execution; worker source unchanged from CPU validation. After checking memory headroom, ran bounded GPU fixture session59207 via the existing CUDA wrapper (preallocation disabled), then independent NumPy-only audit session3565. All16 requests and 48,734,208 raw public sampled states passed; 7680fanin, norm/embedding/conv/linear, multi-chunk tails, zero cases, high budgets, exact int64 accumulations and all scheduled/carry work retained. Readout max independently reordered error 5.002220859751105e-12. Paired CPU/GPU inputs bitwise equal: True; compared48,734,208 raw state bits with0 mismatches and0 FP64 output bitpattern mismatches. These are paired implementation checks, not new independent model-quality seeds, physical hardware measurements or simulator speed evidence. Full matched adapted LFM3 baselines remain unexecuted and require prospective quality-study freeze. Primary one-p-bit run4 continues; three of5 accepted, holdout unscored. Evidence `results/thrml_spw/lfm3_1p7b/adapted_word_worker_gpu_preflight/independent_audit.json` and `results/thrml_spw/lfm3_1p7b/adapted_word_worker_gpu_preflight/root_review.json`.


### 2026-10-30T05:11:10.515962+00:00 — full adapted matched-resource spatial study frozen and queued

Implemented new full-model driver/case verifier as explicit derivatives of the accepted single-bit path, preserving original and same-fit BF16 controls before loading audited spatial words. Per-case audit requires exact F32 control bits for all4batches versus accepted single master, every original/word program/range/alias hash, all saved losses and per-operator retained-work equality to paired single; warmup separately Kfold. Paired K1/K4/K8 statistics and independent Decimal verifier pass synthetic pass/fail/zero-variance plus9tamper cases. Completion guard passes valid/six forged/incomplete receipt cases. All sources archived before fixture use/freeze.

Frozen `adapted_word_validation_protocol.json` SHA256ee74460f133552ef9f86b004499ab8fceba34c077ee1597c533d30bfeb2f1ff0, 48 source files/36 inputs, no full adapted spatial score seen. C1024: K4T256/K8T128 on exact same mean-trained point, all2048targets, five350..354 MCclusters, mandatory all outcomes. C4096: K4T1024/K8T512 on five420..424 only if original conditional one-bit extension executes and finishes its full audited cohort; includes negative outcomes and does not select by spatial scores. No raw holdout score.

Queue session1602 is live PID1047326, start65165496, currently `waiting_for_original_K8_and_all_single_T1024_seeds`. It waits the existing original K8 slot and complete single1024statistics, then checks>=18GiB available host and>=100GiB free disk before each model case. No extra full-model GPU job started. No retry/score stopping/threshold change, no energy or numerical-equivalence claim. Existing one-bit/uniform3arm/allocated/originalK8 campaigns unchanged. Queue review `results/thrml_spw/lfm3_1p7b/adapted_word_validation/root_queue_review.json`.


### 2026-10-30T05:20:34.669465+00:00 — pre-score holdout treatment eligibility rule fixed and guarded

Archived/tested `holdout_treatment_selection.py` and its synthetic fixture before rule freeze. Five eligibility outcomes and nine partial/forged cohort cases pass: incomplete seeds, wrong audit/hash/verifier/protocol, altered gate/interval/control, and nonfinite losses cannot qualify. Guard independently recomputes the five-seed Decimal t4 upper bound from accepted native NLL values versus the exact same adapted BF16 control and unchangedlog1.02margin. Lowest qualifying uniform T1024 or conditionalT4096, still exactly one p-bit per learned parameter; if neither qualifies, do not consume holdout. No decision from partial scores.

Policy `results/thrml_spw/lfm3_1p7b/holdout_treatment_selection_policy.json` SHA2561ef2bd4a1996be2178d6246eefa9b7cd64597aea5fb28e811f34c781a0d90f1b binds source/fixture, both current protocols/verifiers, mean-adapted package and accepted BF16 control, frozen holdout manifest/data/geometry audits. At freeze only three lower seeds accepted, higher/nativeholdout scores absent. No holdout token file read or model score run. Eligibility always returns score_holdout=False: a separate full execution/statistics protocol and fresh keys/all10240targets still required before any scoring. This is tested preparation, not a full holdout experiment or quality result. Existing pipelines unchanged and goal active.


### 20261030T052226Z — current LFM3 entry-point documentation reconciled

Preserved exact old root/research README bytes, then updated their current sections with three accepted individual mean-seed results, primary/conditional/matched protocol-state links, count-observer replay evidence, same-point comparison limits, untouched-holdout rule and current read-only status commands. Verified 48 local link targets. Historical milestone suffixes preserved byte-for-byte. This is documentation maintenance; no experimental source, current treatment, target, threshold, process or goal status changed.


### 2026-10-30T05:32:51.193867+00:00 — fourth full selected-mean native seed accepted, worse outcome retained

Seed202610353 independently accepted all1,728,801,536 unique programmed weights/all2,048 validation targets, canonical aliases and every operator/resource/output-loss receipt. Native NLL4.478352925724705/PPL88.08946323429369 versus same-fit BF16 NLL4.321675645982948/PPL75.31472343088639: PPL ratio1.16961809353426131531259311394939776803339126347287894747800 (16.962% worse). Root recomputed all stage means with Decimal from all saved token losses, checked source/result seals and exact equality of all three control records/resource receipts to the first case. Four of five MCseeds now accepted; this worse run is retained unchanged. No interim cohort CI/gate or favorable-seed filtering. Final seed202610354 live PID1118042/start65291236; root review `results/thrml_spw/lfm3_1p7b/counted_mean_validation/mean/202610353/root_review.json`. HigherT4096 and matched spatial queues still require full preceding cohorts. One p-bit per parameter unchanged, holdout unscored, goal active.


### 2026-10-30T15:29:19.109149+00:00 — resumed audit: both five-seed primary budgets complete; recovery of comparison bookkeeping

Revalidated OS identities after goal resume rather than restarting processes. Both primary cohorts completed with independent case/statistics audits while prior background jobs continued. T1024: meanNLL4.429824196023107, PPL83.9166627303495 vs same-fit BF1675.31472343088639; ratio1.114213249516303, conditional MC95ratio[1.0706182140984457,1.1595834528586897], fixed2percentgatefailed. T4096: meanNLL4.332080208884732, PPL76.1024309785281; ratio1.010458878579891, MC95ratio[0.9987880220603569,1.0222661092738152]. Thus average1.05percent worse, but upper2.2266percent exceeds the unchanged2percentcriterion: gatefailed, not equivalence. Four times retained work, warmup/controls/metadata unchanged; five distinct seed clusters per budget, never pooled. Root independently checked complete input/source/audit/statistic seals and Decimal gate arithmetic in `completed_onebit_cohorts_root_review.json`. Existing holdout selector returns neitherqualifies; no holdout scores.

All five original K8 native cases passed, but its final statistics audit correctly stopped on an input hash mismatch: generator had incorrectly treated live `pipeline.json` as immutable; parent advanced that file before verifier read it. All33 genuine immutable inputs matched. Old failed report, log and pipeline preserved. New `word8_language_statistics_v2` saves a fixed observation snapshot and rederives all15contrasts/resources; independent Decimal verifier passed. Every numerical field exactly equals old report. No sampler/model rerun or loosened tolerance. This repairs bookkeeping, not model quality.

The dependent original adapted-word queue had stopped before ANY model case. Created frozen recovery protocolV2 SHA2565f84ac8bd621c7258ab98e60db514b0831a77ad90e5bcf41f1f9e4c4673f05a1 with57source/47input bindings, unchangedpoint/targets/seeds/budgets/controls/margins. Namespace-only producer/verifier/statistic derivatives and recovery gating use the accepted immutable K8 audit. Nine statistics-tamper and six completion-receipt controls passed again. Original failures and namespace remain unchanged. New scheduler session24208 live PID1283106, child1283107, current `C1024_K4_202610350_adapted_word_validation_v2`. It is the first actual adapted spatial neural case, not a stochastic retry. Existing uniform and allocated studies remain live. Source/audit artifacts preserved; primary goal still one p-bit per parameter and active.


### 2026-10-30T15:37:36.671564+00:00 — new8192sample condition frozen/launched; unchanged quality margin

Both prior complete five-seed cohorts missed the fixed2percentupperMCt4qualitymargin, despite T4096mean1.05percentgap. Created new uniformT8192condition with five independent keys202610600..604, same prior-selected mean commands/ranges/ACT12/F64readout/BF16nonlinearities and all1,728,801,536parameters/2048targets. Namespace-only driver/case/statistic derivatives archived before use; isolated synthetic statistics fixture verifies both gate outcomes and tamper rejections. Preserved all prior outcomes. No old-budget extra-seed sampling, pooling, relaxed margin, retry or score stopping.

Existing GPU8192signed-counter audit and counted-bank/operator raw proofs covertheincreasedT. Maximumper-group absolute int32counter2146435072<2147483647, includingworstsignedACT12inputs; larger16384notvalidunderthisobserver. T8192costis2xT4096/8xT1024retainednodeevaluations, unchangedwarmup/counters/metadata. Protocol SHA256974b478da7420ee63ee930ee481fcf33f6583a8d0f62dc2d4ce3176105b93341; model scheduler session86865 livePID1289566, child1289567, statusrunning. Memory>=22GiBavailable/disk>=80GiBbeforeeachmodelcase. Matchedspatialrecovery continuesseparately; it onlycoversC1024/C4096, soC8192matchedcontrolsremainneededbeforeanyC8192systemsclaim.

Before any holdout scoring, explicitly versioned eligibility policy to add this new predeclared condition after two failed prior budgets; original policy unchanged. Seven valid outcomes and27partial/forgedcohortcases verified. Policy `results/thrml_spw/lfm3_1p7b/holdout_treatment_selection_v2_policy.json` SHA25659c591096aec1da833e6903042dcd47ad7f539a3c695681503cef4f846b2b581 usesexactoriginalcohortpredicate/margin and fixedadaptedBF16control; chooseslowestqualifying1024/4096/8192onlyafterallfiveseedsaccepted. Currentresult `pending_complete_T8192_cohort`, score_holdoutFalse. No holdout tokens read or scoresrun; a full execution/statistics protocol stillrequired. Root launch review `results/thrml_spw/lfm3_1p7b/counted_mean_t8192/root_launch_review.json`. Goalactive, onepbitperparameterunchanged.


### 2026-10-30T15:48:34.381390+00:00 — completed primary sampling/quality figure

Created standalone PNG/SVG plus exact plotted-data JSON for acceptedT1024/T4096, with approximateMCt4 intervals and original2percentupper-boundcriterion. AllNuniqueweights/sameadaptedpoint/2048fixedtargets/5independentkeys percondition; no pendingT8192point or hardware interpretation. Source/archive/inputstatistic-audit hashes checked. Root recomputed ratios with Decimal, checked plot labels/gates/data and every artifact hash, visually reviewed rendered PNG. V1 manifest included an invalid empty-file self-hash; preserved V1 and made V2 exclude self-reference before acceptance; PNG bytes identical. Correct artifact directory `results/thrml_spw/lfm3_1p7b/primary_sampling_figure_v2`, review `results/thrml_spw/lfm3_1p7b/primary_sampling_figure_v2/root_review.json`. No new neural evaluation or experimental-source change.


### 2026-10-30T15:52:42.756175+00:00 — third allocated-sampling seed independently accepted

Reviewed seed202610352 full2,048target/all1,728,801,536uniqueparameter audit, verifier/result seals, all native saved-token mean arithmetic and all three exact control-metric records versus uniform master. NativePPL80.77062201421141 vs same-fitBF1675.31472343088639, ratio1.07244132800051184402044492122056306159132365232725263043915. This individual run differs from same-seed uniformT1024 by-0.020288411981989929914736328125NLL; no paired cohort conclusion beforeall5. Actual retained7,044,962,713,600 <=uniform7,088,068,624,384, checked separately each4batches; warmupunchanged. Third of5accepted/fourthrunning, stillonepbitperparameter. No CI/gate/equivalence/hardware claim. Root review `results/thrml_spw/lfm3_1p7b/allocated_native/mean/202610352/root_review.json`.


### 2026-10-30T16:01:43.060178+00:00 — lossless raw-logit archive component for larger evaluation

Archived `lossless_logits.py`/fixture before execution; session29110 completed. Exact high16payload only ifeverylow16bit alreadyzero, otherwise fullfloat32payload, withrank/shape/dataSHA and version/type/schema checks. Exhaustively roundtripped all65,536high16patterns and1,048,576randomfull32patterns; IEEEboundary/subnormal/signedzero/NaNpayload/noncontiguous/empty/scalar cases pass. Six corrupt archives and two invalid input types rejected. Existing accepted T4096validation excerpt roundtrips byte-for-byte; independently rebuilt using rawbyte interleaving rather than codec decoder. Example8192rawbytes->4096payloadbytes (4222serializedwithheaders); no inference rounding. All raw tests preserved.

This prepares fullrawretention for the planned10,240targetevaluation without reducing its data scope; it is observation storage, not p-bit model compression or a hardware claim. Current neural sources/output archives unchanged, no holdout tokens or scores read. Root review `results/thrml_spw/lfm3_1p7b/lossless_logits_preflight/root_review.json`. Current8kfirstseedhascompleteditsfirstvalidationbatchandcontinues; no fullcase outcome yet.


### 2026-10-30T16:13:56.380449+00:00 — Complete reserved-evaluation batch coverage (preparation only)

Added and archived `holdout_batch_layout.py` and its independent synthetic fixture before execution. Metadata-derived plan covers all 96 short windows and 16 long windows in 16 batches (12 x 8x65, 4 x 4x257). Synthetic losses enumerate all 10,240 targets exactly once for each of five seed rows and recover 40 article clusters of 256 targets. Thirteen invalid record/manifest cases were rejected, including omissions, duplicates, overlap and nonfinite losses. Root independently enumerated every target and checked immutable source/manifest bindings. Artifacts: `lfm3_1p7b/holdout_batch_layout_preflight.json` and `lfm3_1p7b/holdout_batch_layout_root_review.json`. No reserved token arrays or predictions opened, no scoring or eligibility decision, and no changes to running/frozen experiments. This component does not establish forward-shape correctness or replace the pending complete scoring protocol.


### 2026-10-30T16:21:39.408480+00:00 — First adapted matched K4/C1024 seed accepted

The V2 recovered matched study completed and independently audited K4/T256 seed 202610350 on all 2,048 validation targets and all 1,728,801,536 unique parameter coordinates. Native NLL 4.507489884982709, PPL 90.69388048349951; same-fit BF16 NLL 4.321675645982948, PPL 75.31472343088639. Retained scheduled evaluations 7,088,068,624,384 exactly match paired K1/T1024 for every operator; warmup 27,687,768,064 is separately charged (4x K1). Root checked all four saved output hashes, source/protocol/result bindings, 2,048-loss arithmetic per stage, complete per-operator matched-work receipts and control summaries/weight hashes; independent verifier checked raw-logit losses and exact BF16 control bits. Supplemental root arithmetic check initially failed an overly tight 2e-15 tolerance; producer NumPy versus verifier Torch loss means differed by up to 6.22e-15, within the frozen verifier's existing 1e-12 tolerance. No experimental source, artifact or gate changed. Root review: `lfm3_1p7b/adapted_word_validation_v2/C1024/K4/mean/202610350/root_review.json`. This is one seed, with no cohort gate or comparative conclusion; full five-seed K4 and K8 cohorts at both budgets remain required. Pipeline automatically advanced to seed 202610351; no manual restart. Primary K1/T8192 remains running.


### 2026-10-30T16:25:54.348828+00:00 — Prospective long-context frontend geometry validated

Added `holdout_frontend.py` with prepare/restore helpers for linear, depthwise width-seven causal convolution, embedding and already-normalized elementwise activations. Archived source bytes before running `preflight_holdout_frontend.py` on CPU. Eight synthetic cases cover 8x65 and 4x257 at 2,048 activation channels (520/1,028 positions), and match independent canonical operators exactly using bounded dyadic inputs/metadata/probabilities; all truncated output shapes rejected. Fixture preserves the original hook's six extra zero output positions for convolution and repeated embedding lookup ordering. Artifacts: `lfm3_1p7b/holdout_frontend_preflight.json` and `lfm3_1p7b/holdout_frontend_root_review.json`. This is shape/algebra preparation only, not THRML sampling, ACT12, normalization, attention, BF16, full-model quality or heldout scoring validation. No reserved tokens opened, active/frozen writers unchanged.


### 2026-10-30T17:12:18.778384+00:00 — Draft complete reserved-evaluation runner and synthetic loss check

Implemented prospective `holdout_native_case.py`: complete 16-batch/10,240-target plan, all 286 canonical programs/287 uses, four controls/stages, variable-context frontend, exact lossless float32 output codec, chunked FP64 token losses and per-operator receipts. It requires a separately frozen complete execution/statistics protocol, source/input/checkpoint hashes, distinct five-key plan, independent verifier binding, and a complete qualifying validation cohort before loading reserved token arrays. No actual complete protocol exists yet and the main runner was not executed or adopted. Archived sources before executing `preflight_holdout_native_case.py`; its pure contracts rejected 18 invalid protocol/loss cases, and 1,536 synthetic losses at both context lengths agreed with separate scalar arithmetic (max error 7.11e-15). Root additionally recomputed losses with Torch cross-entropy (max error 7.11e-15), checked saved raw synthetic fixtures/source hashes, and archived `lfm3_1p7b/holdout_native_case_preflight/root_review.json`. This is component preparation only: main file-I/O/inference path, extended official-worker shapes, full-model forward integration, independent case verifier, statistical protocol and actual eligibility remain open. No reserved token arrays or predictions opened; active studies unchanged.


### 2026-10-30T17:16:49.056015+00:00 — Original uniform baseline arm completes all five seeds

All baseline-arm roots 202610350..354 in `adaptation_native_v2` are now independently accepted on all 2,048 targets and 1,728,801,536 unique coordinates. Root rechecked every audit/result/protocol/source binding, saved native-loss arithmetic and fixed resources, plus final seed's four raw output hashes. Mean NLL 5.1963058708803704, exp(mean NLL) 180.603834160185124874611244267279610405328918024106943717623; retained scheduled evaluations per seed 7,088,068,624,384, warmup 6,921,942,016. Original BF16 control NLL 4.978250316314612; original finite-program same-fit BF16 NLL 4.978662657873713. This original point differs from the adapted mean point, so it is not a same-fit representation-only comparison to adapted results. Saved `lfm3_1p7b/adaptation_native_v2/baseline_cohort_root_review.json`. Pipeline automatically advanced to mean seed 202610350, with all mean/binomial cases still required by original protocol. Original mean same-key runs duplicate the counted implementation's roots and will not be pooled as new MC seeds. Primary T8192 remains live at final head of seed600.


### 2026-10-30T17:18:06.100851+00:00 — First T8192 single-p-bit seed independently accepted

Uniform T8192 mean seed 202610600 completed all four batches and was independently audited on all 2,048 targets/all 1,728,801,536 unique parameter coordinates. Native NLL 4.313484919131014, PPL 74.70036057719075; exact same-fit BF16 control PPL 75.31472343088639; individual PPL ratio 0.991842725755218206535022581852643500513971723021896341415113. This is one seed only: no cohort gate outcome or BF16-equivalence claim. Root additionally checked all four raw output hashes, source/protocol/result bindings, complete saved-loss arithmetic and every operator's actual retained/warmup work against T4096; retained work 56,704,548,995,072 is exactly 2x T4096, warmup 6,921,942,016 unchanged. All deterministic controls and same-fit weight hashes agree with the accepted lower-budget reference. Root review: `lfm3_1p7b/counted_mean_t8192/mean/202610600/root_review.json`. The fixed pipeline automatically started fresh key 202610601; remaining keys601..604 and independent five-seed statistics are required. No early stopping, changed sampling, threshold relaxation or holdout scoring.


### 2026-10-30T17:19:02.909948+00:00 — Second adapted matched K4/C1024 seed accepted

K4/T256 root202610351 completed all 2,048 validation targets/all 1,728,801,536 unique coordinates and passed its independent audit. Native NLL 4.4984644294072105, PPL 89.87900971571202; exact same-fit BF16 control PPL75.31472343088639. Root rechecked all four raw output hashes, source/protocol/result bindings, every saved-stage loss mean and paired per-operator resource receipts: 7,088,068,624,384 retained scheduled evaluations (equal K1/T1024), 27,687,768,064 warmup (4x K1). Saved `lfm3_1p7b/adapted_word_validation_v2/C1024/K4/mean/202610351/root_review.json`. Two of five K4/C1024 roots accepted; no interim statistical/comparative claim. Pipeline automatically advanced to root202610352. Primary K1/T8192 root202610601 remains live; no holdout scoring.


### 2026-10-30T17:23:08.533044+00:00 — Official CPU worker validated at complete long-batch shape

Archived source bytes and verified clean pinned official THRML before running `preflight_holdout_worker_cpu.py` with CPU-only JAX/four-CPU affinity. Four synthetic 1,028-position native cases cover linear fanin128/T8192, linear fanin7680/T16, width7 depthwise convolution/T16 and elementwise/T8192, all with65channels including a padded final tile. Preserved every public sample, count, integer sum, input and output. Independent offline int64/F64 audit verified 151,009,280 public states, 8,289,792 integer entries and 267,280 output values exactly. Maximum absolute integer total 2,145,823,019 remained within the prospective int32 bound, and positive/negative high-magnitude mirrors matched. Root checked immutable source/artifact/audit bindings, clean pinned vendor checkout and independently derived total counts. Artifacts under `lfm3_1p7b/holdout_worker_cpu_preflight/`. This is CPU component validation only: no GPU long-shape check, full-model execution, heldout text, quality, energy or physical-latency evidence. Current frozen model writers unchanged; complete future scoring protocol and independent scoring verifier remain open.


### 2026-10-30T17:27:17.960389+00:00 — GPU long-shape worker audit and exact CPU replay

Replayed the frozen four-case CPU synthetic 1,028-position fixture on the official GPU worker using identical finite programs, inputs, keys, schedules and ACT12 settings. Sources archived before execution; CPU artifacts/audits hash-bound. Independent offline int64/F64 reconstruction verified all 151,009,280 public states, 8,289,792 integer entries and 267,280 outputs on GPU. Measured CPU-vs-GPU public-state differences: zero; output uint64 bit-pattern differences: zero. Maximum signed integer magnitude 2,145,823,019, same as CPU. Root rechecked source/protocol/result/raw artifact/CPU-reference hashes and cross-backend audit totals. Evidence under `lfm3_1p7b/holdout_worker_gpu_preflight/`. This closes the extended GPU worker-shape component check (7680fanin andT8192 included), without opening reserved tokens or changing active full-model writers. It does not establish full-model frontend/inference integration, heldout task quality, numerical BF16 equivalence or physical performance. Full frozen scoring/statistics protocol and independent scoring verifier still required.


### 2026-10-30T17:37:14.970571+00:00 — Independent future reserved-case auditor core validated on real accepted evidence

Implemented and archived `verify_holdout_native_case.py` before executing its preflight. Main auditor requires a complete frozen scoring/statistics protocol and qualifying validation evidence, checks canonical programs/aliases/point hashes, all per-operator scheduled-work receipts, lossless raw outputs, independent Torch FP64 losses for every reserved target and exact deterministic-control bits versus accepted first seed. Its main reserved-case path remains unexecuted. `preflight_verify_holdout_native_case.py` replayed the core resource checker against all 2,296 actual receipts from accepted K1/T8192 validation seed600 (four batches, all287weighteduses) and reproduced 56,704,548,995,072 retained evaluations and6,921,942,016 warmup exactly. It rejected ten deliberate omissions/duplicates/order/program/work/counter/precision/input corruptions. Saved a 17-target, full128,000-vocabulary excerpt of already-consumed validation logits; independent Torch losses agreed with saved producer losses within1.11e-13. Root checked source/reference/excerpt hashes, totals, and a separate long-double loss calculation (max discrepancy 1.09e-13). Evidence: `lfm3_1p7b/holdout_case_auditor_preflight/`. No new stochastic seed, model evaluation or heldout score. Full reserved scoring/statistics protocol, complete runner/auditor integration and actual eligibility remain required. Active studies unchanged.


### 2026-10-30T17:45:47.982280+00:00 — Prospective full holdout statistics collector and synthetic checks

Implemented `holdout_native_statistics.py`, with paired native-vs-same-fit-BF16 (primary), original BF16 and nominal-control NLL comparisons for all five seeds/all10,240targets. Uses existing independently tested article/MC uncertainty component, preserves24short/16long article clusters and60:40target weighting, reports each stratum separately. Draft tolerance check requires both conditional MC and crossed-bootstrap upper intervals below the unchanged log(1.02) margin in short,long andpooled reports; this is a prospective conservative heldout criterion, not a change to any current validation gate, and the final analysis protocol is not frozen. Archived source before preflight. Five synthetic complete-data outcomes (zero/small/large gap, long degradation masked by pooled mean, shared MC shocks) and five invalid/incomplete-control cases passed. Raw synthetic inputs preserved; root checked all hashes, per-seed stratum means and independent Decimal MC upper-bound arithmetic. Artifacts: `lfm3_1p7b/holdout_statistics_preflight/`. No heldout token arrays/outcomes or new model scores read; no coverage guarantee or BF16-equivalence claim. Independent full statistics verifier, final protocol and complete runner integration remain required before actual holdout scoring.


### 2026-10-30T21:48:07.401735+00:00 — Authorized Modal H100 execution underway

User explicitly authorized remaining work in parallel on H100. Authentication verified; Modal SDK1.6.3 now runs with Python3.12. Created versioned cloud sources and sealed image snapshots, separate scientific output namespace and fixed-source protocol. Preserved infrastructure failures (source snapshot mutation, initial bulk transfer connection failure, scratch-disk minimum, remote module path) and corrected them in versioned infrastructure. App `ap-Uh6vGxxM3asuaoQRdb4une` runs controller/asset hydration. Actual H10080GB native component passed at T8192 with torch2.14.1+cu130/CUDA13.0 and official THRML; pinned3.46GB checkpoint SHA verified. Parameter/control assets are uploading resumably; reference files use lossless gzip transport and must restore original hashes. First full-model K8/C1024/seed350 replay must match local all-four-stage bytes before13 fixed C1024/C4096 comparison cases use max4H100 workers. No cloud neural quality result exists yet. Local four pipelines continue unchanged. See `modal_h100_v1/deployment_status.json` and preserved launch/transfer logs.


### 2026-10-30T21:55:30.660244+00:00 — Independent full holdout analysis verified and frozen before scoring

Added/archived `verify_holdout_native_statistics.py` and its preflight before execution. Independent article/MC checker uses sampled seed/article frequency contractions rather than the producer's expanded-index resampling, and independently checks all9contrasts/strata, sample variance, percentile limits, PPL ratios, fixed tolerance decision and source/case/resource bindings. Nontrivial synthetic shared-MC/article/token variation passes; eight forged reports are rejected. Frozen `lfm3_1p7b/holdout_native_statistics_protocol.json` with fixed5MCseeds/all10,240targets/40articles,10,000bootstrapreplicates/key202610400 and unchanged log(1.02) tolerance for both upper bounds in short,long andpooled reports. This arithmetic freeze is not permission to score: full execution protocol and all-five-seed validation eligibility remain required. No reserved outcomes read.
