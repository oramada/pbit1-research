"""Report a separate prospective numerical check without rewriting old gates."""
import json
from pathlib import Path
ROOT=Path('results/thrml_spw/range_localization')
def main():
    out=ROOT/'fixed_compilation';a=json.loads((out/'evidence_audit.json').read_text());assert a['completed'];diag=ROOT/'reproducibility_diagnostic';cold=json.loads((diag/'cold/result.json').read_text());prefix=json.loads((diag/'prefix/result.json').read_text())
    text=f'''# Fixed compiler configuration: numerical reproducibility

The separate prospective numerical check passed: **{a['passed']}**. Across all 24 Transformer affine projections and three frozen range/sample conditions (72 cases, seed0), production outputs from two independently started processes are byte-identical: **{a['all_cross_process_exact']}**. Independent public-factor/SumOnly outputs match production in both processes: **{a['all_public_sums_exact']}**. All independent public integer-count hashes match the original audit: **{a['all_public_counts_match_previous']}**.

This check uses `XLA_FLAGS=--xla_gpu_autotune_level=0`, CUDA JAX 0.12.0.1, default F32/highest matmul precision, and the same pinned official THRML 0.1.4. It changes compiler configuration; it does not change the sampler, fields, grouping, keys, sample counts or inference equations. The original 360-case study's default-compiler byte-exact gate **remains failed** (270 cases). This separate result does not retroactively pass that gate or change any neural quality threshold.

## Diagnosis and limits

At the fixed stage1/group128/T1024/seed0 diagnostic, repeated calls to one compiled production executable are exactly identical. Production integer counts and an independently constructed public THRML frequency observer are exactly identical, and their hashes match the earlier audit. A cold execution differs from the preserved original output in {cold['preserved_output_difference']['differing_values']:,} values, with maximum absolute difference {cold['preserved_output_difference']['max_abs_difference']:.9g}. Repeating all 15 preceding stage0 calls reproduces those preceding outputs but gives the same stage1 result as cold execution, not the original stage1 output. Thus the tested prefix order alone does not explain the discrepancy.

Two fresh pilot processes with autotuning disabled give byte-identical outputs and unchanged integer counts. A protocol frozen before the broader check then covers all 24 projections and all three original control/sample conditions in two separate processes. Production and public sums, raw outputs, control/input/target hashes, counts and resource counters are retained or checked. Same-output public arrays share the preserved production bytes; different public outputs, if any, are saved separately. An independent CPU auditor checks actual cross-process arrays and recomputes target MSE.

Timing-dependent kernel selection is a plausible explanation consistent with these observations. Original compiled executables were not saved, so the historical cause is not uniquely established. Optimized and unoptimized HLO are retained for the four pilot diagnostics. The numerical result is scoped to this backend, checkpoint, 64 input rows, three configurations and seed0; it is not a universal determinism guarantee.

Maximum absolute relative target-MSE change between the first fixed-compiler process and the old seed0 outputs is **{a['max_abs_relative_MSE_change_from_original']:.9g}**. This diagnoses arithmetic reproducibility rather than a new accuracy improvement. The prior five-seed operator MSE contrast and its unresolved historical gate remain reported unchanged in [the range-localization report](RANGE_LOCALIZATION_REPORT.md).

## What remains

This check performs no full stochastic neural forward or new held-out quality evaluation. All 24 projections use real FP32 teacher inputs from consumed validation contexts. It does not test accumulated student activation noise, embedding/head conversion, new models or physical hardware. One logical p-bit per normalized weight still requires finite bias control plus range metadata. No physical energy, latency or Extropic-superiority claim follows.

The next full stochastic end-to-end confirmation needs at least 10 GB free persistent storage or another persistent path; the current research volume has less than 0.5 GB available. The resource request remains pending. The long-horizon goal remains active.

Native observed node updates: {a['cohort_native_updates']:,} for the two broader processes and {a['diagnostic_native_updates']:,} for four pilots, total **{a['this_study_native_updates']:,}**. These are simulation counters, not measured hardware costs.

## Exact original commands

```bash
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.range_replay_diagnostic --mode cold
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.range_replay_diagnostic --mode prefix
XLA_FLAGS=--xla_gpu_autotune_level=0 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.range_replay_fixed_compilation --replicate 0
XLA_FLAGS=--xla_gpu_autotune_level=0 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.range_replay_fixed_compilation --replicate 1
XLA_FLAGS=--xla_gpu_autotune_level=0 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.range_fixed_replay_cohort --replicate 0
XLA_FLAGS=--xla_gpu_autotune_level=0 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m research.thrml_spw.gpu_environment -- .venv-thrml-gpu/bin/python -m research.thrml_spw.range_fixed_replay_cohort --replicate 1
.venv/bin/python -m research.thrml_spw.verify_fixed_range_compilation
.venv/bin/python -m research.thrml_spw.range_fixed_replay_report
```

Native writers refuse to overwrite completed evidence. These are the original invocations, not cache-only rerun instructions. The independent CPU audit can be repeated on the preserved arrays. Protocol: `range_localization/fixed_compilation/protocol.json`; raw manifests and arrays: `replicate0/` and `replicate1/`; evidence: `evidence_audit.json`; frozen dependency hashes: `execution_sources.json`.
'''
    Path('results/thrml_spw/RANGE_REPRODUCIBILITY_REPORT.md').write_text(text);print(json.dumps({k:v for k,v in a.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
