# Audited full-task R1 primary: mixed outcomes

The first full-task, five-seed **R1 primary results are mixed** at one logical p-bit per parameter and 8,192 samples:

| Model and full task | Original BF16 | PBit1 | Frozen quality gate |
|:---|---:|---:|:---|
| DistilBERT SST-2 checkpoint · 25,000 IMDb reviews | 87.512% accuracy | 87.532% accuracy | **Passed** · two-percentage-point margin |
| GPT-2 · 283,225 WikiText-2 targets | 25.5024 perplexity | 31.8679 perplexity | **Missed** · 2% relative perplexity margin |

Every learned parameter, including biases and normalization parameters, was converted. Independent audits replayed every evaluation batch from saved official THRML counts. The combined primary criterion failed; these results do **not** establish general BF16 task-quality equivalence. The original 104-case sweep continues unchanged. These are weight-reconstruction results reused across the test task, with dense BF16 software inference, rather than direct stochastic operator execution. [Primary statistics](https://github.com/oramada/pbit1/blob/main/benchmarks/r1_primary_20261105.json) · [Independent audit](https://github.com/oramada/pbit1/blob/main/benchmarks/r1_primary_audit_20261105.json).

Against original BF16, the GPT-2 perplexity ratio is **1.24960**, with conditional MC95 t4 interval **[1.20938, 1.29117]** and crossed article/seed bootstrap interval **[1.22172, 1.27513]**. Both upper bounds exceed 1.02. The DistilBERT error-rate gap is **−0.0200 percentage points**, with MC95 interval **[−0.11476, +0.07476] pp** and crossed review/seed bootstrap interval **[−0.11680, +0.07840] pp**. Both upper bounds are below the frozen +2 pp margin. The same pass/miss outcomes hold against finite-program-mean BF16 controls. Each model uses five independent reconstruction draws, evaluated separately; predictions are not ensembled.

The finite-mean GPT-2 control NLL is 3.237656 versus original BF16 3.238772, whereas sampled K1 NLL is 3.461599. This is consistent with finite-sample sensitivity, but does not yet identify a causal layer or exclude all implementation faults. A separate read-only count-moment diagnostic is checking the saved draws without new sampling. Any follow-up will retain the original failed primary condition, frozen margins and complete sweep. Group size is 128; probability commands have 12-bit precision, and both float32 range values per group are counted.

