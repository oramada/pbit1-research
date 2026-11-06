# DGX Spark accepted core evidence · November 6

This bounded snapshot preserves three complete model/task quality checks, their audits and replays, and the layer/control studies completed with them. It is a local working-tree archive update; it has not been pushed.

| Model/task | Accepted result |
|---|---|
| GPT-2 / 283225 WikiText2 targets | Perplexity ratio1.00659; MC95[0.99927,1.01397] |
| DistilBERT / 25000 IMDb reviews | Accuracy87.4944% vs87.512%; both frozen gates pass |
| LFM3 / 294442 WikiText2 targets | Perplexity ratio1.00259; MC95[0.98751,1.01790] |

All use one logical p-bit per learned scalar and finite12bit probability commands. These are independent-binomial R1 reconstructions followed by conventional BF16 GPU execution. Native THRML evidence is at operator scale. The results establish neither universal numerical BF16 equality nor measured physical energy savings. The stronger four-element coarse/fractional control achieves lower operator error with more elements and stored information, so the density/time tradeoff remains explicit.

- [Three-model summary](artifacts/results/thrml_spw/spark_depth_20261106/cross_model_summary_v2/results.json)
- [LFM3 complete score/coverage audit](artifacts/results/thrml_spw/spark_depth_20261106/lfm3_benchmark_audit_v1/results.json)
- [LFM3 independent SDK replay](artifacts/results/thrml_spw/spark_depth_20261106/lfm3_benchmark_replay_v1/results.json)
- [Strong spatial control audit](artifacts/results/thrml_spw/spark_depth_20261106/native_coarse_fractional_v1_audit.json)
- [Workspace research report](artifacts/SPARK_RESEARCH.md) (its original links are workspace-relative)

The lower-work GPU test and conditional error-covariance study were still running/queued at this snapshot cutoff. Their result namespaces are excluded. Sources for planned studies are retained but are not evidence of execution.

Run `python verify_archive.py` here to check every included byte. `INDEX.json` records source paths and hashes; `OMITTED_ASSETS.json.gz` records hashes of raw arrays omitted from Git. Checkpoints and dataset tokens must be obtained under their own licenses. Copy the archived source trees from `artifacts/` into a fresh project workspace, restore the pinned assets named by each preparation protocol, and use the recorded environment versions and command modules. The raw-output audits require the omitted data or a complete rerun; summary JSON alone cannot reproduce them.

Historical root README, snapshot descriptor and hash list are retained under `history/`. Cloud scripts are historical sources; this update authorizes no cloud execution. All new work here used the local GB10 with a2GiB allocator cap.
