# DGX Spark completed research snapshot · November 6

This update preserves the completed local investigation: three full model/task checks, causal and covariance diagnostics, matched-noise controls, stronger spatial controls, and the lower-sampling policy. It is prepared locally and has not been pushed.

| Model/task | Accepted result |
|---|---|
| GPT-2 / 283225 WikiText2 targets | PPL ratio 1.00659; MC95 [0.99927, 1.01397] |
| DistilBERT / 25000 IMDb reviews | Accuracy 87.4944% versus 87.512%; both gates pass |
| LFM3 / 294442 WikiText2 targets | PPL ratio 1.00259; MC95 [0.98751, 1.01790] |
| GPT-2 lower-sampling policy | 41.7% fewer nominal observations; PPL ratio 1.00780, upper MC95 1.01660 |

All tested neural representations retain one logical p-bit per learned scalar, with finite probability commands and shared range metadata. Whole-model tests reconstruct sampled weights and run conventional BF16 neural operators. Native THRML evidence is at operator scale. Universal numerical BF16 equality and physical energy savings have not been established.

- [Consolidated research report](artifacts/SPARK_RESEARCH.md)
- [Three-model summary](artifacts/results/thrml_spw/spark_depth_20261106/cross_model_summary_v2/results.json)
- [Lower-work audit](artifacts/results/thrml_spw/spark_depth_20261106/reduced_work_gpu_v1_audit.json)
- [Lower-work replay](artifacts/results/thrml_spw/spark_depth_20261106/reduced_work_replay_v1/results.json)
- [Conditional covariance audit](artifacts/results/thrml_spw/spark_depth_20261106/lfm3_error_covariance_v2_audit.json)
- [SDK build/install verification](artifacts/results/thrml_spw/spark_depth_20261106/sdk_build_verification_v2.json)

Run `python verify_archive.py` here to check the snapshot. The source tree includes immutable content-addressed versions, and the SDK project and development distributions are included. Environment package versions are recorded under `environments/`.

Most large raw arrays are omitted from Git and identified by hash in `OMITTED_ASSETS.json.gz`; checkpoint weights and dataset tokens must be restored under their own licenses. The compact covariance sufficient statistics are included. Raw-output audits require the omitted assets or complete reruns; summary JSON alone does not reproduce those audits. To rerun, copy the relevant source trees from `artifacts/` into a fresh project workspace and restore assets using the pinned preparation protocols. Existing output directories use exclusive creation or frozen contracts and should not be overwritten.

The earlier core snapshot and original root history remain intact. Planned and failed sources are retained for provenance; their existence does not establish completed experiments. No cloud execution or physical hardware access is part of this snapshot.
