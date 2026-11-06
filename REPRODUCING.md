# Reproduction contracts

The paper's compact figures can be rebuilt independently of model checkpoints:

```bash
python -m pip install numpy matplotlib
cd paper
python tools/build_figures.py
make
```

Full experiments require the precise checkpoints, tokenizer outputs, task layouts, and packages named in each `protocol.json`. Consult `archive/updates/20261106_spark_final/environments/` for recorded packages, the pinned requirements in `archive/`, and the per-experiment source hashes. Source files are also preserved under `results/thrml_spw/source_snapshots/` using their SHA-256 as filename.

The bounded archive omits large arrays and model/dataset assets. `archive/updates/20261106_spark_final/OMITTED_ASSETS.json.gz` lists omitted array hashes. Original model and dataset licenses still apply. A successful manifest check establishes byte integrity, not the scientific validity of a result.

Main model evaluation contracts:

- GPT-2: `research/thrml_spw/spark_depth/benchmark_v1.py`; audit `audit_benchmark_v1.py` and replay sources named by the corresponding protocols.
- DistilBERT: `research/thrml_spw/spark_depth/distilbert_benchmark_v1.py`; model-specific audit/replay modules in the same directory.
- LFM3: `research/thrml_spw/spark_depth/lfm3_benchmark_v2.py`, `audit_lfm3_benchmark_v1.py`, `replay_lfm3_benchmark_v1.py`.
- Reduced-work GPT-2: `research/thrml_spw/spark_depth/reduced_work_gpu_v1.py`, `audit_reduced_work_gpu_v1.py`, `replay_reduced_work_gpu_v1.py`.

Use a fresh workspace/output namespace and restore assets before running producers. Do not overwrite accepted experiment directories or edit a frozen producer to make old results pass. Some source files represent failed, superseded, or unexecuted protocols; presence in the archive does not imply successful execution.

The accepted complete model experiments used a shared DGX Spark/GB10, serial research GPU jobs, deterministic settings, and a 2 GiB PyTorch allocator cap. That cap describes GPU allocations, not total host memory. LFM3 uses CPU master weights and leaf-module offloading. Native THRML operator experiments used the separate THRML/JAX environment. Full models use independent binomial-count simulation and BF16 neural arithmetic; no full-model native THRML or physical chip run is claimed.

The public SDK can be tested without the full research assets:

```bash
python -m pip install 'git+https://github.com/oramada/pbit1.git@main'
```

Add the `thrml` extra for the pinned official sampler. The prior archive includes verified development wheel and source-distribution hashes; current package publication records are kept separately from those frozen artifacts.
