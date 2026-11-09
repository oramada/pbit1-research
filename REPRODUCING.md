# Reproduce the work

There are four useful starting points. Begin with the ones that do not require
model weights or a GPU.

| Task | Needs | What it checks |
|---|---|---|
| Verify the repository | Python + Git | Published file hashes |
| Regenerate figures | NumPy + Matplotlib | Plots from bundled result summaries |
| Build the paper | LaTeX + Poppler + Python | Manuscript PDFs, references, and layout checks |
| Rerun model experiments | Exact checkpoints, data, and recorded runtimes | Original computations and saved-output audits |

## 1. Verify the files

From the cloned repository root:

```bash
python tools/verify_repository.py
python tools/check_guides.py
```

The manifest covers the committed artifact snapshot. Hash agreement proves that
files match that snapshot; it does not prove the scientific conclusions by itself.
The guide checker tests links in the current documentation, not frozen historical
archive pages.

## 2. Regenerate figures

Use a Python virtual environment if you want to isolate dependencies:

```bash
python -m venv .venv-figures
source .venv-figures/bin/activate
python -m pip install -r requirements-figures.txt
python tools/plot_overview.py
python paper/tools/build_figures.py
```

On Windows, activate with `.venv-figures\Scripts\activate`. These commands use
bundled JSON summaries and do not run neural networks. Rendering bytes can differ
between Matplotlib/font versions; source data hashes and reported scores remain
fixed. To check the original snapshot again, use a fresh checkout.

## 3. Build the paper

Install a TeX distribution with `latexmk`, `pdflatex`, BibTeX, Latin Modern, the
standard AMS packages, `natbib`, `booktabs`, `tabularx`, `placeins`, `fancyhdr`,
`microtype`, and `xurl`. PDF checks also use Poppler's `pdfinfo` and `pdftotext`.

```bash
cd paper
make
make verify
```

The committed [paper](paper/main.pdf) and [supplement](paper/supplement.pdf) can be
read without installing these tools. [Paper build notes](paper/README.md) explain
the figure inputs and visual checks.

## 4. Rerun model experiments

### Restore assets first

Full runs require the exact model checkpoints, tokenizer outputs, task layouts,
and software recorded in each experiment's `protocol.json`.

- Runtime records: [`archive/updates/20261106_spark_final/environments/`](archive/updates/20261106_spark_final/environments/).
- Historical dependency pins: requirement files under [`archive/`](archive/).
- Executed source bytes: [`results/thrml_spw/source_snapshots/`](results/thrml_spw/source_snapshots/).
- Omitted arrays: [`OMITTED_ASSETS.json.gz`](archive/updates/20261106_spark_final/OMITTED_ASSETS.json.gz).

Checkpoints, dataset tokens, full logits, and most raw trajectories are omitted
from Git. Restore them under their original licenses and verify their recorded
hashes. Some preparation steps expect local paths from the original workspace;
use the pinned protocol and source snapshot to resolve them in your new workspace.

### Choose the correct runner

| Evaluation | Producer | Audit | Replay |
|---|---|---|---|
| GPT-2 | [`benchmark_v1.py`](research/thrml_spw/spark_depth/benchmark_v1.py) | [`audit_benchmark_v1.py`](research/thrml_spw/spark_depth/audit_benchmark_v1.py) | [`run_benchmark_replays_v1.py`](research/thrml_spw/spark_depth/run_benchmark_replays_v1.py) |
| DistilBERT | [`distilbert_benchmark_v1.py`](research/thrml_spw/spark_depth/distilbert_benchmark_v1.py) | [`audit_distilbert_benchmark_v1.py`](research/thrml_spw/spark_depth/audit_distilbert_benchmark_v1.py) | [`replay_distilbert_v1.py`](research/thrml_spw/spark_depth/replay_distilbert_v1.py) |
| LFM3 | [`lfm3_benchmark_v2.py`](research/thrml_spw/spark_depth/lfm3_benchmark_v2.py) | [`audit_lfm3_benchmark_v1.py`](research/thrml_spw/spark_depth/audit_lfm3_benchmark_v1.py) | [`replay_lfm3_benchmark_v1.py`](research/thrml_spw/spark_depth/replay_lfm3_benchmark_v1.py) |
| Reduced-work GPT-2 | [`reduced_work_gpu_v1.py`](research/thrml_spw/spark_depth/reduced_work_gpu_v1.py) | [`audit_reduced_work_gpu_v1.py`](research/thrml_spw/spark_depth/audit_reduced_work_gpu_v1.py) | [`replay_reduced_work_gpu_v1.py`](research/thrml_spw/spark_depth/replay_reduced_work_gpu_v1.py) |

### Keep protocols intact

Run in a new workspace/output namespace. Accepted result directories are frozen;
producers can reject conflicting or already-failed state. Do not change executed
producer bytes to make old results pass. Planned, failed, and superseded scripts
remain in the archive, and their presence does not imply successful execution.

The original complete-model runs used a shared DGX Spark/GB10, serial research
GPU jobs, deterministic settings, and a 2 GiB PyTorch allocator cap. LFM3 used
CPU master weights and leaf-module offloading. This allocator cap is not a limit
on total host memory. Exact GPU replays can require the recorded runtime and
hardware; a raw-output audit needs the original arrays or a complete rerun.

Native THRML operator experiments use the separate THRML/JAX environment. The
full-model results use independent binomial reconstruction and BF16 neural
operators; they are not physical device measurements.

## Just want to try the package?

Use the current public SDK, which needs no model assets for its demo:

```bash
python -m pip install 'pbit1 @ git+https://github.com/oramada/pbit1.git@main'
python -m pbit1
```

The `sdk/` directory here is the older research snapshot, not the latest public
package. Its tested version and hashes remain in the original publication record.
