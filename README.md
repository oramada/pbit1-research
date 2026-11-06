# One P-Bit per Parameter — research and manuscript

Complete, private research companion to the public [PBit1 package](https://github.com/oramada/pbit1).

**[Read the manuscript](paper/main.pdf)** · **[Supplementary methods and diagnostics](paper/supplement.pdf)** · **[LaTeX source](paper/main.tex)** · **[Research report](SPARK_RESEARCH.md)**

The paper studies when one logical programmable p-bit per learned scalar preserves pretrained-model task quality, why some configurations fail, and what the controller, range, and sampling costs imply. It includes positive and negative results. The work is independent of Extropic and uses its official THRML sampler for operator-scale experiments.

## Main observations

| Model/task | Result versus original BF16 |
|---|---|
| GPT-2 / 283,225 WikiText-2 targets | PPL ratio 1.00659; five-seed MC95 [0.99927, 1.01397] |
| DistilBERT / 25,000 IMDb reviews | Mean accuracy 87.4944% versus 87.512%; both frozen quality gates pass |
| LFM3-1.7B / 294,442 WikiText-2 targets | PPL ratio 1.00259; MC95 [0.98751, 1.01790] |
| GPT-2 with sensitivity-based sampling | 41.7% fewer nominal reconstruction observations; upper MC95 ratio 1.01660 |

All selected full-model conditions use one logical p-bit per learned parameter, including biases and normalization parameters, finite 12-bit commands, and shared range metadata. Main model runs reconstruct weights and use ordinary BF16 neural operators. These are three model/task evaluations on two previously inspected datasets. They do **not** establish universal numerical BF16 equality, a one-bit model file, physical hardware savings, or superiority over optimized quantization.

## Repository layout

- `paper/`: standard LaTeX manuscript, supplement, compiled PDFs, figures, bundled figure inputs, and verification/build instructions.
- `research/`, `weight_encodings/`, `operators/`, `stochastic/`, `models/`, `analysis/`, `configs/`: research implementations and experiments.
- `results/`: compact accepted evidence, frozen protocols, audits, and immutable source snapshots.
- `archive/`: preserved experimental archive, including prior positive, failed, interrupted, and unexecuted work. Its older manifests apply within that directory and are not rewritten here.
- `sdk/`: source snapshot of the corresponding public development package; the public repository is the canonical package destination.
- `master_design.tex`: original experimental plan; proposed phases are not automatically completed experiments.
- `MANIFEST.sha256.json`: integrity manifest for this repository snapshot, excluding Git internals and LaTeX build intermediates.

Large raw arrays, complete model weights, and dataset token files are excluded from Git. Omitted assets are listed by hash in the preserved archive. Model/dataset licenses are separate. Summary JSON is not a replacement for raw-output reproduction.

## Build and inspect the paper

```bash
cd paper
make
make verify
```

See [paper/README.md](paper/README.md) for dependencies, figure regeneration, and PDF checks. The two PDFs are committed for convenient reading.

## Reproduce experiments

See [REPRODUCING.md](REPRODUCING.md). Restore the exact pinned assets and environments before invoking model-scale experiments; run in a fresh output namespace. The repository preserves source versions that deliberately reject incompatible or already-failed protocols.

## Licensing and status

Original code is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). No claim of Extropic affiliation or endorsement is made. This repository remains private; sharing its URL does not grant access. Public package changes are published separately.
