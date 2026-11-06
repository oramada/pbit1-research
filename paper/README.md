# Manuscript and supplementary material

- [Main manuscript](main.pdf): results, method, theory, diagnosed failures, controls, resource costs, and limitations.
- [Supplement](supplement.pdf): execution contracts, seed cohorts, layer/neuron diagnostics, covariance estimation, historical outcomes, and reproduction map.
- `main.tex`, `supplement.tex`, `preamble.tex`, `references.bib`: editable standard LaTeX sources.
- `figures/`: vector PDF plots generated from audited JSON plus preserved SVG/PNG diagnostic figures.
- `evidence/`: compact exact copies of the supporting data/protocols.
- `EVIDENCE_MANIFEST.json`: original paths and SHA-256 hashes for copied inputs and figures.

## Build

Requires a TeX installation with `latexmk`, `pdflatex`, BibTeX, Latin Modern, `natbib`, `booktabs`, `tabularx`, `placeins`, `fancyhdr`, `microtype`, `xurl`, and standard AMS packages. Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`) supplies PDF verification and rendering. Python figure regeneration requires NumPy and Matplotlib.

```bash
make                  # build both PDFs
make verify           # check inputs, numerical claims, final logs and PDF text
make figures          # rebuild five plots from bundled audited JSON
make                  # incorporate regenerated plots
```

Both PDFs are committed; reading them does not require TeX. Large checkpoint/token/logit assets are not included. Reproducing a figure from bundled data is different from rerunning a model experiment.

The main manuscript and supplement were rendered page by page and visually reviewed for clipping, missing plots, legibility, table layout and references. `verification/` records the exact checked PDF hashes. Build dates embedded in PDFs may differ across TeX runs; scientific input hashes remain fixed.

This is a research manuscript, not a peer-reviewed publication. The text distinguishes accepted full-task measurements, training-only diagnostics, historical failures, and proposed future work. No physical device result, universal BF16 numerical equivalence, or advantage over optimized quantization is asserted.
