# Changelog

## 0.1.0a2.dev1 — November 6, 2026

- Add `FinitePBitProgram` for finite 12-bit probability commands, explicit range metadata, selectable matrix grouping, and independent-binomial reconstruction.
- Preserve count hashes and use uint32 counters when the sampling budget exceeds uint16 capacity.
- Export the finite-program API and report the actual package version through the CLI.
- Add finite-program validation tests and update audited GPT-2, DistilBERT, and LFM3 results.
- Document the repaired GPT-2 grouping failure, reduced sampling policy, covariance diagnostics, and stronger multi-element controls.

This development update establishes tested task-quality regimes. It does not claim universal BF16 numerical equivalence, a one-bit total memory footprint, or physical hardware savings. The earlier tagged 0.1.0a1 release and its results remain available.
