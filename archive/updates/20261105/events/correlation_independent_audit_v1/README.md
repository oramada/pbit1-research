# Independent correlation evidence audit

## Correlation audit and nonideality boundary

An independent audit now rebuilds all nine existing small Ising dot-product models and **108 conditions**, checking **6,912,000 saved native state bits**, all lag-covariance matrices and every empirical table metric. It constructs Gibbs updates from Boltzmann conditional distributions and obtains finite-sample variance through a separate Markov-reward recursion. All checks pass; corruption fixtures reject an altered MSE, a removed off-diagonal covariance term and a missing condition. This verifies stored evidence and mathematical accounting; it does not independently replay native random bits.

![Input direction changes correlated sampling cost](https://github.com/oramada/pbit1/blob/main/docs/assets/correlation_limits_20261105.svg)

Across these calibrated two-, three- and four-spin cases, exact output variance is **0.23969×–7.31955×** the independent-sample prediction. The average absolute empirical/predicted MSE-ratio error is **3.126%** across the 108 conditions. Covariance can cancel or amplify noise depending on the input; no single hardware penalty factor follows from this result. [Independent audit](https://github.com/oramada/pbit1/blob/main/benchmarks/correlation_audit_20261105.json) · [Corruption fixtures](https://github.com/oramada/pbit1/blob/main/benchmarks/correlation_audit_fixtures_20261105.json).

Earlier archived calibration and drift studies cover synthetic nonideal operators and a selected quarter-model transformer, with explicit probe work and drift clocks. They do not establish robustness of the additional full-model R1 primaries under calibration errors, drift or coupling. These remain separate from the current ideal-unary full-task results; physical device energy, switching rate and recalibration intervals require actual device measurements.


Execution note: the first source-snapshot copy used a nonexistent directory; the read-only auditor still ran successfully. Its immutable source snapshot was then saved to the established archive, after that first execution. Corruption fixtures and plot sources were snapshotted before their execution. Plotting first failed in the Torch environment because matplotlib was absent, then succeeded in the existing THRML environment; scientific inputs were unchanged.
