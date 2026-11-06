# Numerical precision and task quality are different requirements

A single programmable p-bit can encode a scalar in its stationary probability. This does not guarantee that a finite sample reproduces the scalar's BF16 bit pattern, or that a finite controller can program every BF16 value.

The executable counterexample holds the group range fixed at [0, 1]. Two arrays of exact BF16 values differ only at one coordinate (2^-20 versus 2^-19). The current 12-bit, field-clipped encoder produces identical complete codes and range metadata for both arrays. Its asymptotic output is therefore identical for two different targets. Changing the range/controller protocol may repair this particular collision, at a cost that must be counted.

Separately, even an ideal probability controller has finite-sample uncertainty. The accompanying exact binomial calculation gives the chance of rounding an estimate of p = 0.75 to the correct BF16 value. It is an illustrative numerical test, not a prediction of neural-network accuracy.

For distinct interior probabilities, the same finite binary observation sequence can occur under either one. Observations alone cannot guarantee exact recovery for both. If a decoder instead reads an exact stored parameter from the controller, that stored information must be counted.

None of these arguments rule out matching BF16 **task quality**, which remains the empirical research target. They delimit a stronger, universal numerical-equality claim.
