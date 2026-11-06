# Higher-sampling preparation only

## Higher-sampling follow-up preparation

A distinct GPT-2 follow-up is being prepared at T262144, retaining one logical p-bit per scalar. It has **no completed model-quality result**. A native primitive fixture checks uint32 saved-count round-trips and independently decodes public THRML states at T8192 and T262144. CPU and H100 agree bit-for-bit on all **6,488,064 checked states**, integer counts and decoded FP32 values. It also confirms that uint16 would corrupt counts at the higher budget. [Bridge audit](https://github.com/oramada/pbit1/blob/main/benchmarks/highT_primitive_bridge_20261105.json).

This is an eight-lane fixture, separate from the production 65,536-lane timing check now running. Full-model scoring requires complete integration and statistical gates and a frozen protocol. The candidate uses the already-exposed GPT-2 test corpus with fresh stochastic keys; it cannot be described as a new-text holdout. T262144 costs 32 times the retained stochastic work of T8192, while preserving logical cell count. The original failed T8192 primary remains reported unchanged.

