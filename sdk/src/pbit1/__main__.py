"""Small official THRML demo: one logical p-bit for each probability weight."""
import argparse
import json
import numpy as np
from . import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    options = parser.parse_args()
    try:
        from .native import NativeProbabilityBank
    except ImportError as error:
        raise SystemExit('Install the official simulator with the GitHub [thrml] extra; see the README') from error
    weights = np.array([0., .125, .25, .5, .75, .875, 1.])
    inputs = np.array([1., -.5, .25, 1., -.25, .5, 1.])
    bank = NativeProbabilityBank(weights)
    states = bank.sample(options.seed, options.samples, options.chains)
    outputs = states.astype(np.float64) @ inputs
    print(json.dumps({"version": __version__, "backend": "official THRML",
        "controls": bank.controls(), "weights": weights.tolist(),
        "realized_probabilities_from_float32_controls": bank.realized_probabilities.tolist(),
        "sample_means": states.mean(axis=(0, 1)).tolist(),
        "samples_per_chain": options.samples, "independent_chains": options.chains,
        "expected_dot_product": float(weights @ inputs),
        "sampled_output_mean": float(outputs.mean()),
        "scope": "One logical p-bit per parameter, programmable controls and sampling retained. Endpoints use clamped nodes. CPU simulator demonstration; no BF16-equivalence or hardware-efficiency claim."}, indent=2))


if __name__ == "__main__":
    main()
