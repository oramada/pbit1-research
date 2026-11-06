import numpy as np
import pytest
pytest.importorskip("thrml")
from pbit1 import NativeProbabilityBank

def test_official_sampler_endpoints_and_stationary_mean():
    bank = NativeProbabilityBank([0., 0.25, 0.5, 0.75, 1.])
    states = bank.sample(seed=7, n_samples=1024, n_chains=4)
    assert bank.logical_pbits == 5
    assert states.shape == (4, 1024, 5)
    assert (states[:, :, 0] == 0).all() and (states[:, :, -1] == 1).all()
    np.testing.assert_allclose(states.mean(axis=(0, 1)), bank.realized_probabilities, atol=0.04, rtol=0)

@pytest.mark.parametrize("weights", [[], [-0.1], [1.1], [float("nan")]])
def test_invalid_native_probabilities_rejected(weights):
    with pytest.raises(ValueError): NativeProbabilityBank(weights)
