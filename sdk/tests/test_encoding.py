import numpy as np
import pytest
from pbit1 import SinglePBit

def test_constant_groups_are_exact():
    w = np.array([[-0.5, 0.75], [-0.5, 0.75]])
    groups = np.array([[0, 1], [0, 1]])
    codec = SinglePBit(eps=0)
    code = codec.encode(w, groups)
    np.testing.assert_array_equal(codec.decode(codec.sample(code, 3, np.random.default_rng(3))), w)

def test_linear_estimator_agrees_with_expected_operator():
    w = np.array([[-0.8, 0.2], [0.1, 0.9], [-0.3, 0.5]])
    x = np.array([[0.5, -0.25, 0.75], [-1., 0.25, 0.1]])
    codec = SinglePBit(eps=0)
    y = codec.forward(x, codec.encode(w), 32768, np.random.default_rng(7))
    np.testing.assert_allclose(y, x @ w, atol=0.025, rtol=0)

@pytest.mark.parametrize("eps", [-1, 0.5, 1])
def test_invalid_probability_clipping_rejected(eps):
    with pytest.raises(ValueError): SinglePBit(eps=eps)
