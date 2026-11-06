import numpy as np
import pytest
from pbit1 import FinitePBitProgram


def test_constant_groups_are_exact_and_payload_counted():
    p = FinitePBitProgram.encode(np.full((3, 7), .125), group_size=7)
    np.testing.assert_array_equal(p.expectation(), np.full((3, 7), .125, np.float32))
    np.testing.assert_array_equal(p.reconstruct(1, seed=2).weights, p.expectation())
    assert p.storage()['range_metadata_bytes'] == 24
    assert p.storage()['actual_program_array_bytes'] == 42 + 24


@pytest.mark.parametrize('layout', ['storage_flat', 'transposed_matrix'])
def test_layout_reconstructs_original_shape_and_is_seeded(layout):
    w = np.linspace(-1, 2, 323).reshape(17, 19)
    p = FinitePBitProgram.encode(w, group_size=7, layout=layout)
    a = p.reconstruct(32768, seed=81, parameter_index=3)
    b = p.reconstruct(32768, seed=81, parameter_index=3)
    assert a.weights.shape == w.shape
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.counts_sha256 == b.counts_sha256
    assert p.reconstruct(32768, seed=82, parameter_index=3).counts_sha256 != a.counts_sha256


def test_high_sample_count_does_not_wrap():
    p = FinitePBitProgram.encode([0., .5, 1.], group_size=3)
    a = p.reconstruct(131072, seed=4)
    assert a.counter_dtype == 'uint32'
    assert a.maximum_count > 65535
    assert a.weights[-1] > .99
    assert p.reconstruct(65535, seed=4).counter_dtype == 'uint16'


@pytest.mark.parametrize('weights,kwargs', [([], {}), ([np.nan], {}), ([np.inf], {}), ([1], {'group_size':0}), ([1], {'layout':'transposed_matrix'})])
def test_rejects_invalid_encoding(weights, kwargs):
    with pytest.raises(ValueError):
        FinitePBitProgram.encode(weights, **kwargs)


@pytest.mark.parametrize('samples', [0, 2**32, 1.5])
def test_rejects_invalid_budget(samples):
    with pytest.raises(ValueError):
        FinitePBitProgram.encode([0., 1.]).reconstruct(samples, seed=1)


def test_rejects_unrepresentable_fp32_range():
    with pytest.raises(ValueError, match="ranges"):
        FinitePBitProgram.encode([-np.finfo(np.float32).max, np.finfo(np.float32).max])
