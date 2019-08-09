"""Tests the scipy.special bindings
"""
import numba_scipy.special  # noqa
from scipy.special import beta
from numba import njit
import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays as st_arrays


@given(st.floats(0, 20), st.floats(0, 20))
def test_beta_scalars(x, y):
    @njit
    def test_impl(a, b):
        return beta(a, b)

    np.testing.assert_allclose(test_impl(x, y), test_impl.py_func(x, y))


@given(
    st_arrays(np.float, 10, st.floats(0, 20)), st_arrays(np.float, 10, st.floats(0, 20))
)
@settings(deadline=None)
def test_beta_arrays(x, y):
    @njit
    def test_impl(a, b):
        return beta(a, b)

    np.testing.assert_allclose(test_impl(x, y), test_impl.py_func(x, y))
