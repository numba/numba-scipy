"""Tests the scipy.special bindings
"""
import numba_scipy.special # noqa
from scipy.special import beta
from numba import njit
import numpy as np


def test_beta():

    @njit
    def test_impl(a, b):
        return beta(a, b)

    x = np.linspace(0, 20, 50)
    y = x[::-1]
    np.testing.assert_allclose(test_impl(x, y), test_impl.py_func(x, y))
