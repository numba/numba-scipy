import itertools

import pytest

import numpy as np
import numba
import scipy.special as sc
import numba_scipy.special
from numba_scipy.special import signatures as special_signatures
from numba_scipy.special import types as special_types

NUMBA_TYPES_TO_TEST_POINTS = {
    numba.types.float64: np.array(
        [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0],
        dtype=np.float64
    ),
    numba.types.float32: np.array(
        [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0],
        dtype=np.float32
    ),
    special_types.numba_long: np.array(
        [-100, -10, -1, 0, 1, 10, 100],
        dtype=np.int_
    )
}

SKIP_LIST = {
    'hyperu'  # Should be fixed by https://github.com/scipy/scipy/pull/10455
}


def get_parametrize_arguments():
    signatures = special_signatures.name_to_numba_signatures.items()
    for name, specializations in signatures:
        for signature in specializations:
            yield name, signature


@pytest.mark.parametrize(
    'name, specialization',
    get_parametrize_arguments(),
)
def test_function(name, specialization):
    if name in SKIP_LIST:
        return

    f = getattr(sc, name)

    @numba.njit
    def wrapper(*args):
        return f(*args)

    args = itertools.product(*(
        NUMBA_TYPES_TO_TEST_POINTS[numba_type] for numba_type in specialization
    ))
    for arg in args:
        overload_value = wrapper(*arg)
        scipy_value = f(*arg)
        if np.isnan(overload_value):
            assert np.isnan(scipy_value)
        else:
            assert overload_value == scipy_value
