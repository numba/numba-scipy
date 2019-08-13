import os
import json
import itertools

import pytest

import numpy as np
import numba
import scipy.special as sc
import numba_scipy.special

SIGNATURES_FILE = os.path.join(
    os.path.dirname(numba_scipy.special.__file__),
    'signatures.json'
)

CTYPES_TO_TEST_POINTS = {
    'c_double': [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0],
    'c_float': np.array(
        [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0],
        dtype=np.float32
    ),
    'c_long': [-100, -10, -1, 0, 1, 10, 100],
}

SKIP_LIST = {
    'hyperu'  # Should be fixed by https://github.com/scipy/scipy/pull/10455
}


def get_signatures():
    with open(SIGNATURES_FILE) as f:
        signatures = json.load(f)
    return signatures


def get_parametrize_arguments():
    signatures = get_signatures()
    for name, specializations in signatures.items():
        for ctypes_signature in specializations.values():
            # The first value in `ctypes_signature` is the return
            # type, which we don't need to evaluate the function.
            yield name, ctypes_signature[1:]


@pytest.mark.parametrize(
    'name, ctypes_args',
    get_parametrize_arguments(),
)
def test_function(name, ctypes_args):
    if name in SKIP_LIST:
        return

    f = getattr(sc, name)

    @numba.njit
    def wrapper(*args):
        return f(*args)

    args = itertools.product(*(
        CTYPES_TO_TEST_POINTS[ctype] for ctype in ctypes_args
    ))
    for arg in args:
        print(arg)
        overload_value = wrapper(*arg)
        scipy_value = f(*arg)
        if np.isnan(overload_value):
            assert np.isnan(scipy_value)
        else:
            assert overload_value == scipy_value
