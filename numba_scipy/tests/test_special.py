import itertools
import warnings

import pytest
from unittest.mock import patch, Mock

import numpy as np
from numpy.testing import assert_allclose
import numba
from numba.types import float64
import scipy.special as sc
from numba_scipy.special import signatures as special_signatures
from numba_scipy.special.signatures import (parse_capsule_name,
                                            de_mangle_function_name,
                                            get_signatures_from_pyx_capi,
                                            generate_signatures_dicts,
                                            )

NUMBA_TYPES_TO_TEST_POINTS = {
    numba.types.float64: np.array(
        [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0],
        dtype=np.float64
    ),
    numba.types.float32: np.array(
        [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0],
        dtype=np.float32
    ),
    numba.types.long_: np.array(
        [-100, -10, -1, 0, 1, 10, 100],
        dtype=np.int_
    )
}

SKIP_LIST = {
    # Should be fixed by https://github.com/scipy/scipy/pull/10455
    (
        'hyperu',
        (numba.types.float64,) * 3
    ),
    # Sometimes returns nan, sometimes returns inf. Likely a SciPy bug.
    (
        'eval_jacobi',
        (numba.types.float64,) * 4
    ),
    # Sometimes returns nan, sometimes returns inf. Likely a SciPy bug.
    (
        'eval_sh_jacobi',
        (numba.types.float64,) * 4
    )
}


def compare_functions(args, scipy_func, numba_func):
    for arg in args:
        overload_value = numba_func(*arg)
        scipy_value = scipy_func(*arg)
        if np.isnan(overload_value):
            assert np.isnan(scipy_value)
        else:
            rtol = 2**8 * np.finfo(scipy_value.dtype).eps
            assert_allclose(overload_value, scipy_value, atol=0, rtol=rtol)


def get_parametrize_arguments():
    signatures = special_signatures.name_to_numba_signatures.items()
    for name, specializations in signatures:
        for signature in specializations:
            yield name, signature


def test_parse_capsule_name():
    input_capsule = ('<capsule object "double (double, double, double, '
                     'int __pyx_skip_dispatch)" at 0x7f8c8d5f5150>')
    expected = ['double', 'double', 'double', 'double']
    received = parse_capsule_name(input_capsule)
    assert received == expected


def test_parse_capsule_name_with_invalid_capsule():
    with pytest.raises(ValueError) as excinfo:
        input_capsule = '<TESTING object "double (double)" at 0x7f>'
        parse_capsule_name(input_capsule)
    assert "Unexpected capsule name" in str(excinfo.value)


def test_parse_capsule_name_with_invalid_signature():
    with pytest.raises(ValueError) as excinfo:
        input_capsule = '<capsule object "TESTING" at 0x7f>'
        parse_capsule_name(input_capsule)
    assert "Unexpected signature" in str(excinfo.value)


def test_de_mangle_function_name():
    mangled_name = "__pyx_fuse_0pdtr"
    received = de_mangle_function_name(mangled_name)
    expected = "pdtr"
    assert expected == received


def test_de_mangle_function_name_with_invalid_name():
    with pytest.raises(ValueError) as excinfo:
        # The empty string was the only thing that the regex didn't recognise.
        mangled_name = ""
        print(de_mangle_function_name(mangled_name))
    assert "Unexpected mangled name" in str(excinfo.value)


@patch("numba_scipy.special.signatures.cysc")
def test_get_signatures_from_pyx_capi(cysc_mock):
    capsule = ('<capsule object "double (double, double, double, '
               'int __pyx_skip_dispatch)" at 0x7f8c8d5f5150>')
    mangled_name = "__pyx_fuse_0pdtr"
    cysc_mock.__pyx_capi__ = {
        mangled_name: capsule
    }
    expected = {
        ('__pyx_fuse_0pdtr', float64, float64, float64, float64):
        ('<capsule object "double (double, double, double, '
         'int __pyx_skip_dispatch)" at 0x7f8c8d5f5150>')
    }
    received = get_signatures_from_pyx_capi()
    assert expected == received


@patch("numba_scipy.special.signatures.get_cython_function_address", Mock())
@patch("numba_scipy.special.signatures.ctypes.CFUNCTYPE",
       Mock(return_value=Mock(return_value='0123456789')))
def test_generate_signatures_dicts():
    signature_to_pointer = {
        ('__pyx_fuse_0pdtr', float64, float64, float64, float64):
        ('<capsule object "double (double, double, double, '
         'int __pyx_skip_dispatch)" at 0x7f8c8d5f5150>')
    }
    expected = ({'pdtr': ((float64, float64, float64),)},
                {('pdtr', float64, float64, float64): "0123456789"})
    received = generate_signatures_dicts(signature_to_pointer)
    assert expected == received


def test_ensure_signatures_generated():
    from numba_scipy.special.signatures import (name_to_numba_signatures,
                                                name_and_types_to_pointer,
                                                signature_to_pointer,
                                                )
    assert len(name_to_numba_signatures) != 0
    assert len(name_and_types_to_pointer) != 0
    assert len(signature_to_pointer) != 0
    assert (len(name_and_types_to_pointer) ==
            len(signature_to_pointer))


@pytest.mark.parametrize(
    'name, specialization',
    get_parametrize_arguments(),
)
def test_function(name, specialization):
    if (name, specialization) in SKIP_LIST:
        pytest.xfail()

    scipy_func = getattr(sc, name)

    @numba.njit
    def numba_func(*args):
        return scipy_func(*args)

    args = itertools.product(*(
        NUMBA_TYPES_TO_TEST_POINTS[numba_type] for numba_type in specialization
    ))
    with warnings.catch_warnings():
        # Ignore warnings about unsafe casts generated by SciPy.
        warnings.filterwarnings(
            action='ignore',
            message='floating point number truncated to an integer',
            category=RuntimeWarning,
        )
        compare_functions(args, scipy_func, numba_func)
