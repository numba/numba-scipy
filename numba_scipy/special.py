from numba.extending import get_cython_function_address
from numba import vectorize, njit
from numba.extending import overload
import ctypes
import numpy as np
from scipy import special

_addr = get_cython_function_address("scipy.special.cython_special", "beta")
_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
_beta_fn = _functype(_addr)

@vectorize('float64(float64, float64)')
def _vec_beta(x, y):
    return _beta_fn(x, y)

@overload(special.beta)
def overload_beta(x, y):
    def impl(x, y):
        return _vec_beta(x, y)
    return impl
