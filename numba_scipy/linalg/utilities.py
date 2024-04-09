from numba import jit, njit, types
from numba.core import types, cgutils
from numba.core.errors import TypingError
import numpy as np


def _get_underlying_float(dtype):
    s_dtype = str(dtype)
    out_type = s_dtype
    if s_dtype == 'complex64':
        out_type = 'float32'
    elif s_dtype == 'complex128':
        out_type ='float64'

    return np.dtype(out_type)


def _check_scipy_linalg_matrix(a, func_name):
    prefix = "scipy.linalg"
    interp = (prefix, func_name)
    # Unpack optional type
    if isinstance(a, types.Optional):
        a = a.type
    if not isinstance(a, types.Array):
        msg = "%s.%s() only supported for array types" % interp
        raise TypingError(msg, highlighting=False)
    if not a.ndim == 2:
        msg = "%s.%s() only supported on 2-D arrays." % interp
        raise TypingError(msg, highlighting=False)
    if not isinstance(a.dtype, (types.Float, types.Complex)):
        msg = "%s.%s() only supported on " \
              "float and complex arrays." % interp
        raise TypingError(msg, highlighting=False)


@njit
def direct_lyapunov_solution(A, B):
    lhs = np.kron(A, A.conj())
    lhs = np.eye(lhs.shape[0]) - lhs
    x = np.linalg.solve(lhs, B.flatten())

    return np.reshape(x, B.shape)


@njit
def _lhp(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = (beta != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (np.real(alpha[nonzero]/beta[nonzero]) < 0.0)
    return out

@njit
def _rhp(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = (beta != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (np.real(alpha[nonzero]/beta[nonzero]) > 0.0)
    return out

@njit
def _iuc(alpha, beta):
    out = np.empty(alpha.shape, dtype=np.int32)
    nonzero = (beta != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (np.abs(alpha[nonzero]/beta[nonzero]) < 1.0)

    return out

@njit
def _ouc(alpha, beta):
    """
    Jit-aware version of the function scipy.linalg._decomp_qz._ouc, creates the mask needed for ztgsen to sort
    eigenvalues from stable to unstable.

    Parameters
    ----------
    alpha: Array, complex
        alpha vector, as returned by zgges
    beta: Array, complex
        beta vector, as return by zgges
    Returns
    -------
    out: Array, bool
        Boolean mask indicating which eigenvalues are unstable
    """

    out = np.empty(alpha.shape, dtype=np.int32)
    alpha_zero = (alpha == 0)
    beta_zero = (beta == 0)

    out[alpha_zero & beta_zero] = False
    out[~alpha_zero & beta_zero] = True
    out[~beta_zero] = (np.abs(alpha[~beta_zero] / beta[~beta_zero]) > 1.0)

    return out