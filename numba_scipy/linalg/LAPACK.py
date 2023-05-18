from numba.extending import get_cython_function_address
from numba.np.linalg import ensure_lapack, _blas_kinds
import ctypes

_PTR = ctypes.POINTER

_dbl = ctypes.c_double
_float = ctypes.c_float
_char = ctypes.c_char
_int = ctypes.c_int

_ptr_float = _PTR(_float)
_ptr_dbl = _PTR(_dbl)
_ptr_char = _PTR(_char)
_ptr_int = _PTR(_int)


def _get_float_pointer_for_dtype(blas_dtype):
    if blas_dtype in ['s', 'c']:
        return _ptr_float
    elif blas_dtype in ['d', 'z']:
        return _ptr_dbl


class _LAPACK:
    """
    Functions to return type signatures for wrapped
    LAPACK functions.
    """

    def __init__(self):
        ensure_lapack()

    @classmethod
    def test_blas_kinds(cls, dtype):
        return _blas_kinds[dtype]

    @classmethod
    def numba_rgees(cls, dtype):
        d = _blas_kinds[dtype]
        func_name = f'{d}gees'
        float_pointer = _get_float_pointer_for_dtype(d)
        addr = get_cython_function_address('scipy.linalg.cython_lapack', func_name)
        functype = ctypes.CFUNCTYPE(None,
                                    _ptr_int,  # JOBVS
                                    _ptr_int,  # SORT
                                    _ptr_int,  # SELECT
                                    _ptr_int,  # N
                                    float_pointer,  # A
                                    _ptr_int,  # LDA
                                    _ptr_int,  # SDIM
                                    float_pointer,  # WR
                                    float_pointer,  # WI
                                    float_pointer,  # VS
                                    _ptr_int,  # LDVS
                                    float_pointer,  # WORK
                                    _ptr_int,  # LWORK
                                    _ptr_int,  # BWORK
                                    _ptr_int)  # INFO
        return functype(addr)

    @classmethod
    def numba_cgees(cls, dtype):
        d = _blas_kinds[dtype]
        func_name = f'{d}gees'
        float_pointer = _get_float_pointer_for_dtype(d)
        addr = get_cython_function_address('scipy.linalg.cython_lapack', func_name)
        functype = ctypes.CFUNCTYPE(None,
                                    _ptr_int,  # JOBVS
                                    _ptr_int,  # SORT
                                    _ptr_int,  # SELECT
                                    _ptr_int,  # N
                                    float_pointer,  # A
                                    _ptr_int,  # LDA
                                    _ptr_int,  # SDIM
                                    float_pointer,  # W
                                    float_pointer,  # VS
                                    _ptr_int,  # LDVS
                                    float_pointer,  # WORK
                                    _ptr_int,  # LWORK
                                    float_pointer,  # RWORK
                                    _ptr_int,  # BWORK
                                    _ptr_int)  # INFO
        return functype(addr)

    @classmethod
    def numba_rgges(cls, dtype):
        d = _blas_kinds[dtype]
        func_name = f'{d}gges'
        float_pointer = _get_float_pointer_for_dtype(d)
        addr = get_cython_function_address('scipy.linalg.cython_lapack', func_name)

        functype = ctypes.CFUNCTYPE(None,
                                    _ptr_int,  # JOBVSL
                                    _ptr_int,  # JOBVSR
                                    _ptr_int,  # SORT
                                    _ptr_int,  # SELCTG
                                    _ptr_int,  # N
                                    float_pointer,  # A
                                    _ptr_int,  # LDA
                                    float_pointer,  # B
                                    _ptr_int,  # LDB
                                    _ptr_int,  # SDIM
                                    float_pointer,  # ALPHAR
                                    float_pointer,  # ALPHAI
                                    float_pointer,  # BETA
                                    float_pointer,  # VSL
                                    _ptr_int,  # LDVSL
                                    float_pointer,  # VSR
                                    _ptr_int,  # LDVSR
                                    float_pointer,  # WORK
                                    _ptr_int,  # LWORK
                                    _ptr_int,  # BWORK
                                    _ptr_int)  # INFO
        return functype(addr)

    @classmethod
    def numba_cgges(cls, dtype):
        d = _blas_kinds[dtype]
        func_name = f'{d}gges'
        float_pointer = _get_float_pointer_for_dtype(d)
        addr = get_cython_function_address('scipy.linalg.cython_lapack', func_name)

        functype = ctypes.CFUNCTYPE(None,
                                    _ptr_int,  # JOBVSL
                                    _ptr_int,  # JOBVSR
                                    _ptr_int,  # SORT
                                    _ptr_int,  # SELCTG
                                    _ptr_int,  # N
                                    float_pointer,  # A, complex
                                    _ptr_int,  # LDA
                                    float_pointer,  # B, complex
                                    _ptr_int,  # LDB
                                    _ptr_int,  # SDIM
                                    float_pointer,  # ALPHA, complex
                                    float_pointer,  # BETA, complex
                                    float_pointer,  # VSL, complex
                                    _ptr_int,  # LDVSL
                                    float_pointer,  # VSR, complex
                                    _ptr_int,  # LDVSR
                                    float_pointer,  # WORK, complex
                                    _ptr_int,  # LWORK
                                    float_pointer,  # RWORK
                                    _ptr_int,  # BWORK
                                    _ptr_int)  # INFO
        return functype(addr)

    @classmethod
    def numba_rtgsen(cls, dtype):
        d = _blas_kinds[dtype]
        func_name = f'{d}tgsen'
        float_pointer = _get_float_pointer_for_dtype(d)
        addr = get_cython_function_address('scipy.linalg.cython_lapack', func_name)

        functype = ctypes.CFUNCTYPE(None,
                                    _ptr_int,  # IJOB
                                    _ptr_int,  # WANTQ
                                    _ptr_int,  # WANTZ
                                    _ptr_int,  # SELECT
                                    _ptr_int,  # N
                                    float_pointer,  # A
                                    _ptr_int,  # LDA
                                    float_pointer,  # B
                                    _ptr_int,  # LDB
                                    float_pointer,  # ALPHAR
                                    float_pointer,  # ALPHAI
                                    float_pointer,  # BETA
                                    float_pointer,  # Q
                                    _ptr_int,  # LDQ
                                    float_pointer,  # Z
                                    _ptr_int,  # LDZ
                                    _ptr_int,  # M
                                    float_pointer,  # PL
                                    float_pointer,  # PR
                                    float_pointer,  # DIF
                                    float_pointer,  # WORK
                                    _ptr_int,  # LWORK
                                    _ptr_int,  # IWORK
                                    _ptr_int,  # LIWORK
                                    _ptr_int)  # INFO
        return functype(addr)

    @classmethod
    def numba_ctgsen(cls, dtype):
        d = _blas_kinds[dtype]
        func_name = f'{d}tgsen'
        float_pointer = _get_float_pointer_for_dtype(d)
        addr = get_cython_function_address('scipy.linalg.cython_lapack', func_name)

        functype = ctypes.CFUNCTYPE(None,
                                    _ptr_int,  # IJOB
                                    _ptr_int,  # WANTQ
                                    _ptr_int,  # WANTZ
                                    _ptr_int,  # SELECT
                                    _ptr_int,  # N
                                    float_pointer,  # A
                                    _ptr_int,  # LDA
                                    float_pointer,  # B
                                    _ptr_int,  # LDB
                                    float_pointer,  # ALPHA
                                    float_pointer,  # BETA
                                    float_pointer,  # Q
                                    _ptr_int,  # LDQ
                                    float_pointer,  # Z
                                    _ptr_int,  # LDZ
                                    _ptr_int,  # M
                                    float_pointer,  # PL
                                    float_pointer,  # PR
                                    float_pointer,  # DIF
                                    float_pointer,  # WORK
                                    _ptr_int,  # LWORK
                                    _ptr_int,  # IWORK
                                    _ptr_int,  # LIWORK
                                    _ptr_int)  # INFO
        return functype(addr)

    @classmethod
    def numba_xtrsyl(cls, dtype):
        d = _blas_kinds[dtype]
        func_name = f'{d}trsyl'
        float_pointer = _get_float_pointer_for_dtype(d)
        addr = get_cython_function_address('scipy.linalg.cython_lapack', func_name)

        functype = ctypes.CFUNCTYPE(None,
                                    _ptr_int,  # TRANA
                                    _ptr_int,  # TRANB
                                    _ptr_int,  # ISGN
                                    _ptr_int,  # M
                                    _ptr_int,  # N
                                    float_pointer,  # A
                                    _ptr_int,  # LDA
                                    float_pointer,  # B
                                    _ptr_int,  # LDB
                                    float_pointer,  # C
                                    _ptr_int,  # LDC
                                    float_pointer,  # SCALE
                                    _ptr_int)  # INFO
        return functype(addr)
