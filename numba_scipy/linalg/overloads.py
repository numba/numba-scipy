from numba.core import types, cgutils
from numba.extending import overload
from numba.np.linalg import ensure_lapack, _check_finite_matrix, _copy_to_fortran_order, \
    _handle_err_maybe_convergence_problem

import scipy
import numpy as np
from scipy import linalg

from numba_scipy.linalg.utilities import _check_scipy_linalg_matrix, _get_underlying_float, _ouc, _iuc, _lhp, _rhp, \
    direct_lyapunov_solution
from numba_scipy.linalg.intrinsics import val_to_int_ptr, int_ptr_to_val
from numba_scipy.linalg.LAPACK import _LAPACK


@overload(scipy.linalg.schur)
def schur_impl(A, output):
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "schur")

    dtype = A.dtype
    w_type = _get_underlying_float(dtype)

    numba_rgees = _LAPACK().numba_rgees(dtype)
    numba_cgees = _LAPACK().numba_cgees(dtype)

    def real_schur_impl(A, output):
        """
        schur() implementation for real arrays
        """
        _N = np.int32(A.shape[-1])
        if A.shape[-2] != _N:
            msg = "Last 2 dimensions of the array must be square"
            raise linalg.LinAlgError(msg)

        _check_finite_matrix(A)
        A_copy = _copy_to_fortran_order(A)

        JOBVS = val_to_int_ptr(ord('V'))
        SORT = val_to_int_ptr(ord('N'))
        SELECT = val_to_int_ptr(0.0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(_N)
        WR = np.empty(_N, dtype=dtype)
        WI = np.empty(_N, dtype=dtype)
        _LDVS = _N
        LDVS = val_to_int_ptr(_N)
        VS = np.empty((_LDVS, _N), dtype=dtype)
        LWORK = val_to_int_ptr(-1)
        WORK = np.empty(1, dtype=dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(1)

        # workspace query
        numba_rgees(JOBVS, SORT, SELECT, N, A_copy.ctypes, LDA, SDIM, WR.ctypes, WI.ctypes, VS.ctypes, LDVS,
                    WORK.ctypes, LWORK, BWORK, INFO)
        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual work
        numba_rgees(JOBVS, SORT, SELECT, N, A_copy.ctypes, LDA, SDIM, WR.ctypes, WI.ctypes, VS.ctypes, LDVS,
                    WORK.ctypes, LWORK, BWORK, INFO)

        # if np.any(WI) and output == 'complex':
        #     raise ValueError("schur() argument must not cause a domain change.")
        _handle_err_maybe_convergence_problem(int_ptr_to_val(INFO))

        return A_copy, VS.T

    def complex_schur_impl(A, output):
        """
        schur() implementation for complex arrays
        """

        _N = np.int32(A.shape[-1])
        if A.shape[-2] != _N:
            msg = "Last 2 dimensions of the array must be square"
            raise linalg.LinAlgError(msg)

        _check_finite_matrix(A)
        A_copy = _copy_to_fortran_order(A)

        JOBVS = val_to_int_ptr(ord('V'))
        SORT = val_to_int_ptr(ord('N'))
        SELECT = val_to_int_ptr(0.0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(_N)
        W = np.empty(_N, dtype=dtype)
        _LDVS = _N
        LDVS = val_to_int_ptr(_N)
        VS = np.empty((_LDVS, _N), dtype=dtype)
        LWORK = val_to_int_ptr(-1)
        WORK = np.empty(1, dtype=dtype)
        RWORK = np.empty(_N, dtype=w_type)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(1)

        # workspace query
        numba_cgees(JOBVS, SORT, SELECT, N, A_copy.view(w_type).ctypes, LDA, SDIM, W.view(w_type).ctypes,
                    VS.view(w_type).ctypes, LDVS, WORK.view(w_type).ctypes, LWORK, RWORK.ctypes, BWORK, INFO)

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual work
        numba_cgees(JOBVS, SORT, SELECT, N, A_copy.view(w_type).ctypes, LDA, SDIM, W.view(w_type).ctypes,
                    VS.view(w_type).ctypes, LDVS, WORK.view(w_type).ctypes, LWORK, RWORK.ctypes, BWORK, INFO)

        _handle_err_maybe_convergence_problem(int_ptr_to_val(INFO))

        return A_copy, VS.T

    if isinstance(A.dtype, types.scalars.Complex):
        return complex_schur_impl
    else:
        return real_schur_impl


def full_return_qz(A, B, output):
    pass


@overload(full_return_qz)
def full_return_qz_impl(A, B, output):
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "qz")
    _check_scipy_linalg_matrix(B, "qz")

    dtype = A.dtype
    w_type = _get_underlying_float(dtype)

    numba_rgges = _LAPACK().numba_rgges(dtype)
    numba_cgges = _LAPACK().numba_cgges(dtype)

    def real_full_return_qz_impl(A, B, output):
        """
        schur() implementation for real arrays. Unlike the Scipy function, this has 5 returns, including the
        generalized eigenvalues (alpha, beta), because these are required by ordqz.
        """
        _M, _N = np.int32(A.shape[-2:])
        if A.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of A must be square")
        if B.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of B must be square")

        _check_finite_matrix(A)
        _check_finite_matrix(B)

        A_copy = _copy_to_fortran_order(A)
        B_copy = _copy_to_fortran_order(B)

        JOBVSL = val_to_int_ptr(ord('V'))
        JOBVSR = val_to_int_ptr(ord('V'))
        SORT = val_to_int_ptr(ord('N'))
        SELCTG = val_to_int_ptr(1)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(0)

        ALPHAR = np.empty(_N, dtype=dtype)  # out
        ALPHAI = np.empty(_N, dtype=dtype)  # out
        BETA = np.empty(_N, dtype=dtype)  # out

        _LDVSL = _N
        _LDVSR = _N
        LDVSL = val_to_int_ptr(_LDVSL)
        VSL = np.empty((_LDVSL, _N), dtype=dtype)  # out
        LDVSR = val_to_int_ptr(_LDVSR)
        VSR = np.empty((_LDVSR, _N), dtype=dtype)  # out

        WORK = np.empty((1,), dtype=dtype)  # out
        LWORK = val_to_int_ptr(-1)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(1)

        # workspace query
        numba_rgges(JOBVSL, JOBVSR, SORT, SELCTG, N, A_copy.ctypes, LDA, B_copy.ctypes, LDB,
                    SDIM, ALPHAR.ctypes, ALPHAI.ctypes, BETA.ctypes, VSL.ctypes, LDVSL,
                    VSR.ctypes, LDVSR, WORK.ctypes, LWORK, BWORK, INFO)

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual work
        numba_rgges(JOBVSL, JOBVSR, SORT, SELCTG, N, A_copy.ctypes, LDA, B_copy.ctypes, LDB,
                    SDIM, ALPHAR.ctypes, ALPHAI.ctypes, BETA.ctypes, VSL.ctypes, LDVSL,
                    VSR.ctypes, LDVSR, WORK.ctypes, LWORK, BWORK, INFO)

        _handle_err_maybe_convergence_problem(int_ptr_to_val(INFO))
        ALPHA = ALPHAR + ALPHAI * 1j

        return A_copy, B_copy, ALPHA, BETA, VSL.T, VSR.T

    def complex_full_return_qz_impl(A, B, output):
        """
        qz decomposition for complex arrays. Unlike the Scipy function, this has 5 returns, including the
        generalized eigenvalues (alpha, beta), because these are required by ordqz.
        """

        _M, _N = np.int32(A.shape[-2:])
        if A.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of A must be square")
        if B.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of B must be square")

        _check_finite_matrix(A)
        _check_finite_matrix(B)

        A_copy = _copy_to_fortran_order(A)
        B_copy = _copy_to_fortran_order(B)

        JOBVSL = val_to_int_ptr(ord('V'))
        JOBVSR = val_to_int_ptr(ord('V'))
        SORT = val_to_int_ptr(ord('N'))
        SELCTG = val_to_int_ptr(1)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(0)

        ALPHA = np.empty(_N, dtype=dtype)  # out
        BETA = np.empty(_N, dtype=dtype)  # out
        LDVSL = val_to_int_ptr(_N)
        VSL = np.empty((_N, _N), dtype=dtype)  # out
        LDVSR = val_to_int_ptr(_N)
        VSR = np.empty((_N, _N), dtype=dtype)  # out

        WORK = np.empty((1,), dtype=dtype)  # out
        LWORK = val_to_int_ptr(-1)
        RWORK = np.empty(8 * _N, dtype=w_type)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(1)

        # workspace query
        numba_cgges(JOBVSL, JOBVSR, SORT, SELCTG, N, A_copy.view(w_type).ctypes, LDA, B_copy.view(w_type).ctypes, LDB,
                    SDIM, ALPHA.view(w_type).ctypes, BETA.view(w_type).ctypes, VSL.view(w_type).ctypes,
                    LDVSL, VSR.view(w_type).ctypes, LDVSR, WORK.view(w_type).ctypes, LWORK, RWORK.ctypes, BWORK, INFO)

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual work
        numba_cgges(JOBVSL, JOBVSR, SORT, SELCTG, N, A_copy.view(w_type).ctypes, LDA, B_copy.view(w_type).ctypes, LDB,
                    SDIM, ALPHA.view(w_type).ctypes, BETA.view(w_type).ctypes, VSL.view(w_type).ctypes,
                    LDVSL, VSR.view(w_type).ctypes, LDVSR, WORK.view(w_type).ctypes, LWORK, RWORK.ctypes, BWORK, INFO)

        _handle_err_maybe_convergence_problem(int_ptr_to_val(INFO))

        return A_copy, B_copy, ALPHA, BETA, VSL.T, VSR.T

    if isinstance(A.dtype, types.scalars.Complex):
        return complex_full_return_qz_impl
    else:
        return real_full_return_qz_impl


@overload(scipy.linalg.qz)
def qz_impl(A, B, output):
    """
    scipy.linalg.qz overload. Wraps full_return_qz and returns only A, B, Q ,Z to match the scipy signature.
    """
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "qz")
    _check_scipy_linalg_matrix(B, "qz")

    def real_qz_impl(A, B, output):
        A, B, ALPHA, BETA, VSL, VSR = full_return_qz(A, B, output)

        return A, B, VSL, VSR

    def complex_qz_impl(A, B, output):
        A, B, ALPHA, BETA, VSL, VSR = full_return_qz(A, B, output)
        return A, B, VSL, VSR

    if isinstance(A.dtype, types.scalars.Complex):
        return complex_qz_impl
    else:
        return real_qz_impl


@overload(scipy.linalg.ordqz)
def ordqz_impl(A, B, sort, output):
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "ordqz")
    _check_scipy_linalg_matrix(B, "ordqz")

    dtype = A.dtype
    w_type = _get_underlying_float(dtype)

    numba_rtgsen = _LAPACK().numba_rtgsen(dtype)
    numba_ctgsen = _LAPACK().numba_ctgsen(dtype)

    def real_ordqz_impl(A, B, sort, output):
        _M, _N = np.int32(A.shape[-2:])
        if A.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of A must be square")
        if B.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of B must be square")

        _check_finite_matrix(A)
        _check_finite_matrix(B)

        if sort not in ['lhp', 'rhp', 'iuc', 'ouc']:
            raise ValueError('Argument "sort" should be one of: "lhp", "rhp", "iuc", "ouc"')

        A_copy = _copy_to_fortran_order(A)
        B_copy = _copy_to_fortran_order(B)

        AA, BB, ALPHA, BETA, Q, Z = full_return_qz(A_copy, B_copy, output)

        if sort == 'lhp':
            SELECT = _lhp(ALPHA, BETA)
        elif sort == 'rhp':
            SELECT = _rhp(ALPHA, BETA)
        elif sort == 'iuc':
            SELECT = _iuc(ALPHA, BETA)
        elif sort == 'ouc':
            SELECT = _ouc(ALPHA, BETA)

        IJOB = val_to_int_ptr(0)
        WANTQ = val_to_int_ptr(1)
        WANTZ = val_to_int_ptr(1)
        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_M)
        LDB = val_to_int_ptr(_M)

        ALPHAR = np.empty(_N, dtype=dtype)
        ALPHAI = np.empty(_N, dtype=dtype)

        LDQ = val_to_int_ptr(Q.shape[0])
        LDZ = val_to_int_ptr(Z.shape[0])
        M = val_to_int_ptr(_M)
        PL = np.empty(1, dtype=dtype)
        PR = np.empty(1, dtype=dtype)
        DIF = np.empty(2, dtype=dtype)
        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)
        IWORK = np.empty(1, dtype=np.int32)
        LIWORK = val_to_int_ptr(-1)
        INFO = val_to_int_ptr(1)

        # workspace query
        numba_rtgsen(IJOB, WANTQ, WANTZ, SELECT.ctypes, N, AA.ctypes, LDA, BB.ctypes, LDB, ALPHAR.ctypes,
                     ALPHAI.ctypes, BETA.ctypes, Q.ctypes, LDQ, Z.ctypes, LDZ, M, PL.ctypes,
                     PR.ctypes, DIF.ctypes, WORK.ctypes, LWORK, IWORK.ctypes, LIWORK, INFO)

        WS_SIZE = np.int32(WORK[0].real)
        IW_SIZE = np.int32(IWORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        LIWORK = val_to_int_ptr(IW_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)
        IWORK = np.empty(IW_SIZE, dtype=np.int32)

        numba_rtgsen(IJOB, WANTQ, WANTZ, SELECT.ctypes, N, AA.ctypes, LDA, BB.ctypes, LDB, ALPHAR.ctypes,
                     ALPHAI.ctypes, BETA.ctypes, Q.ctypes, LDQ, Z.ctypes, LDZ, M, PL.ctypes,
                     PR.ctypes, DIF.ctypes, WORK.ctypes, LWORK, IWORK.ctypes, LIWORK, INFO)

        # if np.any(ALPHAI) and output == 'complex':
        #     raise ValueError("ordqz() argument must not cause a domain change.")
        _handle_err_maybe_convergence_problem(int_ptr_to_val(INFO))
        ALPHA = ALPHAR + 1j * ALPHAI
        return AA, BB, ALPHA, BETA, Q, Z

    def complex_ordqz_impl(A, B, sort, output):
        _M, _N = np.int32(A.shape[-2:])
        if A.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of A must be square")
        if B.shape[-2] != _N:
            raise linalg.LinAlgError("Last 2 dimensions of B must be square")

        _check_finite_matrix(A)
        _check_finite_matrix(B)

        if sort not in ['lhp', 'rhp', 'iuc', 'ouc']:
            raise ValueError('Argument "sort" should be one of: "lhp", "rhp", "iuc", "ouc"')

        A_copy = _copy_to_fortran_order(A)
        B_copy = _copy_to_fortran_order(B)

        AA, BB, ALPHA, BETA, Q, Z = full_return_qz(A_copy, B_copy, output)

        if sort == 'lhp':
            SELECT = _lhp(ALPHA, BETA)
        elif sort == 'rhp':
            SELECT = _rhp(ALPHA, BETA)
        elif sort == 'iuc':
            SELECT = _iuc(ALPHA, BETA)
        elif sort == 'ouc':
            SELECT = _ouc(ALPHA, BETA)

        IJOB = val_to_int_ptr(0)
        WANTQ = val_to_int_ptr(1)
        WANTZ = val_to_int_ptr(1)
        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_M)
        LDB = val_to_int_ptr(_M)

        LDQ = val_to_int_ptr(Q.shape[0])
        LDZ = val_to_int_ptr(Z.shape[0])
        M = val_to_int_ptr(_M)
        PL = np.empty(1, dtype=w_type)
        PR = np.empty(1, dtype=w_type)
        DIF = np.empty(2, dtype=w_type)
        WORK = np.empty(1, dtype=dtype)
        LWORK = val_to_int_ptr(-1)
        IWORK = np.empty(1, dtype=np.int32)
        LIWORK = val_to_int_ptr(-1)
        INFO = val_to_int_ptr(1)

        # workspace query
        numba_ctgsen(IJOB, WANTQ, WANTZ, SELECT.ctypes, N, AA.view(w_type).ctypes, LDA, BB.view(w_type).ctypes, LDB,
                     ALPHA.view(w_type).ctypes, BETA.view(w_type).ctypes, Q.view(w_type).ctypes, LDQ,
                     Z.view(w_type).ctypes, LDZ, M, PL.ctypes, PR.ctypes, DIF.ctypes,
                     WORK.view(w_type).ctypes, LWORK, IWORK.ctypes, LIWORK, INFO)

        WS_SIZE = np.int32(WORK[0].real)
        IW_SIZE = np.int32(IWORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        LIWORK = val_to_int_ptr(IW_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)
        IWORK = np.empty(IW_SIZE, dtype=np.int32)

        numba_ctgsen(IJOB, WANTQ, WANTZ, SELECT.ctypes, N, AA.view(w_type).ctypes, LDA, BB.view(w_type).ctypes,
                     LDB, ALPHA.view(w_type).ctypes, BETA.view(w_type).ctypes, Q.view(w_type).ctypes, LDQ,
                     Z.view(w_type).ctypes, LDZ, M, PL.ctypes, PR.ctypes, DIF.ctypes, WORK.view(w_type).ctypes,
                     LWORK, IWORK.ctypes, LIWORK, INFO)

        _handle_err_maybe_convergence_problem(int_ptr_to_val(INFO))

        return AA, BB, ALPHA, BETA, Q, Z

    if isinstance(A.dtype, types.scalars.Complex):
        return complex_ordqz_impl
    else:
        return real_ordqz_impl


@overload(scipy.linalg.solve_continuous_lyapunov)
def solve_continuous_lyapunov_impl(A, Q):
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "solve_continuous_lyapunov")
    _check_scipy_linalg_matrix(Q, "solve_continuous_lyapunov")

    dtype = A.dtype
    w_type = _get_underlying_float(dtype)

    numba_xtrsyl = _LAPACK().numba_xtrsyl(dtype)

    def _solve_cont_lyapunov_impl(A, Q):
        _M, _N = np.int32(A.shape)
        _NQ = np.int32(Q.shape[-1])

        if _N != _NQ:
            raise linalg.LinAlgError('Matrices A and Q must have the same shape')

        if _M != _N:
            raise linalg.LinAlgError("Last 2 dimensions of A must be square")
        if Q.shape[-2] != _NQ:
            raise linalg.LinAlgError("Last 2 dimensions of Q must be square")

        _check_finite_matrix(A)
        _check_finite_matrix(Q)

        is_complex = (np.iscomplexobj(A) | np.iscomplexobj(Q))
        dtype_letter = 'C' if is_complex else 'T'
        output = 'complex' if is_complex else 'real'

        A_copy = _copy_to_fortran_order(A)
        Q_copy = _copy_to_fortran_order(Q)

        R, U = linalg.schur(A_copy, output=output)

        # Construct f = u'*q*u
        F = U.conj().T.dot(Q_copy.dot(U))

        TRANA = val_to_int_ptr(ord('N'))
        TRANB = val_to_int_ptr(ord(dtype_letter))
        ISGN = val_to_int_ptr(1)

        M = val_to_int_ptr(_N)
        N = val_to_int_ptr(_N)
        AA = _copy_to_fortran_order(R)
        LDA = val_to_int_ptr(_N)
        B = _copy_to_fortran_order(R)
        LDB = val_to_int_ptr(_N)
        C = _copy_to_fortran_order(F)
        LDC = val_to_int_ptr(_N)

        # TODO: There is a little bit of overhead here, can I figure out how to assign a
        #  float or double pointer, depending on the case?
        SCALE = np.array(1.0, dtype=w_type)
        INFO = val_to_int_ptr(1)

        numba_xtrsyl(TRANA, TRANB, ISGN, M, N,
                     AA.view(w_type).ctypes, LDA,
                     B.view(w_type).ctypes, LDB,
                     C.view(w_type).ctypes, LDC,
                     SCALE.ctypes, INFO)

        C *= SCALE
        _handle_err_maybe_convergence_problem(int_ptr_to_val(INFO))
        X = U.dot(C).dot(U.conj().T)

        return X

    return _solve_cont_lyapunov_impl


@overload(scipy.linalg.solve_discrete_lyapunov)
def solve_discrete_lyapunov_impl(A, Q, method='auto'):
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "solve_continuous_lyapunov")
    _check_scipy_linalg_matrix(Q, "solve_continuous_lyapunov")

    dtype = A.dtype
    w_type = _get_underlying_float(dtype)

    def impl(A, Q, method='auto'):
        _M, _N = np.int32(A.shape)

        if method == 'auto':
            if _M < 10:
                method = 'direct'
            else:
                method = 'bilinear'

        if method == 'direct':
            X = direct_lyapunov_solution(A, Q)

        if method == 'bilinear':
            eye = np.eye(_M)
            AH = A.conj().transpose()
            AHI_inv = np.linalg.inv(AH + eye)
            B = np.dot(AH - eye, AHI_inv)
            C = 2 * np.dot(np.dot(np.linalg.inv(A + eye), Q), AHI_inv)
            X = linalg.solve_continuous_lyapunov(B.conj().transpose(), -C)

        return X

    return impl
