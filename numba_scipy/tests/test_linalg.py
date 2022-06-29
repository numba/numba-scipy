import unittest
import numpy as np
from scipy import linalg
from numba import njit

from numpy.testing import assert_allclose, assert_array_almost_equal, assert_almost_equal, assert_equal, \
    assert_array_equal

from numba_scipy.linalg.overloads import qz_impl, ordqz_impl, schur_impl, solve_continuous_lyapunov_impl, \
    solve_discrete_lyapunov_impl
from numba_scipy.linalg.utilities import _iuc, _rhp, _ouc, _lhp


def make_data(n, dtype):
    A = np.random.normal(size=(n, n)).astype(dtype)
    B = np.random.normal(size=(n, n)).astype(dtype)

    if 'complex' in dtype:
        A += (1j * np.random.normal(size=(n, n))).astype(dtype)
        B += (1j * np.random.normal(size=(n, n))).astype(dtype)

    return A.astype(dtype), B.astype(dtype)


class numba_schur_test(unittest.TestCase):
    def setUp(self) -> None:
        @njit
        def numba_schur_test(A, output='real'):
            return linalg.schur(A, output)

        self.schur = numba_schur_test

    def test_numba_schur_float32_small(self):
        n = 5
        A, _ = make_data(n, 'float32')

        t, z = linalg.schur(A)
        T, Z = self.schur(A)

        assert_allclose(t, T)
        assert_allclose(z, Z)

        assert_array_almost_equal(Z @ T @ Z.conj().T, A, decimal=3)

    def test_numba_schur_float32_large(self):
        n = 100
        A, _ = make_data(n, 'float32')

        t, z = linalg.schur(A)
        T, Z = self.schur(A)

        assert_allclose(t, T)
        assert_allclose(z, Z)

        assert_array_almost_equal(Z @ T @ Z.conj().T, A, decimal=3)

    def test_numba_schur_float64_small(self):
        n = 5
        A, _ = make_data(n, 'float64')

        t, z = linalg.schur(A)
        T, Z = self.schur(A)

        assert_allclose(t, T)
        assert_allclose(z, Z)

        assert_array_almost_equal(Z @ T @ Z.conj().T, A)

    def test_numba_schur_float64_large(self):
        n = 100
        A, _ = make_data(n, 'float64')

        t, z = linalg.schur(A)
        T, Z = self.schur(A)

        assert_allclose(t, T)
        assert_allclose(z, Z)

        assert_array_almost_equal(Z @ T @ Z.conj().T, A)

    def test_numba_schur_complex64_small(self):
        n = 5
        A, _ = make_data(n, 'complex64')

        t, z = linalg.schur(A)
        T, Z = self.schur(A, output='complex')

        assert_allclose(t, T)
        assert_allclose(z, Z)

        assert_array_almost_equal(Z @ T @ Z.conj().T, A, decimal=3)

    def test_numba_schur_complex64_large(self):
        n = 100
        A, _ = make_data(n, 'complex64')

        t, z = linalg.schur(A)
        T, Z = self.schur(A, output='complex')

        assert_allclose(t, T)
        assert_allclose(z, Z)

        assert_array_almost_equal(Z @ T @ Z.conj().T, A, decimal=3)

    def test_numba_schur_complex128_small(self):
        n = 5
        A, _ = make_data(n, 'complex128')

        t, z = linalg.schur(A)
        T, Z = self.schur(A, output='complex')

        assert_allclose(t, T)
        assert_allclose(z, Z)

        assert_array_almost_equal(Z @ T @ Z.conj().T, A)

    def test_numba_schur_complex128_large(self):
        n = 100
        A, _ = make_data(n, 'complex128')

        t, z = linalg.schur(A)
        T, Z = self.schur(A, output='complex')

        assert_allclose(t, T)
        assert_allclose(z, Z)

        assert_array_almost_equal(Z @ T @ Z.conj().T, A)


class numba_qz_test(unittest.TestCase):
    def setUp(self) -> None:
        @njit
        def numba_qz_test(A, B, output='real'):
            return linalg.qz(A, B, output)

        self.qz = numba_qz_test

    def test_numba_qz_float32_small(self):
        n = 5
        A, B = make_data(n, 'float32')
        aa, bb, q, z = linalg.qz(A, B)
        AA, BB, Q, Z = self.qz(A, B)

        assert_allclose(aa, AA)
        assert_allclose(bb, BB)
        assert_allclose(q, Q)
        assert_allclose(z, Z)

        assert_array_almost_equal(Q @ AA @ Z.T, A, decimal=5)
        assert_array_almost_equal(Q @ BB @ Z.T, B, decimal=5)
        assert_array_almost_equal(Q @ Q.T, np.eye(n), decimal=5)
        assert_array_almost_equal(Z @ Z.T, np.eye(n), decimal=5)
        assert (np.all(np.diag(BB) >= 0))

    def test_numba_qz_float32_large(self):
        n = 100
        A, B = make_data(n, 'float32')
        aa, bb, q, z = linalg.qz(A, B)
        AA, BB, Q, Z = self.qz(A, B)

        assert_allclose(aa, AA)
        assert_allclose(bb, BB)
        assert_allclose(q, Q)
        assert_allclose(z, Z)

        assert_array_almost_equal(Q @ AA @ Z.T, A, decimal=3)
        assert_array_almost_equal(Q @ BB @ Z.T, B, decimal=3)
        assert_array_almost_equal(Q @ Q.T, np.eye(n), decimal=3)
        assert_array_almost_equal(Z @ Z.T, np.eye(n), decimal=3)
        assert (np.all(np.diag(BB) >= 0))

    def test_numba_qz_float64_small(self):
        n = 5
        A, B = make_data(n, 'float64')
        aa, bb, q, z = linalg.qz(A, B)
        AA, BB, Q, Z = self.qz(A, B)

        assert_allclose(aa, AA)
        assert_allclose(bb, BB)
        assert_allclose(q, Q)
        assert_allclose(z, Z)

        assert_array_almost_equal(Q @ AA @ Z.T, A)
        assert_array_almost_equal(Q @ BB @ Z.T, B)
        assert_array_almost_equal(Q @ Q.T, np.eye(n))
        assert_array_almost_equal(Z @ Z.T, np.eye(n))
        assert (np.all(np.diag(BB) >= 0))

    def test_numba_qz_float64_large(self):
        n = 100
        A, B = make_data(n, 'float64')
        aa, bb, q, z = linalg.qz(A, B)
        AA, BB, Q, Z = self.qz(A, B)

        assert_allclose(aa, AA)
        assert_allclose(bb, BB)
        assert_allclose(q, Q)
        assert_allclose(z, Z)

        assert_array_almost_equal(Q @ AA @ Z.T, A)
        assert_array_almost_equal(Q @ BB @ Z.T, B)
        assert_array_almost_equal(Q @ Q.T, np.eye(n))
        assert_array_almost_equal(Z @ Z.T, np.eye(n))
        assert (np.all(np.diag(BB) >= 0))

    def test_numba_qz_complex64_small(self):
        n = 5
        A, B = make_data(n, 'complex64')
        aa, bb, q, z = linalg.qz(A, B)
        AA, BB, Q, Z = self.qz(A, B)

        assert_allclose(aa, AA)
        assert_allclose(bb, BB)
        assert_allclose(q, Q)
        assert_allclose(z, Z)

        assert_array_almost_equal(Q @ AA @ Z.conj().T, A, decimal=5)
        assert_array_almost_equal(Q @ BB @ Z.conj().T, B, decimal=5)
        assert_array_almost_equal(Q @ Q.conj().T, np.eye(n), decimal=5)
        assert_array_almost_equal(Z @ Z.conj().T, np.eye(n), decimal=5)
        assert (np.all(np.diag(BB) >= 0))

    def test_numba_qz_complex64_large(self):
        n = 100
        A, B = make_data(n, 'complex64')
        aa, bb, q, z = linalg.qz(A, B)
        AA, BB, Q, Z = self.qz(A, B)

        assert_allclose(aa, AA)
        assert_allclose(bb, BB)
        assert_allclose(q, Q)
        assert_allclose(z, Z)

        assert_array_almost_equal(Q @ AA @ Z.conj().T, A, decimal=3)
        assert_array_almost_equal(Q @ BB @ Z.conj().T, B, decimal=3)
        assert_array_almost_equal(Q @ Q.conj().T, np.eye(n), decimal=3)
        assert_array_almost_equal(Z @ Z.conj().T, np.eye(n), decimal=3)
        assert (np.all(np.diag(BB) >= 0))

    def test_numba_qz_complex128_small(self):
        n = 5
        A, B = make_data(n, 'complex128')
        aa, bb, q, z = linalg.qz(A, B)
        AA, BB, Q, Z = self.qz(A, B)

        assert_allclose(aa, AA)
        assert_allclose(bb, BB)
        assert_allclose(q, Q)
        assert_allclose(z, Z)

        assert_array_almost_equal(Q @ AA @ Z.conj().T, A)
        assert_array_almost_equal(Q @ BB @ Z.conj().T, B)
        assert_array_almost_equal(Q @ Q.conj().T, np.eye(n))
        assert_array_almost_equal(Z @ Z.conj().T, np.eye(n))
        assert (np.all(np.diag(BB) >= 0))

    def test_numba_qz_complex128_large(self):
        n = 100
        A, B = make_data(n, 'complex128')
        aa, bb, q, z = linalg.qz(A, B)
        AA, BB, Q, Z = self.qz(A, B)

        assert_allclose(aa, AA)
        assert_allclose(bb, BB)
        assert_allclose(q, Q)
        assert_allclose(z, Z)

        assert_array_almost_equal(Q @ AA @ Z.conj().T, A)
        assert_array_almost_equal(Q @ BB @ Z.conj().T, B)
        assert_array_almost_equal(Q @ Q.conj().T, np.eye(n))
        assert_array_almost_equal(Z @ Z.conj().T, np.eye(n))
        assert (np.all(np.diag(BB) >= 0))


def _select_function(sort):
    if sort == 'lhp':
        return _lhp
    elif sort == 'rhp':
        return _rhp
    elif sort == 'iuc':
        return _iuc
    elif sort == 'ouc':
        return _ouc


class numba_ordqz_test(unittest.TestCase):

    def setUp(self) -> None:
        @njit
        def numba_ordqz_test(A, B, sort='lhp', output='real'):
            return linalg.ordqz(A, B, sort=sort, output=output)

        self.ordqz = numba_ordqz_test

    def test_ordqz_case_1(self):
        A = np.array([[-21.10 - 22.50j, 53.5 - 50.5j, -34.5 + 127.5j,
                       7.5 + 0.5j],
                      [-0.46 - 7.78j, -3.5 - 37.5j, -15.5 + 58.5j,
                       -10.5 - 1.5j],
                      [4.30 - 5.50j, 39.7 - 17.1j, -68.5 + 12.5j,
                       -7.5 - 3.5j],
                      [5.50 + 4.40j, 14.4 + 43.3j, -32.5 - 46.0j,
                       -19.0 - 32.5j]], dtype='complex128')

        B = np.array([[1.0 - 5.0j, 1.6 + 1.2j, -3 + 0j, 0.0 - 1.0j],
                      [0.8 - 0.6j, .0 - 5.0j, -4 + 3j, -2.4 - 3.2j],
                      [1.0 + 0.0j, 2.4 + 1.8j, -4 - 5j, 0.0 - 3.0j],
                      [0.0 + 1.0j, -1.8 + 2.4j, 0 - 4j, 4.0 - 5.0j]], dtype='complex128')

        for sort in ['lhp', 'rhp', 'iuc', 'ouc']:
            numba_ret = self.ordqz(A, B, output='complex', sort=sort)
            scipy_ret = linalg.ordqz(A, B, output='complex', sort=sort)

            for A, a in zip(numba_ret, scipy_ret):
                assert_array_almost_equal(A, a)

    def test_ordqz_case_2(self):
        A = np.array([[3.9, 12.5, -34.5, -0.5],
                      [4.3, 21.5, -47.5, 7.5],
                      [4.3, 21.5, -43.5, 3.5],
                      [4.4, 26.0, -46.0, 6.0]], dtype='float64')

        B = np.array([[1, 2, -3, 1],
                      [1, 3, -5, 4],
                      [1, 3, -4, 3],
                      [1, 3, -4, 4]], dtype='float64')

        for sort in ['lhp', 'rhp', 'iuc', 'ouc']:
            numba_ret = self.ordqz(A, B, output='real', sort=sort)
            scipy_ret = linalg.ordqz(A, B, output='real', sort=sort)

            for A, a in zip(numba_ret, scipy_ret):
                assert_array_almost_equal(A, a)

    def test_ordqz_case_3(self):
        A = np.array([[5., 1., 3., 3.],
                      [4., 4., 2., 7.],
                      [7., 4., 1., 3.],
                      [0., 4., 8., 7.]], dtype='float64')
        B = np.array([[8., 10., 6., 10.],
                      [7., 7., 2., 9.],
                      [9., 1., 6., 6.],
                      [5., 1., 4., 7.]], dtype='float64')

        for sort in ['lhp', 'rhp', 'iuc', 'ouc']:
            numba_ret = self.ordqz(A, B, output='real', sort=sort)
            scipy_ret = linalg.ordqz(A, B, output='real', sort=sort)

            for A, a in zip(numba_ret, scipy_ret):
                assert_array_almost_equal(A, a)

    def test_ordqz_case_4(self):
        A = np.eye(2).astype('float64')
        B = np.diag([0, 1]).astype('float64')

        for sort in ['lhp', 'rhp', 'iuc', 'ouc']:
            numba_ret = self.ordqz(A, B, output='real', sort=sort)
            scipy_ret = linalg.ordqz(A, B, output='real', sort=sort)

            for A, a in zip(numba_ret, scipy_ret):
                assert_array_almost_equal(A, a)

    def test_ordqz_case_5(self):
        A = np.diag([1, 0]).astype('float64')
        B = np.diag([1, 0]).astype('float64')

        for sort in ['lhp', 'rhp', 'iuc', 'ouc']:
            numba_ret = self.ordqz(A, B, output='real', sort=sort)
            scipy_ret = linalg.ordqz(A, B, output='real', sort=sort)

            for A, a in zip(numba_ret, scipy_ret):
                assert_array_almost_equal(A, a)


class numba_solve_lyapunov_tests(unittest.TestCase):
    def setUp(self) -> None:
        @njit
        def numba_solve_continuous_lyapunov(A, B):
            return linalg.solve_continuous_lyapunov(A, B)

        @njit
        def numba_solve_discrete_lyapunov(A, B, method='auto'):
            return linalg.solve_discrete_lyapunov(A, B, method)

        self.solve_continuous_lyapunov = numba_solve_continuous_lyapunov
        self.solve_discrete_lyapunov = numba_solve_discrete_lyapunov

        # Numba is much stricter about typing information than vanilla numpy, so type information needed to be
        # added to all these test.
        self.cases = [
            (np.array([[1, 2], [3, 4]], dtype='float64'),
             np.array([[9, 10], [11, 12]], dtype='float64')),
            # a, q all complex.
            (np.array([[1.0 + 1j, 2.0], [3.0 - 4.0j, 5.0]]),
             np.array([[2.0 - 2j, 2.0 + 2j], [-1.0 - 1j, 2.0]])),
            # a real; q complex.
            (np.array([[1.0, 2.0], [3.0, 5.0]], dtype='complex128'),
             np.array([[2.0 - 2j, 2.0 + 2j], [-1.0 - 1j, 2.0]])),
            # a complex; q real.
            (np.array([[1.0 + 1j, 2.0], [3.0 - 4.0j, 5.0]]),
             np.array([[2.0, 2.0], [-1.0, 2.0]], dtype='complex128')),
            # An example from Kitagawa, 1977
            (np.array([[3, 9, 5, 1, 4], [1, 2, 3, 8, 4], [4, 6, 6, 6, 3],
                       [1, 5, 2, 0, 7], [5, 3, 3, 1, 5]], dtype='float64'),
             np.array([[2, 4, 1, 0, 1], [4, 1, 0, 2, 0], [1, 0, 3, 0, 3],
                       [0, 2, 0, 1, 0], [1, 0, 3, 0, 4]], dtype='float64')),
            # Companion matrix example. a complex; q real; a.shape[0] = 11
            (np.array([[0.100 + 0.j, 0.091 + 0.j, 0.082 + 0.j, 0.073 + 0.j, 0.064 + 0.j,
                        0.055 + 0.j, 0.046 + 0.j, 0.037 + 0.j, 0.028 + 0.j, 0.019 + 0.j,
                        0.010 + 0.j],
                       [1.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j],
                       [0.000 + 0.j, 1.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j],
                       [0.000 + 0.j, 0.000 + 0.j, 1.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j],
                       [0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 1.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j],
                       [0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 1.000 + 0.j,
                        0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j],
                       [0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        1.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j],
                       [0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j, 1.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j],
                       [0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j, 0.000 + 0.j, 1.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j],
                       [0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 1.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j],
                       [0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j,
                        0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 0.000 + 0.j, 1.000 + 0.j,
                        0.000 + 0.j]]),
             np.eye(11).astype('complex128')),
            # https://github.com/scipy/scipy/issues/4176
            (np.matrix([[0, 1], [-1 / 2, -1]], dtype='float64'),
             (np.matrix([0, 3], dtype='float64').T @ np.matrix([0, 3], dtype='float64').T.T)),
            # https://github.com/scipy/scipy/issues/4176
            (np.matrix([[0, 1], [-1 / 2, -1]], dtype='float64'),
             (np.array(np.matrix([0, 3], dtype='float64').T @ np.matrix([0, 3], dtype='float64').T.T)))
        ]

    def test_solve_continuous_lyapunov(self):
        for (A, B) in self.cases:
            X = self.solve_continuous_lyapunov(A, B)
            x = linalg.solve_continuous_lyapunov(A, B)
            assert_array_almost_equal(X, x)

    def test_solve_discrete_lyapunov_auto(self):
        for (A, B) in self.cases:
            X = self.solve_discrete_lyapunov(A, B)
            x = linalg.solve_discrete_lyapunov(A, B)
            assert_array_almost_equal(X, x)

    def test_solve_discrete_lyapunov_direct(self):
        for (A, B) in self.cases:
            X = self.solve_discrete_lyapunov(A, B, method='direct')
            x = linalg.solve_discrete_lyapunov(A, B, method='direct')
            assert_array_almost_equal(X, x)

    def test_solve_discrete_lyapunov_bilinear(self):
        for (A, B) in self.cases:
            X = self.solve_discrete_lyapunov(A, B, method='bilinear')
            x = linalg.solve_discrete_lyapunov(A, B, method='bilinear')
            assert_array_almost_equal(X, x)


if __name__ == '__main__':
    unittest.main()
