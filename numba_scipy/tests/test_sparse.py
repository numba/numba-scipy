import pytest
import numba
import numpy as np
import scipy.sparse

from numba_scipy.sparse import NumbaExperimentalFeatureWarning


def test_sparse_unboxing():
    @numba.njit
    def test_unboxing(x, y):
        return x.shape, y.shape

    x_val = scipy.sparse.csr_matrix(np.eye(100))
    y_val = scipy.sparse.csc_matrix(np.eye(101))

    with pytest.warns(NumbaExperimentalFeatureWarning):
        res = test_unboxing(x_val, y_val)

    assert res == (x_val.shape, y_val.shape)


def test_sparse_boxing():
    @numba.njit
    def test_boxing(x, y):
        return x, y

    x_val = scipy.sparse.csr_matrix(np.eye(100))
    y_val = scipy.sparse.csc_matrix(np.eye(101))

    with pytest.warns(NumbaExperimentalFeatureWarning):
        res_x_val, res_y_val = test_boxing(x_val, y_val)

    assert np.array_equal(res_x_val.data, x_val.data)
    assert np.array_equal(res_x_val.indices, x_val.indices)
    assert np.array_equal(res_x_val.indptr, x_val.indptr)
    assert res_x_val.shape == x_val.shape

    assert np.array_equal(res_y_val.data, y_val.data)
    assert np.array_equal(res_y_val.indices, y_val.indices)
    assert np.array_equal(res_y_val.indptr, y_val.indptr)
    assert res_y_val.shape == y_val.shape


def test_sparse_shape():
    @numba.njit
    def test_fn(x):
        return np.shape(x)

    x_val = scipy.sparse.csr_matrix(np.eye(100))

    with pytest.warns(NumbaExperimentalFeatureWarning):
        res = test_fn(x_val)

    assert res == (100, 100)


def test_sparse_ndim():
    @numba.njit
    def test_fn(x):
        return x.ndim

    x_val = scipy.sparse.csr_matrix(np.eye(100))

    with pytest.warns(NumbaExperimentalFeatureWarning):
        res = test_fn(x_val)

    assert res == 2


def test_sparse_copy():
    @numba.njit
    def test_fn(x):
        y = x.copy()
        return (
            y is not x and np.all(x.data == y.data) and np.all(x.indices == y.indices)
        )

    x_val = scipy.sparse.csr_matrix(np.eye(100))

    with pytest.warns(NumbaExperimentalFeatureWarning):
        assert test_fn(x_val)
