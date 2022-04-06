import numpy as np
from vilma import matrix_structures as mat_structs


###############################################################################
###############################################################################
###############################################################################
# test matrix_structures.py

def test_svd_threshold():
    x = np.random.random((5, 5))
    x = x + x.T + 3 * np.eye(5)
    u, s, v = mat_structs._svd_threshold(x, 1)
    assert np.allclose(x, np.einsum('ik,k,kj->ij', u, s, v))

    for t in np.linspace(0, 1):
        u, s, v = mat_structs._svd_threshold(x, t)
        assert np.all(s > 1 - np.sqrt(t))

    x = np.eye(5)
    x[0, 0] = 0
    u, s, v = mat_structs._svd_threshold(x, 0.5)
    assert s.shape[0] == 4
    assert np.allclose(x, np.einsum('ik,k,kj->ij', u, s, v))


def test_get_random_string():
    pass


def test_LowRankMatrix_init():
    pass


def test_LowRankMatrix_dot():
    pass


def test_LowRankMatrix_dot_i():
    pass


def test_LowRankMatrix_inverse_dot():
    pass


def test_LowRankMatrix_diag():
    pass


def test_LowRankMatrix_pow():
    pass


def test_LowRankMatrix_get_rank():
    pass


def test_BlockDiagonalMatrix_init():
    pass


def test_BlockDiagonalMatrix_dot_i():
    pass


def test_BlockDiagonalMatrix_ridge_inverse_dot():
    pass


def test_BlockDiagonalMatrix_dot():
    pass


def test_BlockDiagonalMatrix_pow():
    pass


def test_BlockDiagonalMatrix_inverse():
    pass


def test_BlockDiagonalMatrix_diag():
    pass


def test_BlockDiagonalMatrix_rank():
    pass
