import numpy as np
from pytest import raises
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
    str_one = mat_structs._get_random_string()
    assert isinstance(str_one, str)
    for i in range(1000):
        alt_str = mat_structs._get_random_string()
        assert str_one != alt_str


def test_LowRankMatrix_init():
    with raises(ValueError):
        x = np.eye(5)
        x[0, 1] = 2
        mat_structs.LowRankMatrix(X=x)

    with raises(ValueError):
        x = np.eye(5)
        u = 3
        mat_structs.LowRankMatrix(X=x, u=u)

    with raises(ValueError):
        mat_structs.LowRankMatrix()

    x = np.random.random((5, 5))
    x = x + x.T + 3 * np.eye(5)

    m = mat_structs.LowRankMatrix(X=x, t=1.)
    assert np.allclose(
        x, np.einsum('ik,k,kj->ij', m.u, m.s, m.v)
    )
    assert m.shape == (5, 5)
    assert np.allclose(m.inv_s, 1./m.s)
    assert np.allclose(m.D, 0)

    u, s, v = np.linalg.svd(x)
    D = np.zeros((5, 5))
    m = mat_structs.LowRankMatrix(u=u, s=s, v=v, D=D)
    assert np.allclose(u, m.u)
    assert np.allclose(v, m.v)
    assert np.allclose(s, m.s)
    assert np.allclose(D, m.D)
    assert np.allclose(1./s, m.inv_s)
    assert m.shape == (5, 5)
    assert np.allclose(
        x, np.einsum('ik,k,kj->ij', m.u, m.s, m.v)
    )

    x = np.eye(5)
    x[0, 0] = 0
    m = mat_structs.LowRankMatrix(X=x)
    assert m.shape == (5, 5)
    assert m.s.shape[0] == 4
    assert m.u.shape == (5, 4)
    assert m.v.shape == (4, 5)


def test_LowRankMatrix_dot():
    x = np.random.random((5, 5))
    x = x + x.T + 3 * np.eye(5)
    m = mat_structs.LowRankMatrix(X=x, t=1.)
    v = np.random.random(5)
    assert np.allclose(x.dot(v), m.dot(v))

    D = np.random.random(5)
    m = mat_structs.LowRankMatrix(u=m.u, s=m.s, v=m.v, D=D)
    assert np.allclose((x+np.diag(D)).dot(v), m.dot(v))


def test_LowRankMatrix_dot_i():
    x = np.random.random((5, 5))
    x = x + x.T + 3 * np.eye(5)
    m = mat_structs.LowRankMatrix(X=x, t=1.)
    v = np.random.random(5)
    for i in range(5):
        assert np.isclose(x.dot(v)[i], m.dot_i(v, i))

    D = np.random.random(5)
    m = mat_structs.LowRankMatrix(u=m.u, s=m.s, v=m.v, D=D)
    for i in range(5):
        assert np.isclose(
            (x + np.diag(D)).dot(v)[i],
            m.dot_i(v, i)
        )


def test_LowRankMatrix_inverse_dot():
    x = np.random.random((5, 5))
    x = x + x.T + 3 * np.eye(5)
    m = mat_structs.LowRankMatrix(X=x, t=1.)
    v = np.random.random(5)
    assert np.allclose(
        np.linalg.inv(x).dot(v), m.inverse_dot(v)
    )

    D = np.random.random(5)
    m = mat_structs.LowRankMatrix(u=m.u, s=m.s, v=m.v, D=D)
    assert np.allclose(
        np.linalg.inv(x + np.diag(D)).dot(v),
        m.inverse_dot(v)
    )

    x = np.eye(5)
    x[0, 0] = 0
    m = mat_structs.LowRankMatrix(X=x, t=0.5)
    v_trunc = np.copy(v)
    v_trunc[0] = 0
    assert np.allclose(
        v_trunc, m.inverse_dot(v)
    )


def test_LowRankMatrix_diag():
    x = np.random.random((5, 5))
    x = x + x.T + 3 * np.eye(5)
    m = mat_structs.LowRankMatrix(X=x, t=1.)
    assert np.allclose(np.diag(x), m.diag())

    D = np.random.random(5)
    m = mat_structs.LowRankMatrix(u=m.u, s=m.s, v=m.v, D=D)
    assert np.allclose(np.diag(x) + D, m.diag())


def test_LowRankMatrix_pow():
    x = np.random.random((5, 5))
    x = x + x.T + 3 * np.eye(5)
    m = mat_structs.LowRankMatrix(X=x, t=1.)

    v = np.random.random(5)

    assert np.allclose(x.dot(x).dot(v), (m**2).dot(v))

    x = np.eye(5)*2
    x[0, 0] = 0
    m = mat_structs.LowRankMatrix(X=x, t=0.5)

    assert np.allclose(np.diag(x)**2, (m**2).diag())

    with raises(NotImplementedError):
        D = np.random.random(5)
        m = mat_structs.LowRankMatrix(u=m.u, s=m.s, v=m.v, D=D)
        m**2


def test_LowRankMatrix_get_rank():
    x = np.random.random((5, 5))
    x = x + x.T + 3 * np.eye(5)
    m = mat_structs.LowRankMatrix(X=x, t=1.)

    assert m.get_rank() == 5

    x /= 1000
    curr_rank = 5
    for t in np.linspace(1, 0):
        m = mat_structs.LowRankMatrix(X=x, t=t)
        new_rank = m.get_rank()
        assert new_rank <= curr_rank
        curr_rank = new_rank
    assert new_rank == 0

    x = np.eye(5)
    x[0, 0] = 0
    m = mat_structs.LowRankMatrix(X=x, t=0.5)

    assert m.get_rank() == 4


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
