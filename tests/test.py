import numpy as np
from pytest import raises
from vilma import matrix_structures as mat_structs
from vilma import load


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
    x1 = np.random.random((5, 5))
    x1 = x1 + x1.T + 3 * np.eye(5)
    m1 = mat_structs.LowRankMatrix(X=x1, t=1.)
    x2 = np.random.random((3, 3))
    x2 = x2 + x2.T + 3 * np.eye(3)
    m2 = mat_structs.LowRankMatrix(X=x2, t=1.)
    full_x = np.zeros((8, 8))
    full_x[0:5, 0:5] = x1
    full_x[5:8, 5:8] = x2
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2])

    assert np.allclose(m.perm, np.arange(8))
    assert np.allclose(m.inv_perm, np.arange(8))
    assert m.shape == (8, 8)
    assert m.missing.shape[0] == 0
    assert len(m.matrices) == 2
    assert m.matrices[0].shape == (5, 5)
    assert m.matrices[1].shape == (3, 3)
    assert np.allclose(m.starts, np.array([0., 5., 8.]))

    my_perm = np.random.permutation(np.arange(8))
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        perm=my_perm)
    assert np.allclose(m.perm, my_perm)
    v = np.random.random(8)
    assert np.allclose(v[m.perm][m.inv_perm], v)

    with raises(ValueError):
        mat_structs.BlockDiagonalMatrix(
            [m1, np.eye(3)]
        )

    with raises(ValueError):
        mat_structs.BlockDiagonalMatrix(
            [m1, m2], perm=np.arange(100)
        )

    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        missing=np.arange(8, 100))
    assert len(m.matrices) == 2
    assert m.shape == (100, 100)
    assert m.missing.shape[0] == 92


def test_BlockDiagonalMatrix_dot_i():
    x1 = np.random.random((5, 5))
    x1 = x1 + x1.T + 3 * np.eye(5)
    m1 = mat_structs.LowRankMatrix(X=x1, t=1.)
    x2 = np.random.random((3, 3))
    x2 = x2 + x2.T + 3 * np.eye(3)
    m2 = mat_structs.LowRankMatrix(X=x2, t=1.)
    full_x = np.zeros((8, 8))
    full_x[0:5, 0:5] = x1
    full_x[5:8, 5:8] = x2
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2])
    v = np.random.random(8)
    for i in range(8):
        assert np.isclose(full_x.dot(v)[i], m.dot_i(v, i))

    my_perm = np.random.permutation(np.arange(8))
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        perm=my_perm)
    res = full_x.dot(v[my_perm])[np.argsort(my_perm)]
    for i in range(8):
        assert np.isclose(res[i], m.dot_i(v, i))


def test_BlockDiagonalMatrix_ridge_inverse_dot():
    x1 = np.random.random((5, 5))
    x1 = x1 + x1.T + 3 * np.eye(5)
    m1 = mat_structs.LowRankMatrix(X=x1, t=1.)
    x2 = np.random.random((3, 3))
    x2 = x2 + x2.T + 3 * np.eye(3)
    m2 = mat_structs.LowRankMatrix(X=x2, t=1.)
    full_x = np.zeros((8, 8))
    full_x[0:5, 0:5] = x1
    full_x[5:8, 5:8] = x2
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2])

    v = np.random.random(8)
    r = np.random.random() + 0.1
    assert np.allclose(
        np.linalg.inv(full_x + np.eye(8)*r).dot(v),
        m.ridge_inverse_dot(v, r)
    )

    r = np.random.random(8)
    assert np.allclose(
        np.linalg.inv(full_x + np.diag(r)).dot(v),
        m.ridge_inverse_dot(v, r)
    )

    my_perm = np.random.permutation(np.arange(8))
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        perm=my_perm)
    assert np.allclose(
        (np.linalg.inv(
            full_x + np.diag(r[my_perm])
        ).dot(v[my_perm]))[np.argsort(my_perm)],
        m.ridge_inverse_dot(v, r)
    )

    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        missing=np.arange(8, 100))

    v_full = np.zeros(100)
    v_full[0:8] = v
    r_full = np.random.random(100)
    r_full[0:8] = r
    assert np.allclose(
        np.linalg.inv(full_x + np.diag(r)).dot(v),
        m.ridge_inverse_dot(v_full, r_full)[0:8]
    )
    assert np.allclose(
        m.ridge_inverse_dot(v_full, r_full)[8:], 0.
    )


def test_BlockDiagonalMatrix_dot():
    x1 = np.random.random((5, 5))
    x1 = x1 + x1.T + 3 * np.eye(5)
    m1 = mat_structs.LowRankMatrix(X=x1, t=1.)
    x2 = np.random.random((3, 3))
    x2 = x2 + x2.T + 3 * np.eye(3)
    m2 = mat_structs.LowRankMatrix(X=x2, t=1.)
    full_x = np.zeros((8, 8))
    full_x[0:5, 0:5] = x1
    full_x[5:8, 5:8] = x2
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2])

    v = np.random.random(8)
    assert np.allclose(
        full_x.dot(v), m.dot(v)
    )

    my_perm = np.random.permutation(np.arange(8))
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        perm=my_perm)
    assert np.allclose(
        full_x.dot(v[my_perm])[np.argsort(my_perm)],
        m.dot(v)
    )

    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        missing=np.arange(8, 100))
    v_full = np.zeros(100)
    v_full[0:8] = v
    assert np.allclose(
        full_x.dot(v),
        m.dot(v_full)[0:8]
    )
    assert np.allclose(m.dot(v_full)[8:], 0)


def test_BlockDiagonalMatrix_pow():
    x1 = np.random.random((5, 5))
    x1 = x1 + x1.T + 3 * np.eye(5)
    m1 = mat_structs.LowRankMatrix(X=x1, t=1.)
    x2 = np.random.random((3, 3))
    x2 = x2 + x2.T + 3 * np.eye(3)
    m2 = mat_structs.LowRankMatrix(X=x2, t=1.)
    full_x = np.zeros((8, 8))
    full_x[0:5, 0:5] = x1
    full_x[5:8, 5:8] = x2
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2])

    v = np.random.random(8)
    assert np.allclose(full_x.dot(full_x.dot(v)), (m**2).dot(v))


def test_BlockDiagonalMatrix_inverse():
    x1 = np.random.random((5, 5))
    x1 = x1 + x1.T + 3 * np.eye(5)
    m1 = mat_structs.LowRankMatrix(X=x1, t=1.)
    x2 = np.random.random((3, 3))
    x2 = x2 + x2.T + 3 * np.eye(3)
    m2 = mat_structs.LowRankMatrix(X=x2, t=1.)
    full_x = np.zeros((8, 8))
    full_x[0:5, 0:5] = x1
    full_x[5:8, 5:8] = x2
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2])
    v = np.random.random(8)
    assert np.allclose(
        np.linalg.inv(full_x).dot(v),
        m.inverse.dot(v)
    )

    with raises(NotImplementedError):
        m.inverse.dot_i(v, 2)

    with raises(NotImplementedError):
        m.inverse.ridge_inverse_dot(v, 1.)

    assert np.allclose(
        np.linalg.inv(full_x.dot(full_x)).dot(v),
        (m.inverse ** 2).dot(v)
    )

    with raises(NotImplementedError):
        m.inverse.diag()


def test_BlockDiagonalMatrix_diag():
    x1 = np.random.random((5, 5))
    x1 = x1 + x1.T + 3 * np.eye(5)
    m1 = mat_structs.LowRankMatrix(X=x1, t=1.)
    x2 = np.random.random((3, 3))
    x2 = x2 + x2.T + 3 * np.eye(3)
    m2 = mat_structs.LowRankMatrix(X=x2, t=1.)
    full_x = np.zeros((8, 8))
    full_x[0:5, 0:5] = x1
    full_x[5:8, 5:8] = x2
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2])

    assert np.allclose(
        np.diag(full_x),
        m.diag()
    )

    my_perm = np.random.permutation(np.arange(8))
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        perm=my_perm)
    v = np.random.random(8)
    assert np.isclose(
        np.diag(full_x).dot(v[my_perm]),
        m.diag().dot(v)
    )

    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        missing=np.arange(8, 100))
    assert np.allclose(
        m.diag()[0:8], np.diag(full_x)
    )
    assert np.allclose(
        m.diag()[8:], 0
    )


def test_BlockDiagonalMatrix_get_rank():
    x1 = np.random.random((5, 5))
    x1 = x1 + x1.T + 3 * np.eye(5)
    m1 = mat_structs.LowRankMatrix(X=x1, t=1.)
    x2 = np.random.random((3, 3))
    x2 = x2 + x2.T + 3 * np.eye(3)
    m2 = mat_structs.LowRankMatrix(X=x2, t=1.)
    full_x = np.zeros((8, 8))
    full_x[0:5, 0:5] = x1
    full_x[5:8, 5:8] = x2
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2])
    assert m.get_rank() == 8
    my_perm = np.random.permutation(np.arange(8))
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        perm=my_perm)
    assert m.get_rank() == 8
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2],
                                        missing=np.arange(8, 100))
    assert m.get_rank() == 8

    x1 = np.eye(5)
    x1[0, 0] = 0
    x1[1, 1] = 0
    m1 = mat_structs.LowRankMatrix(X=x1, t=0.5)
    m = mat_structs.BlockDiagonalMatrix(matrices=[m1, m2])
    assert m.get_rank() == 6


###############################################################################
###############################################################################
###############################################################################
# test load.py


def test_load_variant_list():
    with raises(ValueError):
        load.load_variant_list('test_data/bad_variants_missing_id.tsv')
    with raises(ValueError):
        load.load_variant_list('test_data/bad_variants_missing_a1.tsv')
    with raises(ValueError):
        load.load_variant_list('test_data/bad_variants_missing_a2.tsv')


def test_load_annotations():
    pass


def test_load_sumstats():
    pass


def test_load_ld_from_schema():
    pass
