import os
import plinkio.plinkfile
from pytest import raises
import numpy as np
from vilma import matrix_structures as mat_structs
from vilma import load
from vilma import make_ld_schema
from vilma import numerics
from vilma import variational_inference


# everything should be relative to this files directory
def correct_path(fname):
    return os.path.join(os.path.dirname(__file__), 'test_data', fname)


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
        load.load_variant_list(correct_path('bad_variants_missing_id.tsv'))
    with raises(ValueError):
        load.load_variant_list(correct_path('bad_variants_missing_a1.tsv'))
    with raises(ValueError):
        load.load_variant_list(correct_path('bad_variants_missing_a2.tsv'))

    variants = load.load_variant_list(correct_path('good_variants.tsv'))
    assert len(variants) == 13
    assert 'ID' in variants.columns
    assert 'A1' in variants.columns
    assert 'A2' in variants.columns


def test_load_annotations():
    variants = load.load_variant_list(correct_path('good_variants.tsv'))
    null_annotation, denylist = load.load_annotations(None, variants)
    assert null_annotation.shape == (13, 1)
    assert np.allclose(null_annotation, 1)
    assert len(denylist) == 0

    true_annotations, denylist = load.load_annotations(
        correct_path('good_annotations.tsv'), variants
    )
    assert true_annotations.shape == (13, 6)
    assert np.all(np.sum(true_annotations, axis=1) == 1)
    assert np.all(np.sum(true_annotations, axis=0)[1:] == 2)
    assert np.sum(true_annotations, axis=0)[0] == 3
    assert len(denylist) == 1
    assert denylist[0] == 12

    with raises(ValueError):
        load.load_annotations(correct_path('bad_annotations_missing_id.tsv'),
                              variants)
    with raises(ValueError):
        load.load_annotations(
            correct_path('bad_annotations_missing_annotation.tsv'),
            variants
        )


def test_load_sumstats():
    variants = load.load_variant_list(correct_path('good_variants.tsv'))
    stats, denylist = load.load_sumstats(
        correct_path('good_sumstats_beta.tsv'), variants
    )
    assert len(denylist) == 3
    assert set(denylist) == set([10, 11, 12])
    assert len(stats) == 13
    assert np.all(stats.BETA.iloc[0:10] == np.arange(10))
    assert np.all(stats.BETA.iloc[10:13] == 0.)
    assert np.all(stats.SE.iloc[0:10] == np.arange(10) + 1)
    assert np.all(stats.SE.iloc[10:13] == 1.)

    stats, denylist = load.load_sumstats(
        correct_path('good_sumstats_or.tsv'),
        variants
    )
    assert len(denylist) == 3
    assert set(denylist) == set([10, 11, 12])
    assert len(stats) == 13
    assert np.allclose(stats.BETA.iloc[0:10],
                       np.log(np.arange(10) + 1.))
    assert np.all(stats.BETA.iloc[10:13] == 0.)
    assert np.all(stats.SE.iloc[0:10] == np.arange(10) + 1)
    assert np.all(stats.SE.iloc[10:13] == 1.)

    stats, denylist = load.load_sumstats(
        correct_path('good_sumstats_flip.tsv'),
        variants
    )
    assert len(denylist) == 4
    assert set(denylist) == set([0, 10, 11, 12])
    assert len(stats) == 13
    assert np.all(stats.BETA.iloc[0:10] == -np.arange(10))
    assert np.all(stats.BETA.iloc[10:13] == 0.)
    assert np.all(stats.SE.iloc[0:10] == np.arange(10) + 1)
    assert np.all(stats.SE.iloc[10:13] == 1.)

    with raises(ValueError):
        stats, denylist = load.load_sumstats(
            correct_path('bad_sumstats_missing_id.tsv'),
            variants
        )
    with raises(ValueError):
        stats, denylist = load.load_sumstats(
            correct_path('bad_sumstats_missing_beta.tsv'),
            variants
        )
    with raises(ValueError):
        stats, denylist = load.load_sumstats(
            correct_path('bad_sumstats_missing_se.tsv'),
            variants
        )
    with raises(ValueError):
        stats, denylist = load.load_sumstats(
            correct_path('bad_sumstats_missing_a1.tsv'),
            variants
        )

    with raises(ValueError):
        stats, denylist = load.load_sumstats(
            correct_path('bad_sumstats_missing_a2.tsv'),
            variants
        )


def test_load_ld_from_schema():
    variants = load.load_variant_list(correct_path('good_variants.tsv'))
    denylist = []
    ldmat = load.load_ld_from_schema(
        correct_path('ld_manifest.tsv'), variants, denylist, 1., False
    )
    true_ldmat = np.eye(13)
    true_ldmat[0, 2] = -1
    true_ldmat[2, 0] = -1
    true_ldmat[5, 5] = 0
    true_ldmat[12, 12] = 0
    v = np.random.random(13)
    assert np.allclose(ldmat.dot(v), true_ldmat.dot(v))

    ldmat = load.load_ld_from_schema(
        correct_path('ld_manifest.tsv'), variants, denylist, 1., True
    )
    true_ldmat = np.eye(13)
    true_ldmat[0, 2] = -1
    true_ldmat[2, 0] = -1
    true_ldmat[5, 5] = 0
    true_ldmat[12, 12] = 0
    v = np.random.random(13)
    assert np.allclose(ldmat.dot(v), true_ldmat.dot(v))

    denylist = [3, 4, 5]
    ldmat = load.load_ld_from_schema(
        correct_path('ld_manifest.tsv'), variants, denylist, 1., False
    )
    true_ldmat = np.eye(13)
    true_ldmat[0, 2] = -1
    true_ldmat[2, 0] = -1
    true_ldmat[3, 3] = 0
    true_ldmat[4, 4] = 0
    true_ldmat[5, 5] = 0
    true_ldmat[12, 12] = 0
    v = np.random.random(13)
    assert np.allclose(ldmat.dot(v), true_ldmat.dot(v))


def test_load_ld_from_schema_svd():
    variants = load.load_variant_list(correct_path('good_variants.tsv'))
    denylist = []
    ldmat = load.load_ld_from_schema(
        correct_path('ld_manifest_svd.tsv'), variants, denylist, 1., False
    )
    true_ldmat = np.eye(13)
    true_ldmat[0, 2] = -1
    true_ldmat[2, 0] = -1
    true_ldmat[5, 5] = 0
    true_ldmat[12, 12] = 0
    v = np.random.random(13)
    assert np.allclose(ldmat.dot(v), true_ldmat.dot(v))

    ldmat = load.load_ld_from_schema(
        correct_path('ld_manifest_svd.tsv'), variants, denylist, 1., True
    )
    true_ldmat = np.eye(13)
    true_ldmat[0, 2] = -1
    true_ldmat[2, 0] = -1
    true_ldmat[5, 5] = 0
    true_ldmat[12, 12] = 0
    v = np.random.random(13)
    assert np.allclose(ldmat.dot(v), true_ldmat.dot(v))

    denylist = [3, 4, 5]
    ldmat = load.load_ld_from_schema(
        correct_path('ld_manifest_svd.tsv'), variants, denylist, 1., False
    )
    true_ldmat = np.eye(13)
    true_ldmat[0, 2] = -1
    true_ldmat[2, 0] = -1
    true_ldmat[3, 3] = 0
    true_ldmat[4, 4] = 0
    true_ldmat[5, 5] = 0
    true_ldmat[12, 12] = 0
    v = np.random.random(13)
    assert np.allclose(ldmat.dot(v), true_ldmat.dot(v))


###############################################################################
###############################################################################
###############################################################################
# test make_ld_schema.py


def test_make_ld_schema_get_ld_blocks():
    with raises(ValueError):
        make_ld_schema._get_ld_blocks(correct_path('bad_blocks.bed'))

    blocks = make_ld_schema._get_ld_blocks(correct_path('blocks.bed'))
    assert len(blocks) == 1
    assert '1' in blocks.keys()
    assert len(blocks['1']) == 4
    assert np.all(blocks['1']['chrom'] == '1')
    assert np.all(blocks['1']['start'] == np.array([0, 8, 100, 950]))
    assert np.all(blocks['1']['end'] == np.array([8, 100, 200, 1000]))


def test_make_ld_schema_assign_to_blocks():
    blocks = make_ld_schema._get_ld_blocks(correct_path('blocks.bed'))
    plinkdata = plinkio.plinkfile.open(correct_path(
        'sim_genotypes'
    ))

    blocked_data = make_ld_schema._assign_to_blocks(
        blocks, plinkdata
    )
    assert len(blocked_data) == 3
    assert '1 0' in blocked_data.keys()
    assert '1 1' in blocked_data.keys()
    assert '1 3' in blocked_data.keys()
    for k in blocked_data.keys():
        assert 'SNPs' in blocked_data[k].keys()
        assert 'IDs' in blocked_data[k].keys()
        assert blocked_data[k]['SNPs'].shape[1] == len(blocked_data[k]['IDs'])
        assert len(blocked_data[k]['SNPs']) == 10
    assert blocked_data['1 0']['SNPs'].shape[1] == 2
    assert blocked_data['1 1']['SNPs'].shape[1] == 1
    assert blocked_data['1 3']['SNPs'].shape[1] == 2


def test_make_ld_schema():
    # delete this file first to prevent appending to existing
    with open(correct_path('test_ld_mats.schema'), 'w'):
        pass
    plinkdata = plinkio.plinkfile.open(correct_path(
        'sim_genotypes'
    ))
    blocks = make_ld_schema._get_ld_blocks(correct_path('blocks.bed'))
    blocked_data = make_ld_schema._assign_to_blocks(blocks, plinkdata)
    make_ld_schema._process_blocks(blocked_data, correct_path('test_ld_mats'))
    with open(correct_path('test_ld_mats.schema'), 'r') as fh:
        varfile, matfile = fh.readline().split()
        assert correct_path(varfile) == correct_path('test_ld_mats_1:0.var')
        assert correct_path(matfile) == correct_path('test_ld_mats_1:0.npy')
        mat = np.load(correct_path('test_ld_mats_1:0.npy'))
        assert np.allclose(mat, np.ones_like(mat))
        assert len(mat.shape) == 2
        assert mat.shape[0] == 2
        assert mat.shape[1] == 2
        with open(correct_path(varfile), 'r') as vfh:
            assert vfh.readline() == '.\t1\t3\t0.0\tG\tT\n'
            assert vfh.readline() == '.\t1\t4\t0.0\tG\tA\n'
            assert not vfh.readline()

        varfile, matfile = fh.readline().split()
        assert correct_path(varfile) == correct_path('test_ld_mats_1:1.var')
        assert correct_path(matfile) == correct_path('test_ld_mats_1:1.npy')
        mat = np.load(correct_path('test_ld_mats_1:1.npy'))
        assert np.allclose(mat, 1)
        assert len(mat) == 1
        with open(correct_path(varfile), 'r') as vfh:
            assert vfh.readline() == '.\t1\t9\t0.0\tC\tT\n'
            assert not vfh.readline()

        varfile, matfile = fh.readline().split()
        assert correct_path(varfile) == correct_path('test_ld_mats_1:3.var')
        assert correct_path(matfile) == correct_path('test_ld_mats_1:3.npy')
        mat = np.load(correct_path('test_ld_mats_1:3.npy'))
        assert np.allclose(np.diag(mat), 1)
        assert len(mat.shape) == 2
        assert mat.shape[0] == 2
        assert mat.shape[1] == 2
        assert np.isclose(mat[0, 1], -0.28867513)
        assert np.isclose(mat[1, 0], -0.28867513)
        with open(correct_path(varfile), 'r') as vfh:
            assert vfh.readline() == '.\t1\t962\t0.0\tT\tG\n'
            assert vfh.readline() == '.\t1\t975\t0.0\tT\tC\n'
            assert not vfh.readline()
        assert not fh.readline()


def test_make_ld_schema_svd():
    # delete this file first to prevent appending to existing
    with open(correct_path('test_ld_mats_svd.schema'), 'w'):
        pass
    plinkdata = plinkio.plinkfile.open(correct_path(
        'sim_genotypes'
    ))
    blocks = make_ld_schema._get_ld_blocks(correct_path('blocks.bed'))
    blocked_data = make_ld_schema._assign_to_blocks(blocks, plinkdata)
    make_ld_schema._process_blocks(blocked_data,
                                   correct_path('test_ld_mats_svd'),
                                   ldthresh=1.)
    with open(correct_path('test_ld_mats_svd.schema'), 'r') as fh:
        varfile, matfile = fh.readline().split()
        assert (correct_path(varfile)
                == correct_path('test_ld_mats_svd_1:0.var'))
        assert (correct_path(matfile)
                == correct_path('test_ld_mats_svd_1:0.npy'))
        mat = np.load(correct_path('test_ld_mats_svd_1:0.npy'))
        assert len(mat.shape) == 2
        assert mat.shape[0] == 3
        assert mat.shape[1] == 1
        u = mat[0:2]
        s = mat[2]
        v = np.copy(u.T)
        assert np.allclose(u @ np.diag(s) @ v, 1)
        with open(correct_path(varfile), 'r') as vfh:
            assert vfh.readline() == '.\t1\t3\t0.0\tG\tT\n'
            assert vfh.readline() == '.\t1\t4\t0.0\tG\tA\n'
            assert not vfh.readline()

        varfile, matfile = fh.readline().split()
        assert (correct_path(varfile)
                == correct_path('test_ld_mats_svd_1:1.var'))
        assert (correct_path(matfile)
                == correct_path('test_ld_mats_svd_1:1.npy'))
        mat = np.load(correct_path('test_ld_mats_svd_1:1.npy'))
        assert np.allclose(mat, 1)
        assert len(mat.shape) == 2
        assert mat.shape[0] == 2
        assert mat.shape[1] == 1
        with open(correct_path(varfile), 'r') as vfh:
            assert vfh.readline() == '.\t1\t9\t0.0\tC\tT\n'
            assert not vfh.readline()

        varfile, matfile = fh.readline().split()
        assert (correct_path(varfile)
                == correct_path('test_ld_mats_svd_1:3.var'))
        assert (correct_path(matfile)
                == correct_path('test_ld_mats_svd_1:3.npy'))
        mat = np.load(correct_path('test_ld_mats_svd_1:3.npy'))
        assert len(mat.shape) == 2
        assert mat.shape[0] == 3
        assert mat.shape[1] == 2

        u = mat[0:2]
        s = mat[2]
        v = np.copy(u.T)
        mat = u @ np.diag(s) @ v

        assert np.allclose(np.diag(mat), 1)
        assert np.isclose(mat[0, 1], -0.28867513)
        assert np.isclose(mat[1, 0], -0.28867513)
        with open(correct_path(varfile), 'r') as vfh:
            assert vfh.readline() == '.\t1\t962\t0.0\tT\tG\n'
            assert vfh.readline() == '.\t1\t975\t0.0\tT\tC\n'
            assert not vfh.readline()
        assert not fh.readline()


###############################################################################
###############################################################################
###############################################################################
# test numerics.py

def test_sum_betas():
    x = np.random.random((2, 50, 3))
    y = np.random.random((2, 50, 3))
    alpha = np.random.random()
    true = alpha*x + (1-alpha)*y
    est = numerics.sum_betas(y, x, alpha)
    assert np.allclose(true, est)


def test_fast_divide():
    x = np.random.random((10, 100))
    y = np.random.random((10, 100))
    assert np.allclose(x/y, numerics.fast_divide(x, y))


def test_fast_linked_ests():
    x = np.random.random((100, 10))
    y = np.random.random((100, 10))
    w = np.random.random((100, 10))
    z = np.random.random((100, 10))
    assert np.allclose(numerics.fast_linked_ests(x, y, w, z),
                       x/y - z*w)


def test_fast_likelihood():
    mu = np.random.normal(size=(3, 2, 100))
    var = np.zeros((3, 2, 2, 100))
    var[:, np.arange(2), np.arange(2), :] = np.random.random(size=(3, 2, 100))
    std_errs = np.random.random((2, 100))
    delta = np.random.random((100, 3))
    delta /= delta.sum(axis=1, keepdims=True)
    err_scaling = np.random.random(2)
    ld_ranks = np.array([100., 100.])
    chi_stat = np.random.normal(size=2)

    adj_marg = np.random.normal(size=(2, 100))
    ld_diags = np.random.random(size=(2, 100))
    scaled_ld_diags = ld_diags / std_errs**2
    ld_mats = [np.diag(ld_diags[0]), np.diag(ld_diags[1])]

    post_means = numerics.fast_posterior_mean(mu, delta)
    diags = np.einsum('kppi->kpi', var)
    post_vars = numerics.fast_pmv(post_means, mu, delta, diags)
    linked_ests = np.empty_like(post_means)
    scaled_mu = numerics.fast_divide(post_means, std_errs)

    true_like = 0.
    for p in range(2):
        linked_ests[p] = ld_mats[p].dot(scaled_mu[p])
        this_like = -0.5 * (ld_diags[p] * post_vars[p]
                            * (std_errs[p] ** (-2))).sum()
        this_like += -0.5 * scaled_mu[p].dot(ld_mats[p].dot(scaled_mu[p]))
        this_like += post_means[p].dot(adj_marg[p])
        this_like /= err_scaling[p]
        this_like += (-0.5 * ld_ranks[p]
                      * np.log(err_scaling[p]))
        this_like += (-0.5 * chi_stat[p] / err_scaling[p])
        true_like += this_like
    fast_like = numerics.fast_likelihood(post_means,
                                         post_vars,
                                         scaled_mu,
                                         scaled_ld_diags,
                                         linked_ests,
                                         adj_marg,
                                         chi_stat,
                                         ld_ranks,
                                         err_scaling)
    assert np.isclose(true_like, fast_like)


def test_fast_posterior_mean():
    mu = np.random.normal(size=(3, 2, 100))
    delta = np.random.random((100, 3))
    delta /= delta.sum(axis=1, keepdims=True)
    true_mean = np.einsum('kpi,ik->pi',
                          mu,
                          delta)
    assert np.allclose(true_mean,
                       numerics.fast_posterior_mean(mu,
                                                    delta))


def test_fast_pmv():
    mu = np.random.normal(size=(3, 2, 100))
    var = np.zeros((3, 2, 2, 100))
    var[:, np.arange(2), np.arange(2), :] = np.random.random(size=(3, 2, 100))
    delta = np.random.random((100, 3))
    delta /= delta.sum(axis=1, keepdims=True)

    true_mean = np.einsum('kpi,ik->pi',
                          mu,
                          delta)
    mean_sq = true_mean**2
    diags = np.einsum('kppi->kpi', var)
    second_moment = np.einsum('kpi,ik->pi',
                              diags + mu**2,
                              delta)
    assert np.allclose(
        second_moment-mean_sq,
        numerics.fast_pmv(true_mean, mu, delta, diags)
    )


def test_fast_nat_inner_product_m2():
    x = np.random.normal(size=(3, 2, 100))
    v = np.random.random(size=(3, 2, 2, 100))
    assert np.allclose(
        -2*np.einsum('sqi,spqi->spi', x, v),
        numerics.fast_nat_inner_product_m2(x, v)
    )


def test_fast_nat_inner_product():
    x = np.random.normal(size=(3, 2, 100))
    v = np.random.random(size=(3, 2, 2, 100))
    assert np.allclose(
        np.einsum('sqi,spqi->spi', x, v),
        numerics.fast_nat_inner_product(x, v)
    )
    assert np.allclose(
        numerics.fast_nat_inner_product_m2(x, v),
        -2*numerics.fast_nat_inner_product(x, v)
    )


def test_fast_inner_product_comp():
    x = np.random.normal(size=(3, 2, 100))
    v = np.random.random(size=(3, 2, 2, 1))
    d = np.random.random(size=(100, 3))
    assert np.allclose(
        0.5 * np.einsum('kpi,kqi,kpqd,ik->',
                        x, x, v, d),
        numerics.fast_inner_product_comp(x, v, d)
    )
    v = np.random.random(size=(3, 2, 2, 100))
    with raises(ValueError):
        numerics.fast_inner_product_comp(x, v, d)


def test_sum_annotations():
    deltas = np.random.random((100, 5))
    annotations = np.copy(np.array([0]*50 + [1]*50))
    truth = np.zeros((2, 5))
    truth[0] = deltas[0:50].sum(axis=0)
    truth[1] = deltas[50:100].sum(axis=0)
    assert np.allclose(
        truth,
        numerics.sum_annotations(deltas, annotations, 2)
    )


def test_fast_delta_kl():
    deltas = np.random.random((100, 5))
    annotations = np.copy(np.array([0]*50 + [1]*50))
    hyper_delta = np.random.random((2, 5))
    truth = ((deltas[0:50] * np.log(deltas[0:50]/hyper_delta[0])).sum()
             + (deltas[50:] * np.log(deltas[50:]/hyper_delta[1])).sum())
    assert np.isclose(
        truth, numerics.fast_delta_kl(deltas, hyper_delta, annotations)
    )


def test_fast_beta_kl():
    x = np.random.random((100, 3))
    d = np.random.random((100, 3))
    assert np.isclose(0.5 * (x*d).sum(),
                      numerics.fast_beta_kl(x, d))


def test_fast_vi_delta_grad():
    annotations = np.copy(np.array([0]*50 + [1]*50))
    hyper_delta = np.random.random((2, 5))
    log_det = np.random.normal(size=(5))
    truth = np.zeros((100, 5))
    truth = np.log(hyper_delta)[annotations] - 0.5*log_det
    truth[:, 0:4] -= truth[:, -1:]
    truth = truth[:, 0:4]
    assert np.allclose(
        truth,
        numerics.fast_vi_delta_grad(hyper_delta, log_det, annotations)
    )


def test_map_to_nat_cat_2D():
    x = np.random.random((5, 10))
    x /= x.sum(axis=1, keepdims=True)
    nat_x = numerics.map_to_nat_cat_2D(x)
    true_nat = np.log(x[:, :-1]/x[:, -1:])
    assert np.allclose(nat_x, true_nat)


def test_invert_nat_cat_2D():
    x = np.random.random((5, 10))
    x /= x.sum(axis=1, keepdims=True)
    nat_x = numerics.map_to_nat_cat_2D(x)
    inv_nat_x = numerics.invert_nat_cat_2D(nat_x)
    assert np.allclose(x, inv_nat_x)

    nat_x = np.random.normal(size=(5, 10))
    inv_nat_x = numerics.invert_nat_cat_2D(nat_x)
    nat_x_ext = np.zeros((5, 11))
    nat_x_ext[:, :-1] = nat_x
    true = np.exp(nat_x_ext)
    true /= true.sum(axis=1, keepdims=True)
    assert np.allclose(inv_nat_x, true)


def test_fast_invert_nat_vi_delta():
    new_mu = np.random.random((3, 2, 100))
    nat_mu = np.random.random((3, 2, 100))
    const_part = np.random.random((100, 3))
    nat_vi_delta = np.random.random((100, 2))

    quad_forms = (new_mu * nat_mu).sum(axis=1)
    nat_adj = -.5*(quad_forms.T + const_part)
    nat_adj = nat_adj[:, :-1] - nat_adj[:, -1:]
    true_vi_delta = numerics.invert_nat_cat_2D(
        nat_vi_delta - nat_adj
    )

    fast_vi_delta = numerics.fast_invert_nat_vi_delta(
        new_mu,
        nat_mu,
        const_part,
        nat_vi_delta
    )

    assert np.allclose(true_vi_delta, fast_vi_delta)


def test_matrix_invert_4d_numba():
    x = np.random.random((2, 10, 2, 2))
    x = x + np.transpose(x, [0, 1, 3, 2])
    x[:, :, np.arange(2), np.arange(2)] += 3
    assert np.allclose(
        numerics._matrix_invert_4d_numba(x),
        np.linalg.inv(x)
    )
    x = np.random.random((2, 10, 1, 1))
    x = x + np.transpose(x, [0, 1, 3, 2])
    x[:, :, np.arange(1), np.arange(1)] += 3
    assert np.allclose(
        numerics._matrix_invert_4d_numba(x),
        np.linalg.inv(x)
    )


def test_matrix_invert():
    x = np.random.random((2, 2))
    x = x + x.T + 3 * np.eye(2)
    assert np.allclose(
        np.linalg.inv(x),
        numerics.matrix_invert(x)
    )
    x = np.random.random((10, 2, 2))
    x = x + np.transpose(x, [0, 2, 1])
    x[:, np.arange(2), np.arange(2)] += 3
    assert np.allclose(
        numerics.matrix_invert(x),
        np.linalg.inv(x)
    )
    x = np.random.random((2, 10, 2, 2))
    x = x + np.transpose(x, [0, 1, 3, 2])
    x[:, :, np.arange(2), np.arange(2)] += 3
    assert np.allclose(
        numerics.matrix_invert(x),
        np.linalg.inv(x)
    )
    x = np.random.random((2, 2, 10, 2, 2))
    x = x + np.transpose(x, [0, 1, 2, 4, 3])
    x[:, :, :, np.arange(2), np.arange(2)] += 3
    assert np.allclose(
        numerics.matrix_invert(x),
        np.linalg.inv(x)
    )


def test_vi_sigma_inv():
    x = np.random.random((3, 2, 2, 100))
    x = x + np.transpose(x, [0, 2, 1, 3])
    x[:, np.arange(2), np.arange(2), :] += 3
    true = np.array(
        [np.linalg.inv(m.T).T for m in x]
    )
    assert np.allclose(
        numerics.vi_sigma_inv(x),
        true
    )


def test_matrix_log_det_4d_numba():
    x = np.random.random((3, 100, 2, 2))
    x = x + np.transpose(x, [0, 1, 3, 2])
    x[:, :, np.arange(2), np.arange(2)] += 3
    assert np.allclose(
        numerics._matrix_log_det_4d_numba(x),
        np.linalg.slogdet(x)[1]
    )
    x = np.random.random((3, 100, 1, 1))
    assert np.allclose(
        numerics._matrix_log_det_4d_numba(x),
        np.linalg.slogdet(x)[1]
    )


def test_matrix_log_det():
    x = np.random.random((3, 100, 2, 2))
    x = x + np.transpose(x, [0, 1, 3, 2])
    x[:, :, np.arange(2), np.arange(2)] += 3
    assert np.allclose(
        numerics.matrix_log_det(x),
        np.linalg.slogdet(x)[1]
    )
    x = np.random.random((3, 100, 4, 4))
    x = x + np.transpose(x, [0, 1, 3, 2])
    x[:, :, np.arange(4), np.arange(4)] += 3
    assert np.allclose(
        numerics.matrix_log_det(x),
        np.linalg.slogdet(x)[1]
    )
    x = np.random.random((4, 4))
    x = x + np.transpose(x, [1, 0])
    x[np.arange(4), np.arange(4)] += 3
    assert np.allclose(
        numerics.matrix_log_det(x),
        np.linalg.slogdet(x)[1]
    )


def test_vi_sigma_log_det():
    x = np.random.random((3, 2, 2, 100))
    x = x + np.transpose(x, [0, 2, 1, 3])
    x[:, np.arange(2), np.arange(2), :] += 3

    true = np.array(
        [np.linalg.slogdet(m.T)[1].T for m in x]
    )
    assert np.allclose(
        numerics.vi_sigma_log_det(x),
        true
    )


###############################################################################
###############################################################################
###############################################################################
# test variational_inference.py


def make_vischeme(checkpoint, num_annotations, scaled, scale_se):
    betas = np.arange(100).reshape(2, 50).astype(float)
    std_errs = np.array([1]*50 + [2]*50).reshape(2, 50).astype(float)
    ld_mat = (1+np.arange(50*50)).reshape(50, 50)/(50*50+1)
    ld_mat = ld_mat + ld_mat.T + 5*np.eye(50)
    diags = np.diag(1/np.sqrt(np.diag(ld_mat)))
    ld_mat = diags @ ld_mat @ diags
    ld_mat = mat_structs.LowRankMatrix(X=ld_mat, t=1.0)
    ld_mats = [mat_structs.BlockDiagonalMatrix([ld_mat]),
               mat_structs.BlockDiagonalMatrix([ld_mat])]
    mixture_covs = [np.eye(2), 2*np.eye(2)]
    if num_annotations == 2:
        annotations = np.zeros((50, 2), dtype=int)
        annotations[0:25, 0] = 1
        annotations[25:, 1] = 1
    else:
        annotations = np.ones((50, 1), dtype=int)
    gwas_n = np.array([100e3, 10e3])
    init_hg = np.array([0.1, 0.9])
    vischeme = variational_inference.MultiPopVI(
        marginal_effects=betas,
        std_errs=std_errs,
        ld_mats=ld_mats,
        mixture_covs=mixture_covs,
        annotations=annotations,
        checkpoint=checkpoint,
        checkpoint_freq=-1,
        output='test',
        scaled=scaled,
        scale_se=scale_se,
        gwas_N=gwas_n,
        init_hg=init_hg,
        num_its=20
    )
    return vischeme


def make_vischeme_unlinked(checkpoint, num_annotations, scaled, scale_se):
    betas = np.arange(100).reshape(50, 2).T.astype(float)
    std_errs = np.array([1.]*50 + [2.]*50).reshape(2, 50).astype(float)
    ld_mat = np.eye(50)
    ld_mat = mat_structs.LowRankMatrix(X=ld_mat, t=1.0)
    ld_mats = [mat_structs.BlockDiagonalMatrix([ld_mat]),
               mat_structs.BlockDiagonalMatrix([ld_mat])]
    mixture_covs = [np.eye(2), 2*np.eye(2)]
    if num_annotations == 2:
        annotations = np.zeros((50, 2), dtype=int)
        annotations[0:25, 0] = 1
        annotations[25:, 1] = 1
    else:
        annotations = np.ones((50, 1), dtype=int)
    gwas_n = np.array([100e3, 10e3])
    init_hg = np.array([0.1, 0.9])
    vischeme = variational_inference.MultiPopVI(
        marginal_effects=betas,
        std_errs=std_errs,
        ld_mats=ld_mats,
        mixture_covs=mixture_covs,
        annotations=annotations,
        checkpoint=checkpoint,
        checkpoint_freq=-1,
        output='test',
        scaled=scaled,
        scale_se=scale_se,
        gwas_N=gwas_n,
        init_hg=init_hg,
        num_its=20
    )
    return vischeme


def test_MultiPopVI_init():
    vischeme = make_vischeme(checkpoint=False,
                             num_annotations=2,
                             scaled=False,
                             scale_se=False)

    betas = np.arange(100).reshape(2, 50).astype(float)
    std_errs = np.array([1]*50 + [2]*50).reshape(2, 50).astype(float)
    ld_mat = (1+np.arange(50*50)).reshape(50, 50)/(50*50+1)
    ld_mat = ld_mat + ld_mat.T + 5*np.eye(50)
    diags = np.diag(1/np.sqrt(np.diag(ld_mat)))
    ld_mat = diags @ ld_mat @ diags
    assert vischeme.num_pops == 2
    assert vischeme.num_mix == 2
    assert (set(vischeme.param_names)
            == set(['vi_mu', 'vi_delta', 'hyper_delta']))
    assert np.allclose(vischeme.mixture_prec[0, :, :, 0],
                       np.eye(2))
    assert np.allclose(vischeme.mixture_prec[1, :, :, 0],
                       np.eye(2)*0.5)
    assert vischeme.mixture_prec.shape == (2, 2, 2, 1)
    assert np.allclose(
        vischeme.log_det,
        [0., 2*np.log(2)]
    )
    true_vi_sigma = np.zeros((2, 2, 2, 50))
    true_vi_sigma[0, 0, 0, :] = 1/2
    true_vi_sigma[0, 1, 1, :] = 4/5
    true_vi_sigma[1, 0, 0, :] = 2/3
    true_vi_sigma[1, 1, 1, :] = 4/3

    true_nat_sigma = np.zeros((2, 2, 2, 50))
    true_nat_sigma[0, 0, 0, :] = -1
    true_nat_sigma[0, 1, 1, :] = -5/8
    true_nat_sigma[1, 0, 0, :] = -3/4
    true_nat_sigma[1, 1, 1, :] = -3/8

    assert np.allclose(vischeme.vi_sigma, true_vi_sigma)
    assert np.allclose(vischeme.nat_sigma, true_nat_sigma)

    true_sigma_log_det = np.zeros((2, 50))
    true_sigma_log_det[0, :] = np.log(2/5)
    true_sigma_log_det[1, :] = np.log(8/9)
    assert np.allclose(vischeme.vi_sigma_log_det, true_sigma_log_det)

    true_matches = np.zeros((50, 2))
    true_matches[:, 0] = 1/2 + 4/5
    true_matches[:, 1] = 1/3 + 2/3
    assert np.allclose(vischeme.vi_sigma_matches, true_matches)

    true_summary = (np.array([0., 2*np.log(2)])
                    - true_sigma_log_det.T
                    + true_matches)
    assert np.allclose(vischeme.sigma_summary, true_summary)

    assert vischeme.nat_grad_vi_delta is None
    assert not vischeme.scaled
    assert np.allclose(vischeme.error_scaling, 1)
    assert not vischeme.scale_se
    assert vischeme.num_annotations == 2
    assert np.allclose(vischeme.marginal_effects,
                       np.arange(100).reshape(2, 50))
    assert np.allclose(vischeme.std_errs,
                       std_errs)
    assert np.allclose(vischeme.scalings, 1)
    assert np.allclose(vischeme.ld_diags, 1)
    assert np.allclose(vischeme.scaled_ld_diags,
                       std_errs**-2)
    assert len(vischeme.ld_mats) == 2
    assert vischeme.ld_mats[0].shape == (50, 50)
    assert np.allclose(vischeme.ld_mats[0].diag(), 1)
    assert np.allclose(vischeme.annotations[0:25], 0)
    assert np.allclose(vischeme.annotations[25:], 1)
    assert np.allclose(vischeme.annotation_counts, 25)
    assert vischeme.checkpoint_freq == -1
    assert vischeme.checkpoint_path == 'test-checkpoint'
    assert np.allclose(vischeme.init_hg, [0.1, 0.9])
    assert np.allclose(vischeme.gwas_N, [100e3, 10e3])
    assert vischeme.num_its == 20
    assert np.allclose(
        vischeme.adj_marginal_effects[0],
        (np.linalg.inv(ld_mat).dot(
            ld_mat.dot(betas[0]/std_errs[0])))/std_errs[0]
    )
    assert np.allclose(
        vischeme.adj_marginal_effects[1],
        (np.linalg.inv(ld_mat).dot(
            ld_mat.dot(betas[1]/std_errs[1])))/std_errs[1]
    )
    assert np.allclose(
        vischeme.chi_stat[0],
        betas[0].dot(np.linalg.inv(ld_mat).dot(
            betas[0]/std_errs[0])/std_errs[0]
        )
    )
    assert np.allclose(
        vischeme.chi_stat[1],
        betas[1].dot(np.linalg.inv(ld_mat).dot(
            betas[1]/std_errs[1])/std_errs[1]
        )
    )

    assert np.allclose(vischeme.ld_ranks, 50)

    true_ridge_betas = np.zeros((2, 50))
    prior = (2 * np.array([100e3, 10e3]) * np.array([0.1, 0.9])
             / (std_errs**-2).sum(axis=1))
    temp = ld_mat.dot(np.linalg.inv(ld_mat).dot((betas / std_errs).T)).T

    for p in range(2):
        true_ridge_betas[p, :] = np.linalg.inv(
            ld_mat + np.diag(std_errs[p]**2)/prior[p]
        ).dot(temp[p]) * std_errs[p]

    assert np.allclose(vischeme.inverse_betas, true_ridge_betas)


def test_MultiPopVI_init_flags():
    vischeme = make_vischeme(checkpoint=True,
                             num_annotations=2,
                             scaled=False,
                             scale_se=False)
    assert vischeme.checkpoint
    vischeme = make_vischeme(checkpoint=False,
                             num_annotations=2,
                             scaled=False,
                             scale_se=True)
    assert vischeme.scale_se
    vischeme = make_vischeme(checkpoint=False,
                             num_annotations=2,
                             scaled=True,
                             scale_se=False)
    assert vischeme.scaled

    betas = np.arange(100).reshape(2, 50).astype(float)
    std_errs = np.array([1]*50 + [2]*50).reshape(2, 50).astype(float)
    ld_mat = (1+np.arange(50*50)).reshape(50, 50)/(50*50+1)
    ld_mat = ld_mat + ld_mat.T + 5*np.eye(50)
    diags = np.diag(1/np.sqrt(np.diag(ld_mat)))
    ld_mat = diags @ ld_mat @ diags
    assert np.allclose(vischeme.scalings, std_errs)
    assert np.allclose(vischeme.adj_marginal_effects, betas/std_errs)
    assert np.allclose(vischeme.std_errs, 1)
    assert np.allclose(vischeme.scaled_ld_diags, 1)
    true_ridge_betas = np.zeros((2, 50))
    prior = 2 * np.array([100e3, 10e3]) * np.array([0.1, 0.9]) / 50
    temp = ld_mat.dot(np.linalg.inv(ld_mat).dot((betas / std_errs).T)).T

    for p in range(2):
        true_ridge_betas[p, :] = np.linalg.inv(
            ld_mat + np.diag(np.ones_like(std_errs[p]))/prior[p]
        ).dot(temp[p])

    assert np.allclose(vischeme.inverse_betas, true_ridge_betas)


def test_MultiPopVI_fast_einsum():
    vischeme = make_vischeme(False, 1, False, False)
    x = np.random.random((3, 4, 5))
    y = np.random.random((4, 5, 6))
    test = vischeme._fast_einsum('ijk,jkl->ik',
                                 x, y,
                                 key='test_key')
    assert 'test_key' in vischeme._einsum_paths.keys()
    true_path = np.einsum_path('ijk,jkl->ik',
                               x, y,
                               optimize='optimal')[0]
    assert vischeme._einsum_paths['test_key'] == true_path
    true = np.einsum('ijk,jkl->ik', x, y)
    assert np.allclose(true, test)
    test2 = vischeme._fast_einsum('ijk,jkl->ik',
                                  x, y,
                                  key='test_key')
    assert np.allclose(true, test2)


def test_MultiPopVI_optimize():
    vischeme = make_vischeme(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()
    params = (mu, delta, hyper)
    new_params = vischeme.optimize()
    assert vischeme.elbo(new_params) > vischeme.elbo(params)

    vischeme = make_vischeme(False, 1, True, False)
    mu, delta, hyper = vischeme._initialize()
    params = (mu, delta, hyper)
    new_params = vischeme.optimize()
    assert vischeme.elbo(new_params) > vischeme.elbo(params)

    vischeme = make_vischeme(False, 2, False, True)
    mu, delta, hyper = vischeme._initialize()
    params = (mu, delta, hyper)
    new_params = vischeme.optimize()
    assert vischeme.elbo(new_params) > vischeme.elbo(params)

    vischeme = make_vischeme(False, 2, True, True)
    mu, delta, hyper = vischeme._initialize()
    params = (mu, delta, hyper)
    new_params = vischeme.optimize()
    assert vischeme.elbo(new_params) > vischeme.elbo(params)


def test_MultiPopVI_nat_grad_step():
    vischeme = make_vischeme_unlinked(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()
    params = (mu, delta, hyper)
    (new_params, new_L, new_elbo_delta) = vischeme._nat_grad_step(
        params, [1.0, 1., 1.], 2., None
    )
    assert new_L[0] == 1.
    assert vischeme.elbo(new_params) > vischeme.elbo(params)

    (new_params, new_L, new_elbo_delta) = vischeme._nat_grad_step(
        params, [variational_inference.L_MAX-1, 1., 1.], 2., None
    )
    assert new_L[0] < variational_inference.L_MAX-1
    assert vischeme.elbo(new_params) > vischeme.elbo(params)
    assert np.allclose(params[0], new_params[0])


def test_MultiPopVI_elbo():
    vischeme = make_vischeme(False, 2, False, False)
    mu, delta, hyper = vischeme._initialize()
    params = (mu, delta, hyper)
    assert np.isclose(
        vischeme.elbo(params),
        vischeme._log_likelihood(params)
        - vischeme._beta_KL(*params)
        - vischeme._annotation_KL(*params)
    )


def test_MultiPopVI_optimize_step():
    vischeme = make_vischeme_unlinked(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()

    initial_nat_grad_vi_delta = np.copy(vischeme.nat_grad_vi_delta)
    initial_vi_sigma = np.copy(vischeme.vi_sigma)
    initial_nat_sigma = np.copy(vischeme.nat_sigma)
    initial_vi_sigma_log_det = np.copy(vischeme.vi_sigma_log_det)

    mu_copy = np.copy(mu)
    delta_copy = np.copy(delta)
    hyper_copy = np.copy(hyper)
    params = (mu, delta, hyper)

    init_delta = vischeme._nat_to_not_vi_delta(params)[1]
    assert np.allclose(delta, init_delta)

    (new_params, new_L, new_elbo_delta) = vischeme._nat_grad_step(
        params, [1.0, 1., 1., 1., 1.], 2., None
    )
    assert np.allclose(mu, mu_copy)
    assert np.allclose(delta, delta_copy)
    assert np.allclose(hyper, hyper_copy)
    assert np.allclose(params[0], mu_copy)
    assert np.allclose(params[1], delta_copy)
    assert np.allclose(params[2], hyper_copy)
    assert np.allclose(vischeme.vi_sigma,
                       initial_vi_sigma)
    assert np.allclose(vischeme.nat_sigma,
                       initial_nat_sigma)
    assert np.allclose(vischeme.vi_sigma_log_det,
                       initial_vi_sigma_log_det)

    vischeme.nat_grad_vi_delta = initial_nat_grad_vi_delta
    init_delta = vischeme._nat_to_not_vi_delta(params)[1]
    assert np.allclose(init_delta, delta_copy)
    (opt_params,
     L_new, elbo, running_elbo_delta) = vischeme._optimize_step(
        params, [1.0, 1., 1., 1., 1.], vischeme.elbo(params),
         line_search_rate=2.
    )
    assert np.allclose(new_params[0], opt_params[0])
    assert np.allclose(new_params[1], opt_params[1]), (new_params[1][0],
                                                       opt_params[1][0])
    assert np.allclose(new_params[2], opt_params[2])
    assert np.isclose(elbo, vischeme.elbo(opt_params))


def test_MultiPopVI_log_likelihood():
    # Covered by numerics.fast_likelihood
    pass


def test_MultiPopVI_update_error_scaling():
    vischeme = make_vischeme(False, 2, False, False)
    mu, delta, hyper = vischeme._initialize()
    params = (mu, delta, hyper)

    post_mean = vischeme._posterior_mean(mu, delta, hyper)
    pmv = vischeme._posterior_marginal_variance(post_mean, mu, delta, hyper)

    true_tau = np.zeros(2)
    for p in range(2):
        true_tau[p] = 1. / vischeme.ld_ranks[p] * (
            vischeme.chi_stat[p]
            - 2 * vischeme.adj_marginal_effects[p].dot(post_mean[p])
            + post_mean[p].dot(
                vischeme.ld_mats[p].dot(post_mean[p]/vischeme.std_errs[p])
                / vischeme.std_errs[p]
            )
            + (vischeme.scaled_ld_diags[p] * pmv[p]).sum()
        )

    vischeme._update_error_scaling(params)

    assert np.allclose(vischeme.error_scaling, true_tau)


def test_MultiPopVI_beta_objective():
    vischeme = make_vischeme(False, 2, False, False)
    mu, delta, hyper = vischeme._initialize()
    params = (mu, delta, hyper)
    assert np.isclose(vischeme._beta_objective(params),
                      vischeme._log_likelihood(params)
                      - vischeme._beta_KL(*params))


def test_MultiPopVI_hyper_delta_objective():
    vischeme = make_vischeme(False, 2, False, False)
    mu, delta, hyper = vischeme._initialize()
    params = (mu, delta, hyper)
    assert np.isclose(vischeme._hyper_delta_objective(params),
                      -vischeme._delta_KL(*params)
                      - vischeme._annotation_KL(*params))


def test_MultiPopVI_annotation_objective():
    vischeme = make_vischeme(False, 2, False, False)
    mu, delta, hyper = vischeme._initialize()
    params = (mu, delta, hyper)
    assert np.isclose(vischeme._annotation_objective(params),
                      -vischeme._annotation_KL(*params))


def test_MultiPopVI_annotation_KL():
    vischeme = make_vischeme(checkpoint=False,
                             num_annotations=2,
                             scaled=False,
                             scale_se=False)
    assert vischeme._annotation_KL() == 0.


def test_MultiPopVI_beta_KL():
    vischeme = make_vischeme(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()


def test_MultiPopVI_delta_KL():
    vischeme = make_vischeme(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()
    assert np.isclose(
        vischeme._delta_KL(mu, delta, hyper),
        (delta * np.log(delta / hyper[0])).sum()
    )

    vischeme = make_vischeme(False, 2, False, False)
    mu, delta, hyper = vischeme._initialize()
    assert np.isclose(
        vischeme._delta_KL(mu, delta, hyper),
        (delta * np.log(delta / hyper[0])).sum()
    )


def test_MultiPopVI_update_annotation():
    vischeme = make_vischeme(False, 1, False, False)
    L = 3
    mu, delta, hyper = vischeme._initialize()
    ((new_mu, new_delta, new_hyper),
     new_L,
     old_obj,
     new_obj) = vischeme._update_annotation(mu, delta, hyper, 0., L, 0, 0)
    assert new_L == L
    assert old_obj == new_obj
    assert np.allclose(new_mu, mu)
    assert np.allclose(new_delta, delta)
    assert np.allclose(new_hyper, hyper)


def test_MultiPopVI_update_hyper_delta():
    vischeme = make_vischeme_unlinked(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()
    ((new_mu, new_delta, new_hyper),
     L, orig_obj, new_obj) = vischeme._update_hyper_delta(
         mu, delta, hyper, None, [1.], 0, 1.25
     )

    assert L[0] == 1
    assert new_obj > orig_obj
    assert np.allclose(mu, new_mu)
    assert np.allclose(delta.mean(axis=0), new_hyper[0])

    vischeme = make_vischeme_unlinked(False, 2, False, False)
    mu, delta, hyper = vischeme._initialize()
    ((new_mu, new_delta, new_hyper),
     L, orig_obj, new_obj) = vischeme._update_hyper_delta(
         mu, delta, hyper, None, [1.], 0, 1.25
     )

    assert L[0] == 1
    assert new_obj > orig_obj
    assert np.allclose(mu, new_mu)
    assert np.allclose(delta[0:25].mean(axis=0), new_hyper[0])
    assert np.allclose(delta[25:].mean(axis=0), new_hyper[1])


def test_MultiPopVI_update_beta():
    vischeme = make_vischeme_unlinked(False, 1, False, False)
    # For unlinked, should be able to take full natural gradient step
    mu, delta, hyper = vischeme._initialize()
    ((new_mu, new_delta, new_hyper),
     L, orig_obj, new_obj) = vischeme._update_beta(
         mu, delta, hyper, None, [1.], 0, 1.25
     )

    assert L[0] == 1
    assert new_obj > orig_obj
    assert np.allclose(new_hyper, hyper)

    # And it should be "idempotent"
    ((second_mu, second_delta, second_hyper),
     L, orig_obj, new_obj) = vischeme._update_beta(
         new_mu, new_delta, new_hyper, None, [1.], 0, 1.25
     )

    assert np.isclose(new_obj, orig_obj)
    assert np.allclose(new_mu, second_mu)
    assert np.allclose(new_delta, second_delta)
    assert np.allclose(new_hyper, second_hyper)


def test_MultiPopVI_posterior_marginal_variance():
    vischeme = make_vischeme(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()
    post_mean = vischeme._posterior_mean(mu, delta, hyper)
    second_moment = np.einsum('kpi,ik->pi',
                              mu**2 + vischeme.vi_sigma[:, [0, 1], [0, 1], :],
                              delta)
    assert np.allclose(
        vischeme._posterior_marginal_variance(post_mean, mu, delta, hyper),
        second_moment - post_mean**2
    )
    vischeme = make_vischeme(False, 1, True, False)
    mu, delta, hyper = vischeme._initialize()
    post_mean = vischeme._posterior_mean(mu, delta, hyper)
    second_moment = np.einsum('kpi,ik->pi',
                              mu**2 + vischeme.vi_sigma[:, [0, 1], [0, 1], :],
                              delta)
    assert np.allclose(
        vischeme._posterior_marginal_variance(post_mean, mu, delta, hyper),
        second_moment - post_mean**2
    )


def test_MultiPopVI_posterior_mean():
    vischeme = make_vischeme_unlinked(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()
    post_mean = vischeme._posterior_mean(mu, delta, hyper)
    assert np.allclose(
        post_mean,
        np.einsum('kpi,ik->pi', mu, delta)
    )
    vischeme = make_vischeme_unlinked(False, 1, True, False)
    mu, delta, hyper = vischeme._initialize()
    post_mean = vischeme._posterior_mean(mu, delta, hyper)
    assert np.allclose(
        post_mean,
        np.einsum('kpi,ik->pi', mu, delta)
    )


def test_MultiPopVI_real_posterior_mean():
    vischeme = make_vischeme_unlinked(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()
    post_mean = vischeme.real_posterior_mean(mu, delta, hyper)
    assert np.allclose(
        post_mean,
        np.einsum('kpi,ik->pi', mu, delta)
    )

    vischeme = make_vischeme_unlinked(False, 1, True, False)
    std_errs = np.array([1]*50 + [2]*50).reshape(2, 50).astype(float)
    mu, delta, hyper = vischeme._initialize()
    post_mean = vischeme.real_posterior_mean(mu, delta, hyper)
    assert np.allclose(
        post_mean,
        np.einsum('kpi,ik,pi->pi', mu, delta, std_errs),
    )


def test_MultiPopVI_real_posterior_variance():
    vischeme = make_vischeme_unlinked(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()
    post_var = vischeme.real_posterior_variance(mu, delta, hyper)
    real_var = (np.einsum('kppi,ik->pi',
                          vischeme.vi_sigma,
                          delta)
                + np.einsum('kpi,ik->pi',
                            mu**2,
                            delta)
                - np.einsum('kpi,ik->pi',
                            mu,
                            delta)**2)
    assert np.allclose(
        post_var,
        real_var
    )

    vischeme = make_vischeme_unlinked(False, 1, True, False)
    std_errs = np.array([1]*50 + [2]*50).reshape(2, 50).astype(float)
    mu, delta, hyper = vischeme._initialize()
    post_var = vischeme.real_posterior_variance(mu, delta, hyper)
    real_var = (np.einsum('kppi,ik->pi',
                          vischeme.vi_sigma,
                          delta)
                + np.einsum('kpi,ik->pi',
                            mu**2,
                            delta)
                - np.einsum('kpi,ik->pi',
                            mu,
                            delta)**2)
    real_var *= std_errs**2
    assert np.allclose(
        post_var,
        real_var
    )


def test_MultiPopVI_initialize():
    # Initialization is somewhat arbitrary anyway
    # This just checks to make sure it's sensible --
    # if SNPs are unlinked no SNPs should change sign
    # and all SNPs should have true effects smaller than
    # observed effects.  SNPs with larger observations
    # should have large true effects
    vischeme = make_vischeme_unlinked(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()
    betas = np.arange(100).reshape(50, 2).T.astype(float)
    assert np.all(np.abs(mu[0, :, 1:]) < np.abs(betas[:, 1:]))
    assert np.all(np.abs(mu[1, :, 1:]) < np.abs(betas[:, 1:]))
    assert np.all((betas[:, 1:] < 0)[(mu[0, :, 1:] < 0)])
    assert np.all((betas[:, 1:] > 0)[(mu[0, :, 1:] > 0)])
    assert np.all((betas[:, 1:] < 0)[(mu[1, :, 1:] < 0)])
    assert np.all((betas[:, 1:] > 0)[(mu[1, :, 1:] > 0)])
    assert np.all(np.diff(mu[0, 0, :]) > 0)
    assert np.all(np.diff(mu[0, 1, :]) > 0)
    assert np.all(np.diff(mu[1, 0, :]) > 0)
    assert np.all(np.diff(mu[1, 1, :]) > 0)
    assert np.all(hyper > 1/500)


def test_MultiPopVI_nat_to_not_vi_delta():
    vischeme = make_vischeme_unlinked(False, 1, False, False)
    mu, delta, hyper = vischeme._initialize()
    nat_deltas = vischeme.nat_grad_vi_delta
    inv_vi_sigmas = numerics.vi_sigma_inv(vischeme.vi_sigma)
    log_ratios = (
        nat_deltas[:, 0]
        + 0.5 * np.einsum('pi,pqi,qi->i',
                          mu[0],
                          inv_vi_sigmas[0],
                          mu[0])
        + 0.5 * vischeme.vi_sigma_log_det[0]
        - 0.5 * np.einsum('pi,pqi,qi->i',
                          mu[1],
                          inv_vi_sigmas[1],
                          mu[1])
        - 0.5 * vischeme.vi_sigma_log_det[1]
    )
    true_deltas = numerics.invert_nat_cat_2D(log_ratios.reshape((-1, 1)))
    new_mu, new_delta, new_hyper = vischeme._nat_to_not_vi_delta(
        (mu, delta, hyper)
    )
    assert np.allclose(
        true_deltas,
        new_delta
    )
    assert np.allclose(mu, new_mu)
    assert np.allclose(hyper, new_hyper)
