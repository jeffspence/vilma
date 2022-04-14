import numpy as np
import os
import plinkio.plinkfile
from pytest import raises
from vilma import matrix_structures as mat_structs
from vilma import load
from vilma import make_ld_schema
from vilma import numerics


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
        assert varfile == correct_path('test_ld_mats_1:0.var')
        assert matfile == correct_path('test_ld_mats_1:0.npy')
        mat = np.load(correct_path('test_ld_mats_1:0.npy'))
        assert np.allclose(mat, np.ones_like(mat))
        assert len(mat.shape) == 2
        assert mat.shape[0] == 2
        assert mat.shape[1] == 2
        with open(varfile, 'r') as vfh:
            assert vfh.readline() == '.\t1\t3\t0.0\tG\tT\n'
            assert vfh.readline() == '.\t1\t4\t0.0\tG\tA\n'
            assert not vfh.readline()

        varfile, matfile = fh.readline().split()
        assert varfile == correct_path('test_ld_mats_1:1.var')
        assert matfile == correct_path('test_ld_mats_1:1.npy')
        mat = np.load(correct_path('test_ld_mats_1:1.npy'))
        assert np.allclose(mat, 1)
        assert len(mat) == 1
        with open(varfile, 'r') as vfh:
            assert vfh.readline() == '.\t1\t9\t0.0\tC\tT\n'
            assert not vfh.readline()

        varfile, matfile = fh.readline().split()
        assert varfile == correct_path('test_ld_mats_1:3.var')
        assert matfile == correct_path('test_ld_mats_1:3.npy')
        mat = np.load(correct_path('test_ld_mats_1:3.npy'))
        assert np.allclose(np.diag(mat), 1)
        assert len(mat.shape) == 2
        assert mat.shape[0] == 2
        assert mat.shape[1] == 2
        assert np.isclose(mat[0, 1], -0.28867513)
        assert np.isclose(mat[1, 0], -0.28867513)
        with open(varfile, 'r') as vfh:
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
