import numpy as np
import os
import plinkio.plinkfile
from pytest import raises
from vilma import matrix_structures as mat_structs
from vilma import load
from vilma import make_ld_schema


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
            'test_data/bad_annotations_missing_annotation.tsv',
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
