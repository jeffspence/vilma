from __future__ import print_function
from __future__ import division

import numpy as np
import string
# import scipy.sparse.linalg


def diag_approx(X, t):
    s, vecs = np.linalg.eigh(X)
    # u, s, v = np.linalg.svd(X)
    big_sing_vals = np.where(s >= 1 - np.sqrt(t))[0]
    if len(big_sing_vals > 0):
        k = big_sing_vals
    else:
        k = np.array([])
    u = np.copy(vecs[:, k])
    s = np.copy(s[k])
    v = np.copy(u.T)
    return u, s, v, np.zeros(X.shape[0])
    '''
    if k >= X.shape[0]:
        u, s, v = np.linalg.svd(X)
        return u, s, v, np.zeros(X.shape[0])
    converged = False
    v0 = None
    D = np.diag(np.zeros(X.shape[0]))
    frob = float('inf')
    while not converged:
        u, s, v = scipy.sparse.linalg.svds(X - D, k, v0=v0)
        keep = np.abs(s) > (1e-12 * np.max(np.abs(s)))
        u = u[:, keep]
        s = s[keep]
        v = v[keep, :]
        M = (u * s).dot(v)
        D = np.diag(np.diag(X) - np.diag(M))
        v0 = np.copy(v[-1, :])
        new_frob = ((M + D - X)**2).sum()
        if np.isinf(frob):
            print('Naive Frobenius Norm',
                  ((X - M)**2).sum() / (X**2).sum())
            print('Naive RMSE on the diagonal',
                  np.sqrt(((np.diag(M) - np.diag(X))**2).mean()))
            assert np.all(np.diag(D) > -1e-13)
        converged = np.isclose(new_frob, frob, atol=1e-6, rtol=1e-3)
        frob = new_frob
    print('Final Frobenius Norm:', frob / (X**2).sum())
    print('RMSE difference on the diagonal',
          np.sqrt(((np.diag(M) + np.diag(D) - np.diag(X))**2).mean()))
    return u, s, v, np.diag(D)
    '''


def get_random_string():
    return ''.join(np.random.choice(
        list(string.ascii_letters + string.digits), size=100))


class SVD(object):
    def __init__(self, X=None, t=0.0, u=None, s=None, v=None, D=None,
                 hdf_file=None):
        if X is not None:
            assert u is None
            assert s is None
            assert v is None
            assert D is None
            u, s, v, D = diag_approx(X, t)
        else:
            assert X is None
            assert u is not None
            assert s is not None
            assert v is not None
            assert D is not None
        self.D = np.copy(D)
        keep = np.abs(s) > (1e-12 * np.max(np.abs(s)))
        if hdf_file is None:
            self.u = np.zeros((u.shape[0], keep.sum()))
            self.v = np.zeros((keep.sum(), v.shape[1]))
        else:
            random_string = get_random_string()
            while random_string in hdf_file.keys():
                random_string = get_random_string()
            self.u = hdf_file.create_dataset(random_string,
                                             (u.shape[0], keep.sum()),
                                             dtype=np.float64)
            random_string = get_random_string()
            while random_string in hdf_file.keys():
                random_string = get_random_string()
            self.v = hdf_file.create_dataset(random_string,
                                             (keep.sum(), v.shape[1]),
                                             dtype=np.float64)
        if keep.sum() > 0:
            self.u[:, :] = u[:, keep]
            self.s = s[keep]
            self.v[:, :] = v[keep, :]
            self.inv_s = 1 / s[keep]
        else:
            self.u[:, :] = u[:, 0, None]
            self.s = np.zeros(1)
            self.v[:, :] = v[None, 0, :]
            self.inv_s = np.zeros((1))
        self.shape = np.array((self.u.shape[0], self.v.shape[1]))
        print("SVD shape", self.shape, 'With ', self.s.shape)

    # @classmethod
    # def load_from_disk(cls, path):
    #     pass

    def dot(self, vector, transpose=False):

        if transpose:
            right = self.u[:].T.dot(vector)
            middle = (self.s * right.T).T
            return self.v[:].T.dot(middle) + (self.D * vector.T).T
            # return (self.v[:].T.dot(self.s * self.u[:].T.dot(vector))
            #         + self.D * vector)
        else:
            right = self.v[:].dot(vector)
            middle = (self.s * right.T).T
            return self.u[:].dot(middle) + (self.D * vector.T).T
            # return (self.u[:].dot(self.s * self.v[:].dot(vector))
            #         + self.D * vector)

    def dot_i(self, vector, i):
        return (self.u[i].dot(self.s * self.v[:].dot(vector))
                + self.D[i] * vector[i])

    def inverse_dot(self, vector, transpose=False):
        # if self.D is not invertible then we have to do the slow way
        # unfortunately
        if np.any(np.isclose(np.abs(self.D), 0)):
            if np.all(np.isclose(self.D, 0)):
                if transpose:
                    return self.u[:].dot(self.v[:].dot(vector) * self.inv_s)
                else:
                    return self.v[:].T.dot(self.u[:].T.dot(vector)
                                           * self.inv_s)
            reconst = np.diag(self.D) + (self.u[:] * self.s).dot(self.v[:])
            e_vals = np.linalg.eigh(reconst)[0][::-1]
            rcond = np.where(
                np.isclose(np.cumsum(e_vals) / np.sum(e_vals), 1.)
            )[0]
            if len(rcond) > 0:
                rcond = rcond[0]
            else:
                rcond = len(e_vals) - 1
            rcond = e_vals[rcond] / e_vals[0] * 0.1
            if transpose:
                pinv = np.linalg.pinv(reconst.T, rcond=rcond)
            else:
                pinv = np.linalg.pinv(reconst, rcond=rcond)
            return pinv.dot(vector)
        # Woodbury matrix identity
        if transpose:
            small_mat = np.diag(self.inv_s) + self.u[:].T.dot((self.v[:]
                                                               / self.D).T)
            small_mat = np.linalg.inv(small_mat)
            to_return = vector / self.D
            to_return = self.u[:].T.dot(to_return)
            to_return = small_mat.dot(to_return)
            to_return = self.v[:].T.dot(to_return)
            to_return /= self.D
            return vector / self.D - to_return
        else:
            small_mat = np.diag(self.inv_s) + self.v.dot((self.u[:].T
                                                          / self.D).T)
            small_mat = np.linalg.inv(small_mat)
            to_return = vector / self.D
            to_return = self.v[:].dot(to_return)
            to_return = small_mat.dot(to_return)
            to_return = self.u[:].dot(to_return)
            to_return /= self.D
            return vector / self.D - to_return

    def diag(self):
        to_return = np.einsum('ik,ki->i',
                              self.u[:] * self.s,
                              self.v[:]) + self.D
        return to_return

    def __pow__(self, power):
        assert np.allclose(self.D, 0)
        return SVD(u=self.u, s=self.s**power, v=self.v, D=self.D)

    def get_rank(self, vector=None):
        if self.s.shape[0] > 1:
            return self.s.shape[0]
        if self.s[0] == 0:
            return 0
        return 1
    # @property
    # def T(self):
    #     return SVD(self.v.T, self.s, self.u.T)

    # @property
    # def inverse(self):
    #     return SVD(self.v.T, self.inv_s, self.u.T)


class BlockMatrix(object):
    """A stack of numpy SVD approximated matrices."""
    transposed = False
    inverted = False

    def __init__(self, svds, transpose=False, inverse=False, perm=None,
                 missing=None):
        if missing is None:
            missing = np.array([])
        self.missing = np.copy(missing)
        for svd in svds:
            assert len(svd.shape) == 2

        self.svds = svds
        self.transposed = transpose
        self.inverted = inverse

        self.starts = [0]
        for svd in self.svds:
            self.starts.append(svd.shape[0])
        self.starts = np.cumsum(self.starts)

        self.shape = tuple(map(sum, zip(*map(lambda x: x.shape, self.svds))))
        self.shape = tuple([s + missing.shape[0] for s in self.shape])
        if perm is None:
            self.perm = np.arange(self.shape[1])
        else:
            assert perm.shape[0] == self.shape[1]
            self.perm = perm
        self.inv_perm = np.argsort(self.perm)
        print("block shape", self.shape)

    def dot_i(self, vector, i):
        '''Equivalent to cor_mat.dot(vector)[i]'''
        assert not self.inverted
        assert not self.transposed
        if i in self.missing:
            return 0.
        true_i = self.inv_perm[i]
        block = np.searchsorted(self.starts, true_i, 'right') - 1
        offset = self.starts[block]
        svd = self.svds[block]
        return svd.dot_i(vector[self.perm][offset:(offset + svd.shape[0])],
                         true_i - offset)

    def ridge_inverse_dot(self, vector, regularizer):
        assert not self.inverted
        offset = 0
        parts = []
        reg = np.zeros_like(vector)
        reg[:] = regularizer
        reg = reg[self.perm]
        vector = vector[self.perm]
        for svd in self.svds:
            new_svd = SVD(u=svd.u,
                          s=svd.s,
                          v=svd.v,
                          D=svd.D + reg[offset:(offset+svd.shape[0])])
            product = new_svd.inverse_dot(
                vector[offset:(offset + svd.shape[0])],
                transpose=self.transposed
            )
            parts.append(product)
            offset += svd.shape[0]
        parts.append(np.zeros(self.missing.shape[0]))
        return np.concatenate(parts, axis=0)[self.inv_perm]

    def dot(self, vector):
        """Equivalent to cor_mat.dot(vector).

            It is needed because cor_mat is not able to fully load in memory.
            We instead exploit its block diagonal nature to break down pieces.
        """

        offset = 0
        parts = []
        vector = vector[self.perm]
        for svd in self.svds:
            if self.inverted:
                product = svd.inverse_dot(
                    vector[offset:(offset + svd.shape[0])],
                    transpose=self.transposed
                )
            else:
                product = svd.dot(vector[offset:(offset + svd.shape[0])],
                                  transpose=self.transposed)

            parts.append(product)
            offset += svd.shape[0]

        missing_shape = [self.missing.shape[0]] + list(vector.shape[1:])
        parts.append(np.zeros(missing_shape))
        return np.concatenate(parts, axis=0)[self.inv_perm]

    def __pow__(self, power):
        return BlockMatrix([svd**power for svd in self.svds],
                           transpose=self.transposed,
                           inverse=self.inverted,
                           missing=self.missing)

    @property
    def T(self):
        return BlockMatrix(self.svds,
                           transpose=not self.transposed,
                           inverse=self.inverted,
                           perm=self.perm,
                           missing=self.missing)

    @property
    def inverse(self):
        return BlockMatrix(self.svds,
                           transpose=self.transposed,
                           inverse=not self.inverted,
                           perm=self.perm,
                           missing=self.missing)

    def diag(self):
        '''Diagonal of the matrix.'''

        parts = []

        for svd in self.svds:
            if self.inverted:
                product = svd.inverse.diag()
            else:
                product = svd.diag()

            parts.append(product)
        parts.append(np.zeros(self.missing.shape[0]))
        return np.concatenate(parts, axis=0)[self.inv_perm]

    def get_rank(self):
        '''Returns the rank of the matrix.'''
        rank = 0
        for svd in self.svds:
            rank += svd.get_rank()
        return rank
