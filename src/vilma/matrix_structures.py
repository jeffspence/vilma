"""
Utilities for storing block-diagonal symmetric LD matrices

Classes:
    LowRankMatrix: A datastructure for storing and approximating symmetric
        matrices. In particular a matrix is approximated as a low rank
        matrix plus a diagonal matrix.
    BlockDiagonalMatrix: A datastructure for storing symmetric block diagonal
        matrices
"""
import string
import numpy as np


def _svd_threshold(matrix, ld_thresh):
    """Perform SVD on X and truncate to keep large eigenvalues"""
    s_vals, vecs = np.linalg.eigh(matrix)
    big_sing_vals = np.where(s_vals >= 1 - np.sqrt(ld_thresh))[0]
    if len(big_sing_vals > 0):
        k = big_sing_vals
    else:
        return (np.ones((matrix.shape[0], 1)),
                np.zeros(1),
                np.ones((1, matrix.shape[1])))
    u_mat = np.copy(vecs[:, k])
    s_vec = np.copy(s_vals[k])
    v_mat = np.copy(u_mat.T)
    return u_mat, s_vec, v_mat


def _get_random_string():
    """Generate a random length 100 str"""
    return ''.join(np.random.choice(
        list(string.ascii_letters + string.digits), size=100)
    )


class LowRankMatrix():
    """
    A low rank plus diagonal representation of a symmetric matrix

    Efficiently stores and performs matrix operations (inverse, dot product,
    etc...) for symmetric matrices that can be approximated as a low rank
    matrix plus a diagonal matrix. The matrix is P x P dimensional, and
    the low rank approximation is rank K.

    Fields:
        u: left singular vectors of the low-rank component, stored as
            a P x K dimension numpy array.
        s: singular values of the low-rank component stored as a K
            dimensional numpy array.
        inv_s: pseudo inverse of the singular values. For nonzero
            values of s, inv_s is 1 / s.  For zero values of s,
            inv_s is also zero.
        v: right singular vectors of the low-rank component stored as
            a K x P dimensional numpy array.
        D: The diagonal component of the approximation stored as a P
            dimensional numpy array.
        shape: The shape of the matrix as a tuple

    Methods:
        dot: M.dot(v) computes the dot product M @ v
        dot_i: M.dot_i(v, i) computes element i of the dot product of M and v.
            (M @ v)[i]
        inverse_dot: Computes the dot product of the inverse of the matrix with
            a vector.  M.inverse_dot(i) is inverse(M) @ v
        diag: Computes the diagonal entries of the matrix and returns them as a
            numpy array.
        __pow__: Computes the matrix power of the matrix.
        get_rank: Returns the rank of the matrix
    """
    def __init__(self, X=None, t=1.0, u=None, s=None, v=None, D=None,
                 hdf_file=None):
        """
        Build a LowRankMatrix from a symmetric matrix or its SVD decomposition

        The user needs to supply either a dense matrix or a low rank plus
        diagonal approximation. If the user provides a dense matrix then it
        will be approximated using a low rank approximation and the threshold
        t.

        Args:
            X: a dense symmetric matrix
            t: A threshold for truncating the SVD representation of the matrix.
                We keep all eigenvectors with eigenvalues greater than
                1 - sqrt(t). This guarantees that SNPs with an r^2 of less than
                t will be linearly independent in the low rank representation.
            u: The left singular vectors as a P x K dimensional numpy array
            s: The singular values as a K dimensional numpy array.
            v: The right sinular vectors as a K x P dimensional numpy array
            D: The diagonal component of the low rank plus diagonal matrix
            hdf_file: A file in which to store this matrix if not stored in
                memory
        """
        if X is not None:
            if ((u is not None)
                    or (s is not None)
                    or (v is not None)
                    or (D is not None)):
                raise ValueError('Cannot provide both a matrix and an '
                                 'SVD decomposition')
            if not np.allclose(X, X.T):
                raise ValueError('Provided matrix is not symmetric')
            u, s, v = _svd_threshold(X, t)
            D = np.zeros(X.shape[0])
        else:
            if ((u is None)
                    or (s is None)
                    or (v is None)
                    or (D is None)):
                raise ValueError('Need to provide either a matrix or '
                                 'an SVD decomposition')
        self.D = np.copy(D)
        keep = np.abs(s) > (1e-12 * np.max(np.abs(s)))
        if hdf_file is None:
            self.u = np.zeros((u.shape[0], keep.sum()))
            self.v = np.zeros((keep.sum(), v.shape[1]))
        else:
            random_string = _get_random_string()
            while random_string in hdf_file.keys():
                random_string = _get_random_string()
            self.u = hdf_file.create_dataset(random_string,
                                             (u.shape[0], keep.sum()),
                                             dtype=np.float64)
            random_string = _get_random_string()
            while random_string in hdf_file.keys():
                random_string = _get_random_string()
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
        self.shape = (self.u.shape[0], self.v.shape[1])

    def dot(self, vector):
        """Compute Matrix @ vector"""
        right = self.v[:].dot(vector)
        middle = (self.s * right.T).T
        return self.u[:].dot(middle) + (self.D * vector.T).T

    def dot_i(self, vector, i):
        """Compute (Matrix @ vector)[i]"""
        return (self.u[i].dot(self.s * self.v[:].dot(vector))
                + self.D[i] * vector[i])

    def inverse_dot(self, vector):
        """Compute PseudoInverse(Matrix) @ vector"""
        # If D is not invertible we need to be careful
        if np.any(np.isclose(np.abs(self.D), 0)):
            # If D is fully zero, then we can pseudoinverse
            if np.all(np.isclose(self.D, 0)):
                return self.v[:].T.dot(self.u[:].T.dot(vector)
                                       * self.inv_s)

            # Otherwise, we will reconstruct the full matrix and pseduoinvert
            reconst = np.diag(self.D) + (self.u[:] * self.s).dot(self.v[:])

            # Get an estimate of the eigenvalue at which we've got the full
            # matrix
            e_vals = np.linalg.eigh(reconst)[0][::-1]
            rcond = np.where(
                np.isclose(np.cumsum(e_vals) / np.sum(e_vals), 1.)
            )[0]
            if len(rcond) > 0:
                rcond = rcond[0]
            else:
                rcond = len(e_vals) - 1
            rcond = e_vals[rcond] / e_vals[0] * 0.1

            # Pseudoinvert using np,linalg.pinv
            pinv = np.linalg.pinv(reconst, rcond=rcond)
            return pinv.dot(vector)

        # If D is invertible we can use the Woodbury matrix identity
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
        """Return the diagonal of this matrix"""
        to_return = np.einsum('ik,ki->i',
                              self.u[:] * self.s,
                              self.v[:]) + self.D
        return to_return

    def __pow__(self, power):
        """Return the matrix power of this matrix as a LowRankMatrix"""
        if not np.allclose(self.D, 0):
            raise NotImplementedError('Matrix powers where the diagonal '
                                      'approximation is not zero have '
                                      'not yet been implemented.')
        return LowRankMatrix(u=self.u, s=self.s**power, v=self.v, D=self.D)

    def get_rank(self):
        """Return the rank of this matrix"""

        # If D is zero, then we've just got the low rank component
        if np.allclose(self.D, 0):
            if self.s.shape[0] > 1:
                return self.s.shape[0]
            if self.s[0] == 0:
                return 0
            return 1

        # If D is all positive, then full rank
        if np.all(self.D > 0):
            return self.D.shape[0]

        # If D is not all zero or all positive, we have to reconstruct the
        # matrix
        mat = np.diag(self.D) + np.einsum('ik,k,kj->ij',
                                          self.u,
                                          self.s,
                                          self.v)
        return np.linalg.matrix_rank(mat, hermitian=True)


class BlockDiagonalMatrix():
    """
    A symmetric block diagonal matrix, where blocks are low rank

    Fields:
        matrices: a list of the component matrices along the diagonal
        starts: a numpy array of length equal to the number of component
            matrices plus 1.  Entry i is the index at which block matrix i
            begins and the last entry is the total number of indices.
        perm: To ensure the matrix is block diagonal, it is sometimes
            necessary to reorder the indices. For convenience we would like
            to store the matrix as a block diagonal, but then perform
            operations (e.g., dot products) as if the matrix had not been
            reordered. perm maps the original indices to the reordered indices
            so that M @ v[perm] is a reordered version of the original matrix
            dotted with v. In particular perm and inv_perm work together so
            that (M @ v[perm])[inv_perm] returns the correct matrix vector
            in the original indexing. All of this is implemented in the methods
            so that this is hidden from the user.
        inv_perm: See perm -- inv_perm is the inverse permutation of perm so
            that v[perm][inv_perm] = v
        shape: The dimensions of the matrix
        missing: The indices which do not correspond to any component matrix.
            These are implicitly treated as additional columns and rows of
            zeros.

    Methods:
        dot: M.dot(v) computes the dot product M @ v
        dot_i: M.dot(v, i) computes element i of the dot product of M and v.
            (M @ v)[i]
        ridge_inverse_dot: M.ridge_inverse_dot(v, r) computes the regularized
            inverse of the matrix dotted with v. If r is a scalar, then it is
            treated as r * np.ones(len(v)). In particular this method computes
            inverse(M + diag(r)) @ v
        __pow__: Returns the matrix power of the matrix
        inverse: Returns the pseudoinverse of the matrix
        diag: Returns the diagonal of the matrix
        get_rank: Returns the rank of the matrix

    """
    def __init__(self, matrices, inverse=False, perm=None,
                 missing=None):
        """
        Build a BlockDiagonalMatrix from a list of symmetric component matrices

        Args:
            matrices: a list of LowRankMatrices representing the component
                matrices
            inverse: a flag to denote whether this matrix is inverted or not.
                Inverting this datatype is performed lazily -- if a matrix is
                inverted, each time a function is called, the component
                matrices are (potentially implicitly) inverted.
            perm: See the definition of the class -- but perm is used so that
                matrices which are not block diagonal, but can be permuted to
                be diagonal can still be used with this datastructure. In
                particular, if M is not block diagonal, but
                M[np._ix(perm, perm)] is block diagonal with the component
                matrices of ther permuted matrix stored in the list `matrices`
                then this class can still be used.
            missing: Instead of including many component matrices that are
                zero, those indices can be specified in the array of indices
                `missing`. Each index in missing is treated as corresponding to
                a 1 x 1 component matrix that is just 0.
        """
        if missing is None:
            missing = np.array([])
        self.missing = np.copy(missing)
        for matrix in matrices:
            if not isinstance(matrix, LowRankMatrix):
                raise ValueError('Component matrices must be of type '
                                 'LowRankMatrix')
        self.matrices = matrices
        self._inverted = inverse

        self.starts = [0]
        for matrix in self.matrices:
            self.starts.append(matrix.shape[0])
        self.starts = np.cumsum(self.starts)

        self.shape = tuple(
            map(sum, zip(*map(lambda x: x.shape, self.matrices)))
        )
        self.shape = tuple([s + missing.shape[0] for s in self.shape])
        if perm is None:
            self.perm = np.arange(self.shape[1])
        else:
            if perm.shape[0] != self.shape[1]:
                raise ValueError('perm must be a vector conformal '
                                 'to the non-missing parts of the matrix.')
            self.perm = np.copy(perm)
        self.inv_perm = np.argsort(self.perm)
        if not np.allclose(self.perm[self.inv_perm],
                           np.arange(self.perm.shape[0])):
            raise ValueError('perm and missing should together contain all '
                             'of the indices. Some are missing.')

    def dot_i(self, vector, i):
        """Compute (Matrix @ vector)[i]"""
        if self._inverted:
            raise NotImplementedError('dot_i with inverted matrices '
                                      'has not been implemented yet.')
        if i in self.missing:
            return 0.
        true_i = self.inv_perm[i]
        block = np.searchsorted(self.starts, true_i, 'right') - 1
        offset = self.starts[block]
        matrix = self.matrices[block]
        return matrix.dot_i(
            vector[self.perm][offset:(offset + matrix.shape[0])],
            true_i - offset,
        )

    def ridge_inverse_dot(self, vector, regularizer):
        """
        Compute Inverse(Matrix + diag(regularizer)) @ vector

        Args:
            vector: A numpy array of length matching the dimension of the block
                matrix.
            regularizer: Either a numpy array of length matching the dimension
                of the block matrix or a scalar.  If a scalar, then will be
                treated as a vector of ones times that scalar.  This vector is
                added to the diagonal of the matrix before inverting.

        returns:
            A numpy array containing the inverse of the matrix plus a diagonal
            matrix containing the regularizer dotted with vector.
        """
        if self._inverted:
            raise NotImplementedError('ridge_inverse_dot with inverted '
                                      'matrices has not been implemented '
                                      'yet.')
        offset = 0
        parts = []
        reg = np.zeros_like(vector)
        reg[:] = regularizer
        reg = reg[self.perm]
        vector = vector[self.perm]
        for matrix in self.matrices:
            regularized_diag = matrix.D + reg[offset:(offset+matrix.shape[0])]
            new_svd = LowRankMatrix(u=matrix.u,
                                    s=matrix.s,
                                    v=matrix.v,
                                    D=regularized_diag)
            product = new_svd.inverse_dot(
                vector[offset:(offset + matrix.shape[0])],
            )
            parts.append(product)
            offset += matrix.shape[0]
        parts.append(np.zeros(self.missing.shape[0]))
        return np.concatenate(parts, axis=0)[self.inv_perm]

    def dot(self, vector):
        """Compute Matrix @ vector"""

        offset = 0
        parts = []
        vector = vector[self.perm]
        for matrix in self.matrices:
            if self._inverted:
                product = matrix.inverse_dot(
                    vector[offset:(offset + matrix.shape[0])],
                )
            else:
                product = matrix.dot(vector[offset:(offset + matrix.shape[0])])

            parts.append(product)
            offset += matrix.shape[0]

        missing_shape = [self.missing.shape[0]] + list(vector.shape[1:])
        parts.append(np.zeros(missing_shape))
        return np.concatenate(parts, axis=0)[self.inv_perm]

    def __pow__(self, power):
        """Compute the matrix power of this matrix"""
        return BlockDiagonalMatrix([x**power for x in self.matrices],
                                   inverse=self._inverted,
                                   missing=self.missing)

    @property
    def inverse(self):
        """Pseudoinvert this matrix"""
        return BlockDiagonalMatrix(self.matrices,
                                   inverse=not self._inverted,
                                   perm=self.perm,
                                   missing=self.missing)

    def diag(self):
        '''Diagonal of the matrix.'''
        if self._inverted:
            raise NotImplementedError('Getting the diagonal of an '
                                      'inverted matrix has not been '
                                      'implemented yet.')

        parts = []
        for matrix in self.matrices:
            product = matrix.diag()
            parts.append(product)

        parts.append(np.zeros(self.missing.shape[0]))

        return np.concatenate(parts, axis=0)[self.inv_perm]

    def get_rank(self):
        '''Returns the rank of the matrix.'''
        rank = 0
        for matrix in self.matrices:
            rank += matrix.get_rank()
        return rank
