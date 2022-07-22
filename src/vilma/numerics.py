"""
Optimized numerical routines for VI schemes
"""
import numpy as np
from numba import njit, prange


EPSILON = 1e-100    # fudge factor to avoid division by zero


@njit('float64[:, :, :](float64[:, :, :], float64[:, :, :], float64)',
      parallel=True, cache=False)
def sum_betas(old_beta, new_beta, step_size):
    """Compute a natural gradient update for the variational means"""
    return step_size * new_beta + (1.-step_size) * old_beta


@njit('float64[:, :](float64[:, :], float64[:, :])', parallel=True, cache=False)
def fast_divide(x, y):
    """Element-wise division of two conformable matrices"""
    return x/y


@njit('float64[:, :](float64[:, :], float64[:, :], float64[:, :],'
      'float64[:, :])', parallel=True, cache=False)
def fast_linked_ests(w, x, y, z):
    """Element-wise w/x + y*z"""
    return w / x - y * z


@njit('float64(float64[:, :], float64[:, :], float64[:, :], float64[:, :],'
      'float64[:, :],'
      'float64[:, :], float64[:], float64[:], float64[:])',
      parallel=True, cache=False)
def fast_likelihood(post_means, post_vars, scaled_mu, scaled_ld_diags,
                    linked_ests,
                    adj_marginal, chi_stat, ld_ranks, error_scaling):
    """Compute the expected log likelihood"""
    likelihood = np.zeros(post_means.shape[0])
    for i in prange(post_means.shape[1]):
        likelihood += (-0.5 * (scaled_ld_diags[:, i] * post_vars[:, i]
                               + linked_ests[:, i] * scaled_mu[:, i])
                       + post_means[:, i] * adj_marginal[:, i])
    likelihood += -0.5 * chi_stat
    return (likelihood/error_scaling
            - 0.5 * ld_ranks * np.log(error_scaling)).sum()


@njit('float64[:, :](float64[:, :, :], float64[:, :])',
      parallel=True, cache=False)
def fast_posterior_mean(vi_mu, vi_delta):
    """Compute the average of vi_mu weighted by vi_delta"""
    to_return = np.zeros((vi_mu.shape[1], vi_mu.shape[2]))
    for i in prange(vi_mu.shape[2]):
        for p in range(vi_mu.shape[1]):
            to_return[p, i] += (vi_mu[:, p, i] * vi_delta[i, :]).sum()
    return to_return


@njit('float64[:, :](float64[:, :], float64[:, :, :], float64[:, :],'
      'float64[:, :, :])', parallel=True, cache=False)
def fast_pmv(mean, vi_mu, vi_delta, temp):
    """Compute the posterior marginal variance"""
    second_moment = fast_posterior_mean(temp + vi_mu**2, vi_delta)
    return second_moment - mean**2


@njit('float64[:, :, :](float64[:, :, :], float64[:, :, :, :])',
      parallel=True, cache=False)
def fast_nat_inner_product_m2(vi_mu, nat_sigma):
    """Compute -2 times 'sqi,spqi->spi' of vi_mu, nat_sigma"""
    to_return = np.empty_like(vi_mu)
    for i in prange(vi_mu.shape[2]):
        for p in range(vi_mu.shape[1]):
            for s in range(vi_mu.shape[0]):
                this_summand = 0.
                for q in range(vi_mu.shape[1]):
                    this_summand += nat_sigma[s, p, q, i] * vi_mu[s, q, i]
                to_return[s, p, i] = -2 * this_summand
    return to_return


@njit('float64[:, :, :](float64[:, :, :], float64[:, :, :, :])',
      parallel=True, cache=False)
def fast_nat_inner_product(vi_mu, nat_sigma):
    """Compute 'sqi,spqi->spi' of vi_mu, nat_sigma"""
    to_return = np.empty_like(vi_mu)
    for i in prange(vi_mu.shape[2]):
        for p in range(vi_mu.shape[1]):
            for s in range(vi_mu.shape[0]):
                this_summand = 0.
                for q in range(vi_mu.shape[1]):
                    this_summand += nat_sigma[s, p, q, i] * vi_mu[s, q, i]
                to_return[s, p, i] = this_summand
    return to_return


@njit('float64(float64[:, :, :], float64[:, :, :, :], float64[:, :])',
      parallel=True, cache=False)
def fast_inner_product_comp(vi_mu, mixture_prec, vi_delta):
    """Half 'kpi,kqi,kpqd,ik->' for vi_mu, vi_mu, mixture_prec, vi_delta"""
    if mixture_prec.shape[-1] != 1:
        raise ValueError('mixture_prec must be 1 dimensional along last mode.')
    to_return = 0.
    for i in prange(vi_delta.shape[0]):
        for k in range(vi_delta.shape[1]):
            this_summand = 0.
            for p in range(vi_mu.shape[1]):
                for q in range(vi_mu.shape[1]):
                    this_summand += (vi_mu[k, p, i]
                                     * vi_mu[k, q, i]
                                     * mixture_prec[k, q, p, 0])
            this_summand *= vi_delta[i, k]
            to_return += this_summand
    return 0.5 * to_return


@njit('float64[:, :](float64[:, :], int64[:], int64)',
      parallel=True, cache=False)
def sum_annotations(deltas, annotations, num_annotations):
    """Compute vector sum of deltas with the same annotations"""
    to_return = np.zeros((num_annotations, deltas.shape[1]))
    for a in range(num_annotations):
        summand = np.zeros(deltas.shape[1], dtype=np.float64)
        for i in prange(annotations.shape[0]):
            if annotations[i] == a:
                summand += deltas[i]
        to_return[a] += summand
    return to_return


@njit('float64(float64[:, :], float64[:, :], int64[:])',
      parallel=True, cache=False)
def fast_delta_kl(vi_delta, hyper_delta, annotations):
    """Compute sum vi_delta[i]*log(vi_delta[i]/hyper_delta[annotations[i]])"""
    log_hyper = np.log(hyper_delta)
    to_return = 0.
    for i in prange(vi_delta.shape[0]):
        to_return += (vi_delta[i] * (np.log(vi_delta[i])
                                     - log_hyper[annotations[i]])).sum()
    return to_return


@njit('float64(float64[:, :], float64[:, :])', parallel=True, cache=False)
def fast_beta_kl(sigma_summary, vi_delta):
    return 0.5 * (sigma_summary * vi_delta).sum()


@njit('float64[:, :](float64[:, :], float64[:], int64[:])',
      parallel=True, cache=False)
def fast_vi_delta_grad(hyper_delta, log_det, annotations):
    """Computes the natural gradient of the VI delta parameter"""
    to_return = np.empty((annotations.shape[0],
                          hyper_delta.shape[1]-1))
    log_hyper = np.log(hyper_delta)
    scaled_sizes = -0.5*(log_det)
    for i in prange(to_return.shape[0]):
        last = (log_hyper[annotations[i], -1]
                + scaled_sizes[-1])
        for k in range(to_return.shape[1]):
            this = (log_hyper[annotations[i], k]
                    + scaled_sizes[k])
            to_return[i, k] = this - last
    return to_return


@njit('float64[:, :](float64[:, :])', parallel=True, cache=False)
def map_to_nat_cat_2D(probs):
    """Compute log(probs[i] / probs[-1]) for all but last element"""
    to_return = np.zeros((probs.shape[0], probs.shape[1] - 1))
    K = probs.shape[1]
    for i in prange(probs.shape[0]):
        last = np.log(probs[i, -1])
        for k in range(K-1):
            to_return[i, k] = np.log(probs[i, k]) - last
    return to_return


@njit('float64[:, :](float64[:, :])', parallel=True, cache=False)
def invert_nat_cat_2D(probs):
    """Convert from log(probs[i] / probs[-1]) to probabilities"""
    to_return = np.empty((probs.shape[0], probs.shape[1] + 1))
    for i in prange(to_return.shape[0]):
        max_p = np.maximum(np.max(probs[i]), 0)
        last_p = np.exp(-max_p)
        denom = last_p
        this_p = np.empty(probs.shape[1])
        for k in range(this_p.shape[0]):
            this = np.exp(probs[i, k] - max_p)
            this_p[k] = this
            denom += this
        for k in range(this_p.shape[0]):
            to_return[i, k] = np.maximum(this_p[k]/denom, EPSILON)
        to_return[i, -1] = np.maximum(last_p / denom, EPSILON)
    return to_return


@njit('float64[:, :](float64[:, :, :], float64[:, :, :], float64[:, :],'
      'float64[:, :])', parallel=True, cache=False)
def fast_invert_nat_vi_delta(new_mu, nat_mu, const_part, nat_vi_delta):
    """Convert natural parameterization of VI delta to standard params"""
    to_invert = np.empty_like(nat_vi_delta)
    for i in prange(to_invert.shape[0]):
        last = const_part[i, -1]
        for j in range(nat_mu.shape[1]):
            last += new_mu[-1, j, i] * nat_mu[-1, j, i]
        for k in range(to_invert.shape[1]):
            this_addenda = const_part[i, k]
            for j in range(nat_mu.shape[1]):
                this_addenda += new_mu[k, j, i] * nat_mu[k, j, i]
            to_invert[i, k] = (0.5*(this_addenda - last)
                               + nat_vi_delta[i, k])
    return invert_nat_cat_2D(to_invert)


@njit('float64[:, :, :, :](float64[:, :, :, :])', parallel=True, cache=False)
def _matrix_invert_4d_numba(matrix):
    """Matrix inversion For 4D arrays where last 2Ds are 2x2 matrices"""
    if matrix.shape[-1] == 0:
        return np.zeros_like(matrix)
    if matrix.shape[-1] == 1:
        return 1. / matrix
    if matrix.shape[-1] == 2:
        matrix = np.transpose(matrix, (3, 2, 1, 0))
        to_return = np.empty_like(matrix)
        det = 1. / (matrix[0, 0, :, :] * matrix[1, 1, :, :]
                    - matrix[0, 1, :, :] * matrix[1, 0, :, :])
        to_return[0, 0, :, :] = matrix[1, 1, :, :] * det
        to_return[1, 1, :, :] = matrix[0, 0, :, :] * det
        to_return[0, 1, :, :] = -matrix[0, 1, :, :] * det
        to_return[1, 0, :, :] = to_return[0, 1, :, :]
        return np.transpose(to_return, (3, 2, 1, 0))
    else:
        raise ValueError('_matrix_invert_4d_numba cannot be used '
                         'on matrices larger than 2x2')


def matrix_invert(matrix):
    """Matrix inversion that is faster for some special cases"""
    if matrix.shape[-1] > 2:
        return np.linalg.inv(matrix)
    if len(matrix.shape) == 4:
        return _matrix_invert_4d_numba(matrix)
    return np.linalg.inv(matrix)


def vi_sigma_inv(matrices):
    """Invert 4D arrays where middle 2 Ds form a matrix"""
    # kpqi->ikpq
    inv = matrix_invert(
        np.transpose(matrices, (3, 0, 1, 2))
    )
    # ikpq->kpqi
    return np.transpose(inv, (1, 2, 3, 0))


@njit('float64[:, :](float64[:, :, :, :])', parallel=True, cache=False)
def _matrix_log_det_4d_numba(matrix):
    """Compute the log determinants of a 4D array of 2x2 matrices"""
    if matrix.shape[-1] == 0:
        return np.zeros((matrix.shape[0], matrix.shape[1]))
    if matrix.shape[-1] == 1:
        return np.log(matrix[:, :, 0, 0])
    if matrix.shape[-1] == 2:
        matrix = np.transpose(matrix, (3, 2, 0, 1))
        det = (matrix[0, 0, :, :] * matrix[1, 1, :, :]
               - matrix[0, 1, :, :] * matrix[1, 0, :, :])
        return np.log(det)
    else:
        raise ValueError('_matrix_log_det_4d_numba cannot be used '
                         'on matrices larger than 2x2')


def matrix_log_det(matrix):
    """Compute matrix log determinant more quickly in some special cases"""
    if matrix.shape[-1] > 2:
        return np.linalg.slogdet(matrix)[1]
    if len(matrix.shape) == 4:
        return _matrix_log_det_4d_numba(matrix)
    return np.linalg.slogdet(matrix)[1]


def vi_sigma_log_det(matrices):
    """Log determinants for 4D arrays where middle 2 Ds form a matrix"""
    # kpqi->ikpq
    log_det = matrix_log_det(
        np.transpose(matrices, (3, 0, 1, 2))
    )
    # ik->ki
    return np.transpose(log_det)
