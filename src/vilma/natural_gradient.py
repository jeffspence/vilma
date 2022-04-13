from __future__ import division

import numpy as np
from numba import njit, prange
import logging

L_MAX = 1e12
REL_TOL = 1e-6
ABS_TOL = 1e-6
ELBO_TOL = 0.1
EM_TOL = 10
MAX_GRAD = 100
EPSILON = 1e-100
ELBO_MOMENTUM = 0.5
NUM_BIG_PASSES = 10
logging.basicConfig()

# TODO: document and comment


@njit('float64[:, :, :](float64[:, :, :], float64[:, :, :], float64)',
      parallel=True)
def _sum_betas(old_beta, new_beta, step_size):
    return step_size * new_beta + (1.-step_size) * old_beta


@njit('float64[:, :](float64[:, :], float64[:, :])', parallel=True)
def _fast_divide(x, y):
    return x/y


@njit('float64[:, :](float64[:, :], float64[:, :], float64[:, :],'
      'float64[:, :])', parallel=True)
def _fast_linked_ests(linked_ests, std_errs, post_mean, scaled_ld_diags):
    return linked_ests / std_errs - scaled_ld_diags * post_mean


@njit('float64(float64[:, :], float64[:, :], float64[:, :], float64[:, :],'
      'float64[:, :],'
      'float64[:, :], float64[:], float64[:], float64[:])', parallel=True)
def _fast_likelihood(post_means, post_vars, scaled_mu, scaled_ld_diags,
                     linked_ests,
                     adj_marginal, chi_stat, ld_ranks, error_scaling):
    likelihood = np.zeros(post_means.shape[0])
    for i in prange(post_means.shape[1]):
        likelihood += (-0.5 * (scaled_ld_diags[:, i] * post_vars[:, i]
                               + linked_ests[:, i] * scaled_mu[:, i])
                       + post_means[:, i] * adj_marginal[:, i])
    likelihood += -0.5 * chi_stat
    return (likelihood/error_scaling
            - 0.5 * ld_ranks * np.log(error_scaling)).sum()


@njit('float64[:, :](float64[:, :], float64[:, :], float64)', parallel=True)
def _sum_deltas(old_delta, new_delta, step_size):
    return step_size * new_delta + (1.-step_size) * old_delta


@njit('float64[:, :](float64[:, :, :], float64[:, :])', parallel=True)
def _fast_posterior_mean(vi_mu, vi_delta):
    to_return = np.zeros((vi_mu.shape[1], vi_mu.shape[2]))
    for i in prange(vi_mu.shape[2]):
        for p in range(vi_mu.shape[1]):
            to_return[p, i] += (vi_mu[:, p, i] * vi_delta[i, :]).sum()
    return to_return


@njit('float64[:, :](float64[:, :], float64[:, :, :], float64[:, :],'
      'float64[:, :, :])', parallel=True)
def _fast_pmv(mean, vi_mu, vi_delta, temp):
    second_moment = _fast_posterior_mean(temp + vi_mu**2, vi_delta)
    return second_moment - mean**2


@njit('float64[:, :, :](float64[:, :, :], float64[:, :, :, :])', parallel=True)
def _fast_nat_inner_product_m2(vi_mu, nat_sigma):
    to_return = np.empty_like(vi_mu)
    for i in prange(vi_mu.shape[2]):
        for p in range(vi_mu.shape[1]):
            for s in range(vi_mu.shape[0]):
                this_summand = 0.
                for q in range(vi_mu.shape[1]):
                    this_summand += nat_sigma[s, p, q, i] * vi_mu[s, q, i]
                to_return[s, p, i] = -2 * this_summand
    return to_return


@njit('float64[:, :, :](float64[:, :, :], float64[:, :, :, :])', parallel=True)
def _fast_nat_inner_product(vi_mu, nat_sigma):
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
      parallel=True)
def _fast_inner_product_comp(vi_mu, mixture_prec, vi_delta):
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


@njit('float64[:, :](float64[:, :], int64[:], int64)', parallel=True)
def _sum_annotations(deltas, annotations, num_annotations):
    to_return = np.zeros((num_annotations, deltas.shape[1]))
    for a in range(num_annotations):
        summand = np.zeros(deltas.shape[1], dtype=np.float64)
        for i in prange(annotations.shape[0]):
            if annotations[i] == a:
                summand += deltas[i]
        to_return[a] += summand
    return to_return


@njit('float64(float64[:, :], float64[:, :], int64[:])', parallel=True)
def _fast_delta_kl(vi_delta, hyper_delta, annotations):
    log_hyper = np.log(hyper_delta)
    to_return = 0.
    for i in prange(vi_delta.shape[0]):
        to_return += (vi_delta[i] * (np.log(vi_delta[i])
                                     - log_hyper[annotations[i]])).sum()
    return to_return


@njit('float64(float64[:, :], float64[:, :])', parallel=True)
def _fast_beta_kl(sigma_summary, vi_delta):
    return 0.5 * (sigma_summary * vi_delta).sum()


@njit('float64[:, :](float64[:, :], float64[:], int64[:])',
      parallel=True)
def _fast_vi_delta_grad(hyper_delta, log_det, annotations):
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


@njit('float64[:, :](float64[:, :])', parallel=True)
def _map_to_nat_cat_2D(probs):
    to_return = np.zeros((probs.shape[0], probs.shape[1] - 1))
    K = probs.shape[1]
    for i in prange(probs.shape[0]):
        last = np.log(probs[i, -1])
        for k in range(K-1):
            to_return[i, k] = np.log(probs[i, k]) - last
    return to_return


@njit('float64[:, :](float64[:, :])', parallel=True)
def _get_const_part(vi_log_det):
    return np.copy(vi_log_det.T)


@njit('float64[:, :](float64[:, :, :], float64[:, :, :],'
      'float64[:, :], float64[:, :])', parallel=True)
def _fast_map_to_nat_vi_delta(vi_mu, old_nat_mu, const_part, vi_delta):
    old_nat_vi_delta = _map_to_nat_cat_2D(vi_delta)
    K = vi_mu.shape[0]
    for i in prange(old_nat_vi_delta.shape[0]):
        last = const_part[i, K-1]
        for j in range(old_nat_mu.shape[1]):
            last += vi_mu[K-1, j, i] * old_nat_mu[K-1, j, i]
        for k in range(K-1):
            this_addenda = const_part[i, k]
            for j in range(old_nat_mu.shape[1]):
                this_addenda += vi_mu[k, j, i] * old_nat_mu[k, j, i]
            old_nat_vi_delta[i, k] += -0.5 * (this_addenda - last)
    return old_nat_vi_delta


@njit('float64[:, :](float64[:, :])', parallel=True)
def _invert_nat_cat_2D(probs):
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
      'float64[:, :])', parallel=True)
def _fast_invert_nat_vi_delta(new_mu, nat_mu, const_part, nat_vi_delta):
    to_invert = np.empty_like(nat_vi_delta)
    # K = to_invert.shape[1]+1
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
    return _invert_nat_cat_2D(to_invert)


@njit('float64[:, :, :, :](float64[:, :, :, :])', parallel=True)
def _matrix_invert_4d_numba(matrix):
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
        return np.full_like(matrix, np.nan)


@njit('float64[:, :, :](float64[:, :, :])', parallel=True)
def _matrix_invert_3d_numba(matrix):
    if matrix.shape[-1] == 0:
        return np.zeros_like(matrix)
    if matrix.shape[-1] == 1:
        return 1. / matrix
    if matrix.shape[-1] == 2:
        matrix = np.transpose(matrix, (2, 1, 0))
        to_return = np.empty_like(matrix)
        det = 1. / (matrix[0, 0, :] * matrix[1, 1, :]
                    - matrix[0, 1, :] * matrix[1, 0, :])
        to_return[0, 0, :] = matrix[1, 1, :] * det
        to_return[1, 1, :] = matrix[0, 0, :] * det
        to_return[0, 1, :] = -matrix[0, 1, :] * det
        to_return[1, 0, :] = to_return[0, 1, :]
        return np.transpose(to_return, (2, 1, 0))
    else:
        return np.full_like(matrix, np.nan)


def _matrix_invert(matrix):
    if matrix.shape[-1] > 2:
        return np.linalg.inv(matrix)
    if len(matrix.shape) == 4:
        return _matrix_invert_4d_numba(matrix)
    if len(matrix.shape) == 3:
        return _matrix_invert_3d_numba(matrix)
    return np.linalg.inv(matrix)


'''
def _matrix_invert(matrix):
    if matrix.shape[-1] == 0:
        return np.zeros_like(matrix)
    if matrix.shape[-1] == 1:
        return 1. / matrix
    if matrix.shape[-1] == 2:
        to_return = np.empty_like(matrix)
        det = 1. / (matrix[..., 0, 0] * matrix[..., 1, 1]
                    - matrix[..., 0, 1] * matrix[..., 1, 0])
        to_return[..., 0, 0] = matrix[..., 1, 1] * det
        to_return[..., 1, 1] = matrix[..., 0, 0] * det
        to_return[..., 0, 1] = -matrix[..., 0, 1] * det
        to_return[..., 1, 0] = to_return[..., 0, 1]
        return to_return
    return np.linalg.inv(matrix)
'''


def _vi_sigma_inv(matrices):
    # kpqi->ikpq
    inv = _matrix_invert(
        np.transpose(matrices, (3, 0, 1, 2))
    )
    # ikpq->kpqi
    return np.transpose(inv, (1, 2, 3, 0))


@njit('float64[:, :](float64[:, :, :, :])', parallel=True)
def _matrix_log_det_4d_numba(matrix):
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
        return np.full((matrix.shape[0], matrix.shape[1]), np.nan)


@njit('float64[:](float64[:, :, :])', parallel=True)
def _matrix_log_det_3d_numba(matrix):
    if matrix.shape[-1] == 0:
        return np.zeros(matrix.shape[0])
    if matrix.shape[-1] == 1:
        return np.log(matrix[:, 0, 0])
    if matrix.shape[-1] == 2:
        matrix = np.transpose(matrix, (2, 1, 0))
        det = (matrix[0, 0, :] * matrix[1, 1, :]
               - matrix[0, 1, :] * matrix[1, 0, :])
        return np.log(det)
    else:
        return np.full(matrix.shape[0], np.nan)


def _matrix_log_det(matrix):
    if matrix.shape[-1] > 2:
        return np.linalg.slogdet(matrix)[1]
    if len(matrix.shape) == 4:
        return _matrix_log_det_4d_numba(matrix)
    if len(matrix.shape) == 3:
        return _matrix_log_det_3d_numba(matrix)


# TODO: numba this
def _vi_sigma_log_det(matrices):
    # kpqi->ikpq
    log_det = _matrix_log_det(
        np.transpose(matrices, (3, 0, 1, 2))
    )
    # ik->ki
    return np.transpose(log_det)


class Nat_grad_optimizer(object):

    def __init__(self,
                 marginal_effects=None,
                 std_errs=None,
                 ld_mats=None,
                 annotations=None,
                 mixture_covs=None,
                 checkpoint=True,
                 checkpoint_freq=5,
                 scaled=False,
                 scale_se=False,
                 output='vilma_output',
                 gwas_N=None,
                 init_hg=None,
                 num_its=None):

        assert init_hg is not None
        assert gwas_N is not None
        assert marginal_effects is not None
        assert std_errs is not None
        assert ld_mats is not None
        assert annotations is not None
        assert mixture_covs is not None
        assert num_its is not None
        assert np.all(np.isfinite(marginal_effects))
        assert np.all(np.isfinite(std_errs))

        self.scaled = scaled    # whether to do Z-scores or not

        self.scale_se = scale_se    # whether to learn an SE scaling
        self.error_scaling = np.ones(marginal_effects.shape[0])

        self.checkpoint = checkpoint
        checkpoint_path = '%s-checkpoint' % output

        self.num_pops = marginal_effects.shape[0]
        self.num_loci = marginal_effects.shape[1]
        assert marginal_effects.shape == (self.num_pops, self.num_loci)
        assert len(ld_mats) == self.num_pops
        try:
            self.ld_diags = np.concatenate(
                [ld.diag().reshape((1, -1)) for ld in ld_mats],
                axis=0
            )
        except AttributeError:
            diags = []
            for ld in ld_mats:
                idxr = np.arange(len(ld))
                diags.append(ld[idxr, idxr].reshape((1, -1)))
            self.ld_diags = np.concatenate(diags, axis=0)
        for p, ldm in enumerate(ld_mats):
            assert ldm.shape == (self.num_loci, self.num_loci)
        assert np.allclose(annotations.sum(axis=1), 1)
        self.num_annotations = annotations.shape[1]
        assert annotations.shape[0] == self.num_loci

        # Store things
        self.marginal_effects = np.copy(marginal_effects)
        if self.scaled:
            self.marginal_effects = self.marginal_effects / (std_errs +
                                                             EPSILON)
        # self.marginal_effects.flags.writeable = False
        if self.scaled:
            self.std_errs = np.ones_like(std_errs)
            self.scalings = (std_errs + EPSILON)
        else:
            self.std_errs = np.copy(std_errs)
            self.scalings = np.ones_like(std_errs)
        self.scaled_ld_diags = self.std_errs**-2 * self.ld_diags
        # self.std_errs.flags.writeable = False
        self.ld_mats = ld_mats
        # self.ld_diags.flags.writeable = False
        self.annotations = np.copy(np.where(annotations)[1])
        # self.annotations.flags.writeable = False
        self.annotation_counts = annotations.sum(axis=0)
        # self.annotation_counts.flags.writeable = False
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path
        self.init_hg = init_hg
        self.gwas_N = gwas_N
        self.num_its = num_its

        # store adjusted marginal effects S^{-1}XX^{-1}S^{-1}\hat{beta}
        # Note that if X is not full rank, then this is not the same as
        # S^{-2} \hat{beta}
        self.adj_marginal_effects = np.zeros_like(self.marginal_effects)
        self.chi_stat = np.zeros(self.num_pops)
        self.mle = np.zeros((self.num_pops, self.num_loci))
        self.ld_ranks = np.zeros(self.num_pops)
        self.einsum_paths = {}

        # Compute MLE of betas
        self.inverse_betas = np.zeros_like(self.marginal_effects)
        for p in range(self.num_pops):
            try:
                # this is basically an approximation to
                # LDpred-inf with a relatively arbitrarily
                # chosen regularization penalty.
                # inv_z_scores = ld_mats[p].ridge_inverse_dot(
                #     self.marginal_effects[p] / self.std_errs[p],
                #     self.marginal_effects.shape[1] * 10 / gwas_N[p]
                # )
                # self.inverse_betas[p, :] = inv_z_scores * self.std_errs[p]
                z_scores = self.marginal_effects[p] / self.std_errs[p]
                self.mle[p] = ld_mats[p].inverse.dot(z_scores)
                self.chi_stat[p] = z_scores.dot(self.mle[p])
                this_adj_marg = np.copy(self.mle[p])
                this_adj_marg = ld_mats[p].dot(this_adj_marg)
                this_adj_marg = this_adj_marg / self.std_errs[p]
                self.adj_marginal_effects[p, :] = this_adj_marg
                self.ld_ranks[p] = ld_mats[p].get_rank()

                # this is the LDpred model
                # inv_z_scores = ld_mats[p].ridge_inverse_dot(
                #     this_adj_marg * self.std_errs[p],
                #     self.marginal_effects.shape[1] / gwas_N[p] / init_hg[p]
                # )
                # self.inverse_betas[p, :] = inv_z_scores * self.std_errs[p]

                # this is the effect size is ind. of freq model
                prior = (2 * gwas_N[p] * init_hg[p]
                         / (self.std_errs[p, :]**-2).sum())
                inv_z_scores = ld_mats[p].ridge_inverse_dot(
                    this_adj_marg * self.std_errs[p],
                    self.std_errs[p, :]**2 / prior
                )
                self.inverse_betas[p, :] = inv_z_scores * self.std_errs[p]

            except AttributeError:
                logging.warning('Computing inverse of LD matrix via direct '
                                'methods because the LD matrix for population '
                                '%d does not have a .inverse attribute...'
                                'Could be very slow.', p)
                scale = self.marginal_effects.shape[1] * 10 / 100000.
                inv_z_scores = np.linalg.solve(
                    ld_mats[p] + np.eye(ld_mats[p].shape[0]) * scale,
                    self.marginal_effects[p] / self.std_errs[p]
                )
                self.inverse_betas[p, :] = inv_z_scores * self.std_errs[p]
                self.adj_marginal_effects[p, :] = (
                    self.marginal_effects[p, :] / (self.std_errs[p, :] ** 2)
                )
                self.ld_ranks[p] = ld_mats[p].shape[0]
                z_scores = self.marginal_effects[p, :] / self.std_errs[p, :]
                self.chi_stat[p] = z_scores.dot(np.lianlg.solve(
                    ld_mats[p], z_scores
                ))
        assert np.allclose(
            self.adj_marginal_effects[np.isclose(self.ld_diags, 0)],
            0
        )
        # self.inverse_betas.flags.writeable = False
        # self.adj_marginal_effects.flags.writeable = False
        # self.chi_stat.flags.writeable = False
        # self.ld_ranks.flags.writeable = False

    def _fast_einsum(self, *args, key=None):
        if key is None:
            return np.einsum(*args)
        if key not in self.einsum_paths:
            e_path = np.einsum_path(*args, optimize='optimal')
            logging.info('Einsum contraction path optimization results for '
                         '%s :\n %s',
                         key, e_path[1])
            self.einsum_paths[key] = e_path[0]
        return np.einsum(*args, optimize=self.einsum_paths[key])

    def optimize(self):
        params = self._initialize()
        converged = False
        elbo = self.elbo(params)
        running_elbo_delta = None
        num_its = 0
        L = np.ones(5)
        checkpoint_params = params
        post_mean = self._real_posterior_mean(*params)
        ckp_post_mean = self._real_posterior_mean(*checkpoint_params)
        while num_its < self.num_its and not converged:
            if num_its % self.checkpoint_freq == 0 and self.checkpoint:
                checkpoint_params = params
                ckp_post_mean = self._real_posterior_mean(*checkpoint_params)
                fname = '{}.{}'.format(self.checkpoint_path, num_its)
                dump_dict = dict(zip(self.param_names, params))
                dump_dict['marginal'] = self.marginal_effects,
                dump_dict['stderr'] = self.std_errs
                dump_dict['annotations'] = self.annotations
                np.savez(fname, **dump_dict)
            line_search_rate = 2.
            new_params, L, elbo, running_elbo_delta = self._optimize_step(
                params, L=L, curr_elbo=elbo, line_search_rate=line_search_rate,
                running_elbo_delta=running_elbo_delta
            )

            new_post_mean = self._real_posterior_mean(*new_params)

            converged = np.allclose(new_post_mean, post_mean, atol=ABS_TOL,
                                    rtol=REL_TOL)
            converged = converged or np.isclose(running_elbo_delta, 0,
                                                atol=ELBO_TOL,
                                                rtol=0)
            if num_its < 10:
                converged = False

            logging.info('Completed iteration %d', num_its + 1)
            logging.info('Maximum posterior mean beta: %e',
                         np.max(np.abs(new_post_mean)))
            logging.info('SE scaling is: %r', self.error_scaling)
            logging.info(
                'Max relative difference is: %e',
                np.max(np.abs((new_post_mean - post_mean) / post_mean))
            )
            logging.info(
                'Max absolute difference is: %e',
                np.max(np.abs(new_post_mean - post_mean))
            )
            logging.info(
                'Mean absolute difference is: %e',
                np.mean(np.abs(new_post_mean - post_mean))
            )
            logging.info(
                'RMSE difference is: %e',
                np.sqrt(np.mean((new_post_mean - post_mean)**2))
            )
            logging.info(
                'Max relative difference (checkpoint iterations) is: %e',
                np.max(np.abs((new_post_mean - ckp_post_mean)
                              / ckp_post_mean))
            )
            logging.info(
                'Max absolute difference (checkpoint iterations) is: %e',
                np.max(np.abs(new_post_mean - ckp_post_mean))
            )
            logging.info(
                'Mean absolute difference (checkpoint iterations) is: %e',
                np.mean(np.abs(new_post_mean - ckp_post_mean))
            )
            logging.info(
                'RMSE difference (checkpoint iterations) is: %e',
                np.sqrt(np.mean((new_post_mean - ckp_post_mean)**2))
            )

            post_mean = new_post_mean
            num_its += 1

            params = tuple(new_params)

        if num_its == self.num_its:
            logging.warning('Failed to converge')
        logging.info('Optimization ran for %d iterations', num_its)
        return params

    def _optimize_step(self, params, L, curr_elbo,
                       line_search_rate=1.25, running_elbo_delta=None):
        logging.info('Current ELBO = %f and L = %f,%f,%f,%f,%f',
                     curr_elbo, L[0], L[1], L[2], L[3], L[4])
        (new_params,
         L_new,
         elbo_change) = self._nat_grad_step(params, L, line_search_rate,
                                            running_elbo_delta)
        elbo = curr_elbo + elbo_change
        # assert elbo_change > -1e-5
        if running_elbo_delta is None:
            running_elbo_delta = elbo_change
        running_elbo_delta *= ELBO_MOMENTUM
        running_elbo_delta += (1 - ELBO_MOMENTUM) * elbo_change
        return new_params, L_new, elbo, running_elbo_delta

    def elbo(self, params):
        elbo = self._log_likelihood(params)
        elbo -= self._beta_KL(*params)
        elbo -= self._annotation_KL(*params)
        return elbo

    def _nat_grad_step(self, params, L, line_search_rate,
                       running_elbo_delta=None):
        updates = [self._update_beta, self._update_hyper_delta,
                   self._update_annotation]
        conv_tol = (float('inf') if running_elbo_delta is None
                    else 0.1 * running_elbo_delta)
        new_elbo_delta = 0
        for idx, update in enumerate(updates):
            orig_obj = None
            while True:
                L[idx] = max([1., L[idx] / 1.25])
                logging.info('...Updating paramset %d, L=%f', idx, L[idx])
                params, L, orig_obj, new_obj = update(*params, orig_obj,
                                                      L, idx,
                                                      line_search_rate)
                new_elbo_delta += new_obj - orig_obj
                if (np.isclose(new_obj - orig_obj, 0, atol=conv_tol, rtol=0)
                        or L[idx] == 1 or L[idx] > L_MAX):
                    break
                orig_obj = new_obj

        if self.scale_se and new_elbo_delta < EM_TOL:
            orig_obj = self.elbo(params)
            self._update_error_scaling(params)
            params = self._nat_to_not_vi_delta(params)
            new_obj = self.elbo(params)
            new_elbo_delta += new_obj - orig_obj
            # assert new_obj - orig_obj >= np.min(
            #     [orig_obj-np.abs(1e-5*orig_obj), -1e-5]), (new_obj, orig_obj)
            logging.info('...Updating error_scaling, old ELBo=%f, '
                         'new ELBo=%f', orig_obj, new_obj)

        return params, L, new_elbo_delta

    def _log_likelihood(self, params):
        """Expected log likelihood of data under VI posterior"""
        post_means = self._posterior_mean(*params)
        post_vars = self._posterior_marginal_variance(post_means, *params)
        linked_ests = np.empty_like(post_means)
        scaled_mu = _fast_divide(post_means, self.std_errs)
        for p in range(self.num_pops):
            linked_ests[p] = self.ld_mats[p].dot(scaled_mu[p])
            # this_like = -0.5 * (self.ld_diags[p] * post_vars[p]
            #                     * (self.std_errs[p] ** (-2))).sum()
            # scaled_mu = post_means[p] / self.std_errs[p]
            # this_like += -0.5 * scaled_mu.dot(self.ld_mats[p].dot(scaled_mu))
            # this_like += post_means[p].dot(self.adj_marginal_effects[p])
            # this_like /= self.error_scaling[p]
            # this_like += (-0.5 * self.ld_ranks[p]
            #               * np.log(self.error_scaling[p]))
            # this_like += (-0.5 * self.chi_stat[p] / self.error_scaling[p])
            # to_return += this_like
        to_return = _fast_likelihood(post_means,
                                     post_vars,
                                     scaled_mu,
                                     self.scaled_ld_diags,
                                     linked_ests,
                                     self.adj_marginal_effects,
                                     self.chi_stat,
                                     self.ld_ranks,
                                     self.error_scaling)
        return to_return

    def _update_error_scaling(self, params):
        to_return = np.zeros_like(self.error_scaling)
        post_means = self._posterior_mean(*params)
        post_vars = self._posterior_marginal_variance(post_means, *params)
        for p in range(self.num_pops):
            scaled_mu = post_means[p] / self.std_errs[p]
            to_return[p] = (
                self.chi_stat[p]
                - 2 * post_means[p].dot(self.adj_marginal_effects[p])
                + scaled_mu.dot(self.ld_mats[p].dot(scaled_mu))
                + (self.ld_diags[p] * post_vars[p]
                   * (self.std_errs[p]**(-2))).sum()
            ) / self.ld_ranks[p]
        self.error_scaling = to_return

    def _beta_objective(self, params):
        return self._log_likelihood(params) - self._beta_KL(*params)

    def _hyper_delta_objective(self, params):
        return -self._delta_KL(*params) - self._annotation_KL(*params)

    def _annotation_objective(self, params):
        return -self._annotation_KL(*params)


class MultiPopVI(Nat_grad_optimizer):
    def __init__(self, mixture_covs=None,
                 num_random=0, **kwargs):
        num_pops = kwargs['marginal_effects'].shape[0]
        for mc in mixture_covs:
            assert mc.shape == (num_pops, num_pops)
        signs, _ = np.linalg.slogdet(mixture_covs)
        assert np.all(signs == 1)

        self.num_mix = len(mixture_covs)
        Nat_grad_optimizer.__init__(self,
                                    mixture_covs=mixture_covs,
                                    **kwargs)
        self.param_names = ['vi_mu', 'vi_delta', 'hyper_delta']

        # get out precision matrices
        new_mixture_covs = np.array(mixture_covs)[:, :, :, None]
        self.mixture_prec = _vi_sigma_inv(new_mixture_covs)
        self.log_det = _vi_sigma_log_det(new_mixture_covs)

        self.log_det = np.copy(self.log_det[:, 0])

        variances = np.zeros((self.num_mix,
                              self.num_pops,
                              self.num_pops,
                              self.num_loci), dtype=np.float64)
        variances[:,
                  np.arange(self.num_pops),
                  np.arange(self.num_pops),
                  :] = (self.std_errs ** -2 * self.ld_diags)
        variances += self.mixture_prec
        self.vi_sigma = _vi_sigma_inv(variances)
        self.nat_sigma = -0.5 * variances
        self.vi_sigma_log_det = _vi_sigma_log_det(self.vi_sigma)
        self.vi_sigma_matches = self._fast_einsum('kpqd,kqpi->ik',
                                                  self.mixture_prec,
                                                  self.vi_sigma,
                                                  key='vi_sigma_match')
        self.sigma_summary = (self.log_det
                              - self.vi_sigma_log_det.T
                              + self.vi_sigma_matches)
        self.nat_grad_vi_delta = None

    def _nat_to_not_vi_delta(self, params):
        vi_mu, vi_delta, hyper_delta = params
        nat_mu = _fast_nat_inner_product_m2(vi_mu, self.nat_sigma)
        const_part = _get_const_part(self.vi_sigma_log_det)
        vi_delta = _fast_invert_nat_vi_delta(vi_mu,
                                             nat_mu,
                                             const_part,
                                             self.nat_grad_vi_delta)
        return vi_mu, vi_delta, hyper_delta

    def _initialize(self):
        real_mu = self.inverse_betas
        logging.info('Largest inverse_beta is %f', np.max(np.abs(real_mu)))
        missing = np.isclose(self.ld_diags, 0)

        fake_mu = np.copy(real_mu)
        fake_mu = np.random.normal(loc=fake_mu,
                                   scale=1e-3*self.std_errs,
                                   size=fake_mu.shape)
        fake_mu[missing] = np.nan
        mu_fill = np.tile(np.nanmean(fake_mu, axis=0),
                          [fake_mu.shape[0], 1])
        fake_mu[missing] = mu_fill[missing]
        fake_mu[np.isnan(fake_mu)] = 0.
        probs = np.einsum('pi,oi,kpod->ik',
                          1.6*fake_mu,
                          1.6*fake_mu,
                          self.mixture_prec)
        probs += self.vi_sigma_matches
        probs -= self.log_det
        probs = np.exp(-0.5 * (probs - np.min(probs, axis=1, keepdims=True)))
        vi_delta = np.maximum(probs / probs.sum(axis=1, keepdims=True),
                              EPSILON)
        real_hyper_delta = _sum_annotations(vi_delta,
                                            self.annotations,
                                            self.num_annotations)
        # new addition:
        real_hyper_delta += 1. / self.num_loci
        # done
        real_hyper_delta /= np.sum(real_hyper_delta, axis=1, keepdims=True)
        real_hyper_delta = np.maximum(real_hyper_delta, EPSILON)

        vi_mu = np.einsum('k,pi->kpi',
                          np.ones(self.num_mix),
                          fake_mu)

        # new additions:
        nat_vi_delta = _fast_vi_delta_grad(
            real_hyper_delta,
            self.log_det,
            self.annotations
        )
        self.nat_grad_vi_delta = nat_vi_delta
        const_part = _get_const_part(self.vi_sigma_log_det)
        nat_mu = _fast_nat_inner_product_m2(vi_mu, self.nat_sigma)
        vi_delta = _fast_invert_nat_vi_delta(vi_mu,
                                             nat_mu,
                                             const_part,
                                             nat_vi_delta,)
        # done

        return vi_mu, vi_delta, real_hyper_delta

    def _update_error_scaling(self, params):
        Nat_grad_optimizer._update_error_scaling(self, params)
        variances = np.zeros((self.num_mix,
                              self.num_pops,
                              self.num_pops,
                              self.num_loci), dtype=np.float64)
        variances[:,
                  np.arange(self.num_pops),
                  np.arange(self.num_pops),
                  :] = (self.std_errs ** -2 * self.ld_diags
                        / self.error_scaling.reshape((-1, 1)))
        variances += self.mixture_prec
        self.vi_sigma = _vi_sigma_inv(variances)
        self.nat_sigma = -0.5 * variances
        self.vi_sigma_log_det = _vi_sigma_log_det(self.vi_sigma)
        self.vi_sigma_matches = self._fast_einsum('kpqd,kqpi->ik',
                                                  self.mixture_prec,
                                                  self.vi_sigma,
                                                  key='vi_sigma_match')
        self.sigma_summary = (self.log_det
                              - self.vi_sigma_log_det.T
                              + self.vi_sigma_matches)

    def _real_posterior_mean(self, vi_mu, vi_delta, hyper_delta):
        return self._fast_einsum('kpi,ik,pi->pi',
                                 vi_mu,
                                 vi_delta,
                                 self.scalings,
                                 key='_real_posterior_mean1')

    def _posterior_mean(self, vi_mu, vi_delta, hyper_delta):
        return _fast_posterior_mean(vi_mu, vi_delta)
        # return self._fast_einsum('kpi,ik->pi',
        #                          vi_mu,
        #                          vi_delta,
        #                          key='_posterior_mean1')

    def _posterior_marginal_variance(self, mean, vi_mu, vi_delta, hyper_delta):
        # mean_sq = self._posterior_mean(vi_mu, vi_delta, hyper_delta)**2
        temp = self._fast_einsum('kppi->kpi', self.vi_sigma, key='pmv1')
        # second_moment = self._fast_einsum('kpi,ik->pi',
        #                                   temp + vi_mu**2,
        #                                   vi_delta,
        #                                   key='pmv2')
        return _fast_pmv(mean, vi_mu, vi_delta, temp)

    def _update_beta(self, vi_mu, vi_delta, hyper_delta, orig_obj, L,
                     idx, lsr):
        if orig_obj is None:
            orig_obj = self._beta_objective((vi_mu, vi_delta, hyper_delta))
        # old_nat_mu = -2 * self._fast_einsum('spqi,sqi->spi',
        #                                     self.nat_sigma,
        #                                     vi_mu,
        #                                     key='_update_beta1')
        old_nat_mu = _fast_nat_inner_product_m2(vi_mu, self.nat_sigma)
        const_part = _get_const_part(self.vi_sigma_log_det)

        assert self.nat_grad_vi_delta is not None

        nat_grad_mu = self._nat_grad_beta(vi_mu,
                                          vi_delta,
                                          hyper_delta)
        while True:
            step_size = 1. / L[idx]
            # nat_mu = step_size * nat_grad_mu + (1. - step_size) * old_nat_mu
            nat_mu = _sum_betas(old_nat_mu, nat_grad_mu, step_size)
            # nat_vi_delta = (step_size * nat_grad_vi_delta
            #                 + (1. - step_size) * old_nat_vi_delta)
            # nat_vi_delta = _sum_deltas(old_nat_vi_delta,
            #                            self.nat_grad_vi_delta,
            #                            step_size)
            # new_mu = self._fast_einsum('spqi,sqi->spi',
            #                            self.vi_sigma,
            #                            nat_mu,
            #                            key='_update_beta1')
            new_mu = _fast_nat_inner_product(nat_mu, self.vi_sigma)
            # quad_forms = (new_mu * nat_mu).sum(axis=1)
            # nat_adj = -.5*(quad_forms.T + const_part)
            # nat_adj = nat_adj[:, :-1] - nat_adj[:, -1:]
            # new_vi_delta = _invert_nat_cat_2D(nat_vi_delta - nat_adj)
            # new_vi_delta = np.maximum(new_vi_delta, EPSILON)
            new_vi_delta = _fast_invert_nat_vi_delta(new_mu,
                                                     nat_mu,
                                                     const_part,
                                                     self.nat_grad_vi_delta)
            new_obj = self._beta_objective((new_mu, new_vi_delta, hyper_delta))
            logging.info('...Old objective = %f, new objective = %f',
                         orig_obj, new_obj)
            if new_obj >= orig_obj:
                if L[idx] > L_MAX:
                    assert np.isclose(orig_obj, new_obj)
                break
            if L[idx] > L_MAX:
                assert np.isclose(orig_obj, new_obj), np.max(vi_delta)
                return (vi_mu, vi_delta,
                        hyper_delta), L, orig_obj, orig_obj
            L[idx] *= lsr
        return (new_mu, new_vi_delta,
                hyper_delta), L, orig_obj, new_obj

    def _nat_grad_beta(self, vi_mu, vi_delta, hyper_delta):

        post_mean = self._posterior_mean(vi_mu, vi_delta, hyper_delta)

        linked_ests = np.zeros_like(post_mean)
        post_zs = _fast_divide(post_mean, self.std_errs)
        for p in range(self.num_pops):
            linked_ests[p] = self.ld_mats[p].dot(post_zs[p, :])
        linked_ests = _fast_linked_ests(linked_ests,
                                        self.std_errs,
                                        post_mean,
                                        self.scaled_ld_diags)
        nat_grad_mu = self._fast_einsum('p,pi,k->kpi',
                                        1./self.error_scaling,
                                        self.adj_marginal_effects-linked_ests,
                                        np.ones(self.num_mix),
                                        key='_nat_grad_beta4')

        return nat_grad_mu

    def _update_hyper_delta(self, vi_mu, vi_delta, hyper_delta,
                            orig_obj, L, idx, lsr):
        if orig_obj is None:
            # orig_obj = self._hyper_delta_objective(
            #     (vi_mu, vi_delta, hyper_delta)
            # )
            orig_obj = self.elbo(
                (vi_mu, vi_delta, hyper_delta)
            )
        new_hyper_delta = _sum_annotations(vi_delta,
                                           self.annotations,
                                           self.num_annotations)
        new_hyper_delta = np.maximum(
            new_hyper_delta
            / (self.annotation_counts.reshape((-1, 1)) + EPSILON),
            EPSILON
        )
        new_hyper_delta /= new_hyper_delta.sum(axis=1, keepdims=True)
        # new_obj = self._hyper_delta_objective(
        #     (vi_mu, vi_delta, new_hyper_delta)
        # )

        self.nat_grad_vi_delta = _fast_vi_delta_grad(
            new_hyper_delta,
            self.log_det,
            self.annotations
        )
        _, new_vi_delta, _ = self._nat_to_not_vi_delta(
            (vi_mu, vi_delta, new_hyper_delta)
        )
        new_obj = self.elbo(
            (vi_mu, new_vi_delta, new_hyper_delta)
        )

        # assert new_obj > orig_obj - 1e-5

        logging.info('...Old objective = %f, new objective = %f',
                     orig_obj, new_obj)

        return (vi_mu, new_vi_delta,
                new_hyper_delta), L, orig_obj, new_obj

    def _update_annotation(self, vi_mu, vi_delta, hyper_delta,
                           orig_obj, L, idx, lsr):
        return (vi_mu, vi_delta,
                hyper_delta), L, 0., 0.

    def _delta_KL(self, vi_mu, vi_delta, hyper_delta):
        return _fast_delta_kl(vi_delta, hyper_delta,
                              np.copy(self.annotations))

    def _beta_KL(self, vi_mu, vi_delta, hyper_delta):
        delta_comp = _fast_delta_kl(vi_delta, hyper_delta,
                                    np.copy(self.annotations))
        inner_product_comp = _fast_inner_product_comp(vi_mu,
                                                      self.mixture_prec,
                                                      vi_delta)
        # var_comp = 0.5 * (self.vi_sigma_matches * vi_delta).sum()
        fast_comp = _fast_beta_kl(self.sigma_summary, vi_delta)

        # beta_kl = (delta_comp + log_det_comp + dim_comp
        #            + inner_product_comp + var_comp)
        beta_kl = delta_comp + inner_product_comp + fast_comp
        return beta_kl

    def _annotation_KL(self, *params):
        return 0.
