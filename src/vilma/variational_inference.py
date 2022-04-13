"""
Variational Inference engine to optimize ELBos

Contains a parent class for any VI scheme and an implementation of the
multi-population VI scheme.

Classes:
    VIScheme: Parent class for any VI scheme. front_end.py assumes
        that any VI implementation will subclass VIScheme
    MultiPopVI: Main implementation of Vilma VI scheme.
"""
import logging
import numpy as np
from vilma import numerics


L_MAX = 1e12        # sets minimum natural gradient stepsize is 1/L_MAX
REL_TOL = 1e-6      # relative change convergence criterion for optimization
ABS_TOL = 1e-6      # absolute change convergence criterion for optimization
ELBO_TOL = 0.1      # ELBo change convergence criterion for optimization
EM_TOL = 10         # ELBo change threshold for updating error scaling
MAX_GRAD = 100      # Truncate gradient coordinates larger than MAX_GRAD
ELBO_MOMENTUM = 0.5     # Smoothing parameter for assessing ELBo changes


class VIScheme():

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
                                                             numerics.EPSILON)
        # self.marginal_effects.flags.writeable = False
        if self.scaled:
            self.std_errs = np.ones_like(std_errs)
            self.scalings = (std_errs + numerics.EPSILON)
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
        post_mean = self.real_posterior_mean(*params)
        ckp_post_mean = self.real_posterior_mean(*checkpoint_params)
        while num_its < self.num_its and not converged:
            if num_its % self.checkpoint_freq == 0 and self.checkpoint:
                checkpoint_params = params
                ckp_post_mean = self.real_posterior_mean(*checkpoint_params)
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

            new_post_mean = self.real_posterior_mean(*new_params)

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
        scaled_mu = numerics.fast_divide(post_means, self.std_errs)
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
        to_return = numerics.fast_likelihood(post_means,
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


class MultiPopVI(VIScheme):
    def __init__(self, mixture_covs=None,
                 num_random=0, **kwargs):
        num_pops = kwargs['marginal_effects'].shape[0]
        for mc in mixture_covs:
            assert mc.shape == (num_pops, num_pops)
        signs, _ = np.linalg.slogdet(mixture_covs)
        assert np.all(signs == 1)

        self.num_mix = len(mixture_covs)
        VIScheme.__init__(self,
                          mixture_covs=mixture_covs,
                          **kwargs)
        self.param_names = ['vi_mu', 'vi_delta', 'hyper_delta']

        # get out precision matrices
        new_mixture_covs = np.array(mixture_covs)[:, :, :, None]
        self.mixture_prec = numerics.vi_sigma_inv(new_mixture_covs)
        self.log_det = numerics.vi_sigma_log_det(new_mixture_covs)

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
        self.vi_sigma = numerics.vi_sigma_inv(variances)
        self.nat_sigma = -0.5 * variances
        self.vi_sigma_log_det = numerics.vi_sigma_log_det(self.vi_sigma)
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
        nat_mu = numerics.fast_nat_inner_product_m2(vi_mu, self.nat_sigma)
        const_part = np.copy(self.vi_sigma_log_det.T)
        vi_delta = numerics.fast_invert_nat_vi_delta(vi_mu,
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
                              numerics.EPSILON)
        real_hyper_delta = numerics.sum_annotations(
            vi_delta,
            self.annotations,
            self.num_annotations
        )
        real_hyper_delta += 1. / self.num_loci
        real_hyper_delta /= np.sum(real_hyper_delta, axis=1, keepdims=True)
        real_hyper_delta = np.maximum(real_hyper_delta, numerics.EPSILON)

        vi_mu = np.einsum('k,pi->kpi',
                          np.ones(self.num_mix),
                          fake_mu)

        nat_vi_delta = numerics.fast_vi_delta_grad(
            real_hyper_delta,
            self.log_det,
            self.annotations
        )
        self.nat_grad_vi_delta = nat_vi_delta
        const_part = np.copy(self.vi_sigma_log_det.T)
        nat_mu = numerics.fast_nat_inner_product_m2(vi_mu, self.nat_sigma)
        vi_delta = numerics.fast_invert_nat_vi_delta(vi_mu,
                                                     nat_mu,
                                                     const_part,
                                                     nat_vi_delta,)

        return vi_mu, vi_delta, real_hyper_delta

    def _update_error_scaling(self, params):
        VIScheme._update_error_scaling(self, params)
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
        self.vi_sigma = numerics.vi_sigma_inv(variances)
        self.nat_sigma = -0.5 * variances
        self.vi_sigma_log_det = numerics.vi_sigma_log_det(self.vi_sigma)
        self.vi_sigma_matches = self._fast_einsum('kpqd,kqpi->ik',
                                                  self.mixture_prec,
                                                  self.vi_sigma,
                                                  key='vi_sigma_match')
        self.sigma_summary = (self.log_det
                              - self.vi_sigma_log_det.T
                              + self.vi_sigma_matches)

    def real_posterior_mean(self, vi_mu, vi_delta, hyper_delta):
        return self._fast_einsum('kpi,ik,pi->pi',
                                 vi_mu,
                                 vi_delta,
                                 self.scalings,
                                 key='_real_posterior_mean1')

    def _posterior_mean(self, vi_mu, vi_delta, hyper_delta):
        return numerics.fast_posterior_mean(vi_mu, vi_delta)
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
        return numerics.fast_pmv(mean, vi_mu, vi_delta, temp)

    def _update_beta(self, vi_mu, vi_delta, hyper_delta, orig_obj, L,
                     idx, lsr):
        if orig_obj is None:
            orig_obj = self._beta_objective((vi_mu, vi_delta, hyper_delta))
        # old_nat_mu = -2 * self._fast_einsum('spqi,sqi->spi',
        #                                     self.nat_sigma,
        #                                     vi_mu,
        #                                     key='_update_beta1')
        old_nat_mu = numerics.fast_nat_inner_product_m2(vi_mu, self.nat_sigma)
        const_part = np.copy(self.vi_sigma_log_det.T)

        assert self.nat_grad_vi_delta is not None

        nat_grad_mu = self._nat_grad_beta(vi_mu,
                                          vi_delta,
                                          hyper_delta)
        while True:
            step_size = 1. / L[idx]
            # nat_mu = step_size * nat_grad_mu + (1. - step_size) * old_nat_mu
            nat_mu = numerics.sum_betas(old_nat_mu, nat_grad_mu, step_size)
            # nat_vi_delta = (step_size * nat_grad_vi_delta
            #                 + (1. - step_size) * old_nat_vi_delta)
            # nat_vi_delta = _sum_deltas(old_nat_vi_delta,
            #                            self.nat_grad_vi_delta,
            #                            step_size)
            # new_mu = self._fast_einsum('spqi,sqi->spi',
            #                            self.vi_sigma,
            #                            nat_mu,
            #                            key='_update_beta1')
            new_mu = numerics.fast_nat_inner_product(nat_mu, self.vi_sigma)
            # quad_forms = (new_mu * nat_mu).sum(axis=1)
            # nat_adj = -.5*(quad_forms.T + const_part)
            # nat_adj = nat_adj[:, :-1] - nat_adj[:, -1:]
            # new_vi_delta = _invert_nat_cat_2D(nat_vi_delta - nat_adj)
            # new_vi_delta = np.maximum(new_vi_delta, EPSILON)
            new_vi_delta = numerics.fast_invert_nat_vi_delta(
                new_mu,
                nat_mu,
                const_part,
                self.nat_grad_vi_delta
            )
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
        post_zs = numerics.fast_divide(post_mean, self.std_errs)
        for p in range(self.num_pops):
            linked_ests[p] = self.ld_mats[p].dot(post_zs[p, :])
        linked_ests = numerics.fast_linked_ests(linked_ests,
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
        new_hyper_delta = numerics.sum_annotations(
            vi_delta,
            self.annotations,
            self.num_annotations
        )
        new_hyper_delta = np.maximum(
            new_hyper_delta
            / (self.annotation_counts.reshape((-1, 1)) + numerics.EPSILON),
            numerics.EPSILON
        )
        new_hyper_delta /= new_hyper_delta.sum(axis=1, keepdims=True)
        # new_obj = self._hyper_delta_objective(
        #     (vi_mu, vi_delta, new_hyper_delta)
        # )

        self.nat_grad_vi_delta = numerics.fast_vi_delta_grad(
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
        return numerics.fast_delta_kl(vi_delta, hyper_delta,
                                      np.copy(self.annotations))

    def _beta_KL(self, vi_mu, vi_delta, hyper_delta):
        delta_comp = numerics.fast_delta_kl(vi_delta, hyper_delta,
                                            np.copy(self.annotations))
        inner_product_comp = numerics.fast_inner_product_comp(
            vi_mu,
            self.mixture_prec,
            vi_delta
        )
        fast_comp = numerics.fast_beta_kl(self.sigma_summary, vi_delta)

        # beta_kl = (delta_comp + log_det_comp + dim_comp
        #            + inner_product_comp + var_comp)
        beta_kl = delta_comp + inner_product_comp + fast_comp
        return beta_kl

    def _annotation_KL(self, *params):
        return 0.
