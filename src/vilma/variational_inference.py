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
from vilma.matrix_structure import BlockDiagonalMatrix


L_MAX = 1e12        # sets minimum natural gradient stepsize is 1/L_MAX
REL_TOL = 1e-6      # relative change convergence criterion for optimization
ABS_TOL = 1e-6      # absolute change convergence criterion for optimization
ELBO_TOL = 0.1      # ELBo change convergence criterion for optimization
EM_TOL = 10         # ELBo change threshold for updating error scaling
MAX_GRAD = 100      # Truncate gradient coordinates larger than MAX_GRAD
ELBO_MOMENTUM = 0.5     # Smoothing parameter for assessing ELBo changes


class VIScheme():
    """
    Parent class for VI models of GWAS summary statistics

    This class contains machinery to optimize the ELBo to fit variational
    families to the RSS likelihood model of GWAS data. The details of the VI
    family must be specified in subclasses, but the abstract machinery of
    computing likelihoods and performing coordinate ascent on the ELBo is
    implemented here.

    Fields:
        param_names: A list of strings containing the names of all of the VI
            parameters and hyperparameters.
        scaled: True if the prior is on frequency-scaled effect sizes. False if
            on unscaled-effect sizes.
        error_scaling: Numbers to scale the standard errors of each GWAS by.
        checkpoint: True if model parameters should be stored periodically.
        num_pops: Number of populations (or traits).
        num_loci: Number of SNPs.
        ld_diags: [num_pops][num_loci] numpy array containing the diagonal
            values of the LD matrices in each population.
        num_annotations: Total number of distinct annotations
        marginal_effects: [num_pops][num_loci] numpy array containing the
            marginal effect sizes estimated by GWAS
        std_errs: [num_pops][num_loci] numpy array containing the standard
            errors of the GWAS marginal effect sizes
        scalings: If prior is on scaled effect sizes, then `scalings` contains
            the values needed to multiply by to obtain the unscaled effect
            sizes.
        scaled_ld_diags: ld_diags * (std_errs**-2) -- useful for some repeated
            computations
        ld_mats: [num_pops] length list of BlockDiagonalMatrix objects that
            represent the LD matrix in each populations
        annotations: a [num_loci][num_annotations] numpy array one-hot encoded
            representation of which annotation each SNP belongs to.
        annotation_counts: Total number of SNPs belonging to each annotation
        checkpoint_freq: Number of iterations between saving all VI and model
            parameters
        checkpoint_path: Destination for saving model checkpoints
        init_hg: Estimate of the heritability of the trait in each population.
            Only used in initializing the VI parameters
        gwas_N:  (Effective) GWAS sample size in each population.  Only used in
            initializing the VI parameters.
        num_its: Maximum number of iterations of coordinate ascent to perform
            before terminating.
        adj_marginal_effects: S^{-1}XX^{-1}S^{-1}hat{beta} -- that is the
            GWAS marginal effect sizes, scaled by the standard errors,
            projected onto the space spanned by the LD Matrix and then scaled
            by the standard errors again. Stored for use in repeated
            computations. This is for each population, so it is a numpy array
            of shape [num_pops][num_loci]
        chi_stat: hat{beta}^T (SXS)^{-1} hat{beta} -- that is, the Mahalonobis
            distance of the marginal effect size estimates from the origin,
            with respect to the covariance matrix SXS, which is the LD matrix
            scaled on each side by the standard errors. This is a numpy array
            of shape [num_pops].
        ld_ranks: The rank of the LD matrix in each population.  A numpy array
            of shape [num_pops].
        inverse_betas: LDpred-inf style estimate of the true effect sizes. Only
            used for initialization of VI parameters.
    Methods:
        optimize: Initializes VI family and then performs coordinate ascent
            until convergence, returning optimal VI parameters.
        elbo: Compute the evidence lower bound of the VI distribution and model
            defined by the (hyper)parameters.
        real_posterior_mean: Compute the posterior mean (in the non-frequency
            scaled space) of the effect sizes at each SNP in each population.
    """
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
        """
        Initialize a VIScheme

        Args:
            marginal_effects: A [num_populations][num_SNPs] numpy array
                containing the marginal effect size estimates from a GWAS
            std_errs: A [num_populations][num_SNPs] numpy array containing the
                standard errors of the marginal effect size estimates from a
                GWAS
            ld_mats: A [num_populations] long list containing
                BlockDiagonalMatrix objects that represent the LD matrix within
                each population
            annotations: A [num_SNPs] integer numpy array containing the label
                of which annotation each SNP belongs to. For example, if
                annotations[i] is 0 then SNP i is in the zero'th annotation.
            mixture_covs: A [num_mixture_components][num_pops][num_pops] numpy
                array, where mixture_covs[k] represents the k'th covariance
                matrix in the prior.
            checkpoint: Whether to periodically store the (hyper)parameters of
                the model.
            checkpoint_freq: How many iterations between checkpointing the
                model.
            scaled: If true, the prior is on the frequency-scaled effect sizes.
                If false, the prior in on the natural-scale effect sizes.
            scale_se: If true, treat the scaling of the standard errors in each
                GWAS as a hyperparameter and optimize over them.
            output: Basename for saving files for checkpointing.
            gwas_N: Estimate of the (effective) GWAS sample size in each
                population. Only used in initializing the VI parameters.
            init_hg: Estimate of the heritability of the trait in each
                population. Only used in initializing the VI parameters.
            num_its: Maximum number of iterations of coordinate ascent to
                perform before terminating.
        """
        if init_hg is None:
            raise ValueError('init_hg must be specified when calling '
                             'VIScheme()')
        if gwas_N is None:
            raise ValueError('gwas_N must be specified when calling '
                             'VIScheme()')
        if marginal_effects is None:
            raise ValueError('marginal_effects must be specified when '
                             'calling VIScheme()')
        if std_errs is None:
            raise ValueError('std_errs must be specified when calling '
                             'VIScheme()')
        if ld_mats is None:
            raise ValueError('ld_mats must be specified when calling '
                             'VIScheme()')
        if annotations is None:
            raise ValueError('annotations must be specified when calling '
                             'VIScheme()')
        if mixture_covs is None:
            raise ValueError('mixture_covs must be specififed when calling '
                             'VIScheme()')
        if num_its is None:
            raise ValueError('num_its must be specified when calling '
                             'VIScheme()')
        if not np.all(np.isfinite(marginal_effects)):
            raise ValueError('Encountered an infinite or NaN value in the '
                             'GWAS effect size estimates')
        if not np.all(np.isfinite(std_errs)):
            raise ValueError('Encountered an infinity or NaN value in the '
                             'GWAS standard errors')

        self.scaled = scaled    # whether to do Z-scores or not
        self.scale_se = scale_se    # whether to learn an SE scaling
        self.error_scaling = np.ones(marginal_effects.shape[0])

        self.checkpoint = checkpoint
        checkpoint_path = '%s-checkpoint' % output

        self.num_pops = marginal_effects.shape[0]
        self.num_loci = marginal_effects.shape[1]
        if len(ld_mats) != self.num_pops:
            raise ValueError('Fewer LD matrices than populations.')
        for ld in ld_mats:
            if not isinstance(ld, BlockDiagonalMatrix):
                raise ValueError('LD Matrices must be '
                                 'of type BlockDiagonalMatrix.')
        self.ld_diags = np.concatenate(
            [ld.diag().reshape((1, -1)) for ld in ld_mats],
            axis=0
        )
        for p, ldm in enumerate(ld_mats):
            if ldm.shape != (self.num_loci, self.num_loci):
                raise ValueError('LD matrix shape does not match '
                                 'GWAS marginal effect size shape.')
        if not np.allclose(annotations.sum(axis=1), 1):
            raise ValueError('Some SNPs are either missing annotations '
                             'or have more than one annotation.')
        self.num_annotations = annotations.shape[1]
        if annotations.shape[0] != self.num_loci:
            raise ValueError('annotations dimension does not match GWAS '
                             'marginal effect size shape.')

        self.marginal_effects = np.copy(marginal_effects)
        if self.scaled:
            self.marginal_effects = self.marginal_effects / (std_errs +
                                                             numerics.EPSILON)
        if self.scaled:
            self.std_errs = np.ones_like(std_errs)
            self.scalings = (std_errs + numerics.EPSILON)
        else:
            self.std_errs = np.copy(std_errs)
            self.scalings = np.ones_like(std_errs)
        self.scaled_ld_diags = self.std_errs**-2 * self.ld_diags
        self.ld_mats = ld_mats
        self.annotations = np.copy(np.where(annotations)[1])
        self.annotation_counts = annotations.sum(axis=0)
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
        mle = np.zeros((self.num_pops, self.num_loci))
        self.ld_ranks = np.zeros(self.num_pops)
        self._einsum_paths = {}

        self.inverse_betas = np.zeros_like(self.marginal_effects)
        for p in range(self.num_pops):
            z_scores = self.marginal_effects[p] / self.std_errs[p]
            mle[p] = ld_mats[p].inverse.dot(z_scores)
            self.chi_stat[p] = z_scores.dot(mle[p])
            this_adj_marg = np.copy(mle[p])
            this_adj_marg = ld_mats[p].dot(this_adj_marg)
            this_adj_marg = this_adj_marg / self.std_errs[p]
            self.adj_marginal_effects[p, :] = this_adj_marg
            self.ld_ranks[p] = ld_mats[p].get_rank()

            prior = (2 * gwas_N[p] * init_hg[p]
                     / (self.std_errs[p, :]**-2).sum())
            inv_z_scores = ld_mats[p].ridge_inverse_dot(
                this_adj_marg * self.std_errs[p],
                self.std_errs[p, :]**2 / prior
            )
            self.inverse_betas[p, :] = inv_z_scores * self.std_errs[p]

        if not np.allclose(
            self.adj_marginal_effects[np.isclose(self.ld_diags, 0)],
            0
        ):
            raise ValueError('Some SNPs that are missing in the LD matrix '
                             'are not being treated as missing.')

    def _fast_einsum(self, *args, key=None):
        """
        Wrapper for np.einsum to enable caching

        Repeatedly computing optimal Einsum contraction paths for optimized
        np.einsum calls can be costly. These paths will remain the same
        throughout the lifetime of a VIScheme object, and hence we can cache
        them. Caching is done by calling with `key` and any time _fast_einsum
        is called with the same `key` the same Einsum contraction path will be
        used.

        Args:
            key: a hashable object that acts as a unique identifier for this
                einsum path. In general the same key should be used for any
                _fast_einsum call that will use the same contractions on
                arguments of the same shape.

        Returns:
            The results of running np.einsum on *args while using the optimal
            contraction path.
        """
        if key is None:
            return np.einsum(*args)
        if key not in self._einsum_paths:
            e_path = np.einsum_path(*args, optimize='optimal')
            logging.info('Einsum contraction path optimization results for '
                         '%s :\n %s',
                         key, e_path[1])
            self._einsum_paths[key] = e_path[0]
        return np.einsum(*args, optimize=self._einsum_paths[key])

    def _dump_info(self, num_its, new_post_mean, post_mean, ckp_post_mean):
        """Log information about convergence"""
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

    def optimize(self):
        """Initialize params and optimize objective function"""
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

            self._dump_info(num_its, new_post_mean, post_mean, ckp_post_mean)

            post_mean = new_post_mean
            num_its += 1

            params = tuple(new_params)

        if num_its == self.num_its:
            logging.warning('Failed to converge')
        logging.info('Optimization ran for %d iterations', num_its)
        return params

    def _optimize_step(self, params, L, curr_elbo,
                       line_search_rate=1.25, running_elbo_delta=None):
        """Update each set of params and tally up improvement in ELBo"""
        logging.info('Current ELBO = %f and L = %f,%f,%f,%f,%f',
                     curr_elbo, L[0], L[1], L[2], L[3], L[4])
        (new_params,
         L_new,
         elbo_change) = self._nat_grad_step(params, L, line_search_rate,
                                            running_elbo_delta)
        elbo = curr_elbo + elbo_change
        if running_elbo_delta is None:
            running_elbo_delta = elbo_change
        running_elbo_delta *= ELBO_MOMENTUM
        running_elbo_delta += (1 - ELBO_MOMENTUM) * elbo_change
        return new_params, L_new, elbo, running_elbo_delta

    def elbo(self, params):
        """Compute the ELBo for VI and model specified by `params`"""
        elbo = self._log_likelihood(params)
        elbo -= self._beta_KL(*params)
        elbo -= self._annotation_KL(*params)
        return elbo

    def _nat_grad_step(self, params, L, line_search_rate,
                       running_elbo_delta=None):
        """Perform one iteration of updating each set of (hyper)parameters"""
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
        """Update error scaling hyperparameters given VI distribution"""
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
        """ELBo terms containing VI distribution on betas"""
        return self._log_likelihood(params) - self._beta_KL(*params)

    def _hyper_delta_objective(self, params):
        """ELBo terms containing per-annotation mixture weights"""
        return -self._delta_KL(*params) - self._annotation_KL(*params)

    def _annotation_objective(self, params):
        """ELBo terms containing prior on per-annotation mixture weights"""
        return -self._annotation_KL(*params)

    # the following methods must be implemented by any subclass
    def _annotation_KL(self, *params):
        """Compute KL divergence between VI and prior for annotations"""
        raise NotImplementedError('_annotation_KL must be implemented by '
                                  'any VIScheme subclass')

    def _beta_KL(self, *params):
        """Compute KL divergence between VI and prior for betas"""
        raise NotImplementedError('_beta_KL must be implemented by '
                                  'any VIScheme subclass')

    def _delta_KL(self, *params):
        """Compute KL divergence between VI and prior for mixture weights"""
        raise NotImplementedError('_delta_KL must be implemented by '
                                  'any VIScheme subclass')

    def _update_annotation(self, *params, orig_obj, L, idx, lsr):
        """Update VI distribution for annotations"""
        raise NotImplementedError('_update_annotation must be implemented by '
                                  'any VIScheme subclass')

    def _update_hyper_delta(self, *params, orig_obj, L, idx, lsr):
        """Update VI distribution for mixture weights"""
        raise NotImplementedError('_update_hyper_delta must be implemented by '
                                  'any VIScheme subclass')

    def _update_beta(self, *params, orig_obj, L, idx, lsr):
        """Update VI distribution for effect sizes"""
        raise NotImplementedError('_update_beta must be implemented by any '
                                  'VIScheme subclass')

    def _posterior_marginal_variance(self, mean, *params):
        """Compute the posterior variance of each beta"""
        raise NotImplementedError('_posterior_marginal_variance must be '
                                  'implemented by any VIScheme subclass')

    def _posterior_mean(self, *params):
        """Compute the posterior mean of each beta"""
        raise NotImplementedError('_posterior_mean must be implemented '
                                  'by any VIScheme subclass')

    def real_posterior_mean(self, *params):
        """Compute the posterior mean of each beta in unscaled space"""
        raise NotImplementedError('real_posterior_mean must be implemented '
                                  'by any VIScheme subclass')

    def _initialize(self):
        """Compute initial values for all VI parameters"""
        raise NotImplementedError('_initialize must be implemented by any '
                                  'VIScheme subclass')


class MultiPopVI(VIScheme):
    """
    Standard VI scheme for GWAS across one or more populations

    Implements all of the methods needed to fit a particular VIScheme where
    cohorts are assumed to be independent given their true effect sizes.

    See superclass for additional Fields and Methods

    Fields:
        num_mix: The number of mixture components in the prior
        mixture_prec: List of the inverses of the covariance matrices for each
            of the mixture components. Numpy array of shape
            [num_mix][num_pops][num_pops][1]
        log_det: Log determinants of each of covariances for each of the
            mixture components.  Numpy array of shape
            [num_mix]
        vi_sigma: Optimal value of the variances for the variational family.
            Numpy array of shape [num_mix][num_pops][num_pops][num_loci]
        nat_sigma: -1/2 * inverse(vi_sigma). The natural parameterization of
            the variances for the variational family.  A numpy array of shape
            [num_mix][num_pops][num_pops][num_loci]
        vi_sigma_log_det: Log determinants of each of the covariance matrices
            in `vi_sigma`.  Numpy array of shape [num_mix][num_loci]
        vi_sigma_matches: Traces of `mixture_prec` and `vi_sigma` -- used to
            measure their discrepancy in computing KL divergences.
        vi_sigma_summary: All of the terms involving only the covariance
            matrices in the KL divergence between a Normal distributions with
            covariance matrices `mixture_covs` and `vi_sigma`.
        nat_grad_vi_delta: Optimal natural parameterization of the mixture
            weights of the variational family.
    """
    def __init__(self, mixture_covs=None, **kwargs):
        """
        Initialize a MultiPopVI scheme

        See superclass initialization for arguments.
        """
        num_pops = kwargs['marginal_effects'].shape[0]
        for mc in mixture_covs:
            if mc.shape != (num_pops, num_pops):
                raise ValueError('Mixture component has a '
                                 'covariance matrix of the wrong shape.')
        signs, _ = np.linalg.slogdet(mixture_covs)
        if not np.all(signs == 1):
            raise ValueError('Mixture component has a non-positive definite '
                             'covariance matrix.')

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
        """Convert natural parameterization of delta to delta"""
        vi_mu, vi_delta, hyper_delta = params
        nat_mu = numerics.fast_nat_inner_product_m2(vi_mu, self.nat_sigma)
        const_part = np.copy(self.vi_sigma_log_det.T)
        vi_delta = numerics.fast_invert_nat_vi_delta(vi_mu,
                                                     nat_mu,
                                                     const_part,
                                                     self.nat_grad_vi_delta)
        return vi_mu, vi_delta, hyper_delta

    def _initialize(self):
        """Obtain starting values of the variational parameters"""
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
        """Update the standard error scaling hyperparameter"""
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
        """Compute the posterior mean in unscaled units"""
        return self._fast_einsum('kpi,ik,pi->pi',
                                 vi_mu,
                                 vi_delta,
                                 self.scalings,
                                 key='_real_posterior_mean1')

    def _posterior_mean(self, vi_mu, vi_delta, hyper_delta):
        """Compute the posterior mean"""
        return numerics.fast_posterior_mean(vi_mu, vi_delta)

    def _posterior_marginal_variance(self, mean, vi_mu, vi_delta, hyper_delta):
        """Compute the posterior variance at in each population at each SNP"""
        temp = self._fast_einsum('kppi->kpi', self.vi_sigma, key='pmv1')
        return numerics.fast_pmv(mean, vi_mu, vi_delta, temp)

    def _update_beta(self, vi_mu, vi_delta, hyper_delta, orig_obj, L,
                     idx, lsr):
        """Take a natural gradient step for the variational family for beta"""
        if orig_obj is None:
            orig_obj = self._beta_objective((vi_mu, vi_delta, hyper_delta))
        old_nat_mu = numerics.fast_nat_inner_product_m2(vi_mu, self.nat_sigma)
        const_part = np.copy(self.vi_sigma_log_det.T)

        if self.nat_grad_vi_delta is None:
            raise RuntimeError('nat_grad_vi_delta must always be set '
                               'prior to running _update_beta')

        nat_grad_mu = self._nat_grad_beta(vi_mu,
                                          vi_delta,
                                          hyper_delta)
        while True:
            step_size = 1. / L[idx]
            nat_mu = numerics.sum_betas(old_nat_mu, nat_grad_mu, step_size)
            new_mu = numerics.fast_nat_inner_product(nat_mu, self.vi_sigma)
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
                    if not np.isclose(orig_obj, new_obj):
                        raise RuntimeError('Encountered a numerical error.')
                break
            if L[idx] > L_MAX:
                if not np.isclose(orig_obj, new_obj):
                    raise RuntimeError('Encountered a numerical error.')
                return (vi_mu, vi_delta,
                        hyper_delta), L, orig_obj, orig_obj
            L[idx] *= lsr
        return (new_mu, new_vi_delta,
                hyper_delta), L, orig_obj, new_obj

    def _nat_grad_beta(self, vi_mu, vi_delta, hyper_delta):
        """Compute the natural gradient of the variational family of beta"""

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
        """Update the mixture weights hyperparameter"""
        if orig_obj is None:
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

        logging.info('...Old objective = %f, new objective = %f',
                     orig_obj, new_obj)

        return (vi_mu, new_vi_delta,
                new_hyper_delta), L, orig_obj, new_obj

    def _update_annotation(self, vi_mu, vi_delta, hyper_delta,
                           orig_obj, L, idx, lsr):
        """In this VI scheme, this does nothing"""
        return (vi_mu, vi_delta,
                hyper_delta), L, 0., 0.

    def _delta_KL(self, vi_mu, vi_delta, hyper_delta):
        """The KL divergence of the mixture weights distributions"""
        return numerics.fast_delta_kl(vi_delta, hyper_delta,
                                      np.copy(self.annotations))

    def _beta_KL(self, vi_mu, vi_delta, hyper_delta):
        """The KL divergence of the beta distributions"""
        delta_comp = numerics.fast_delta_kl(vi_delta, hyper_delta,
                                            np.copy(self.annotations))
        inner_product_comp = numerics.fast_inner_product_comp(
            vi_mu,
            self.mixture_prec,
            vi_delta
        )
        fast_comp = numerics.fast_beta_kl(self.sigma_summary, vi_delta)

        beta_kl = delta_comp + inner_product_comp + fast_comp
        return beta_kl

    def _annotation_KL(self, *params):
        """In this VI scheme, this does nothing"""
        return 0.
