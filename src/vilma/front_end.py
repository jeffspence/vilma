from __future__ import division

import numpy as np
import logging
import argparse
import itertools

from natural_gradient import Easy_annotation_smart_vi
import load
import scipy.stats
import pickle


def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-K', '--components', default=12, type=int,
                        help='number of mixture components in prior'),
    parser.add_argument('--num-its', default=1000, type=int,
                        help='Maximum number of optimization iterations.'),
    parser.add_argument('--xpcor', default=0.5, type=float,
                        help='Expected Cross-population correlation')
    parser.add_argument('--covariance-structure', default='simple', type=str,
                        help='Named cross-population covariance prior')
    parser.add_argument('--ld-schema', required=True, type=str,
                        help='Comma-separate paths to LD panel schemas')
    parser.add_argument('--sumstats', required=True, type=str,
                        help='Comma-separated paths to summary statistics')
    parser.add_argument('--stderrscale', default='1.0', type=str,
                        required=False,
                        help='Comma separated list of values to multiply'
                             'summary stat stderrs by.')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Comma-separated paths to annotation file')
    parser.add_argument('--output', required=True, type=str,
                        help='Output path prefix')
    parser.add_argument('--names', type=str,
                        help='Comma-separated names of the '
                             'populations for output')
    parser.add_argument('--method', type=str, default='easy',
                        help='Use the "easy" annotation model, '
                             'or the "hard" annotation model.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Verbose output')
    parser.add_argument('--extract', required=True, type=str,
                        help='List of SNPs to include in analysis, '
                             'with A1/A2 columns for alignment')
    parser.add_argument('--scaled', dest='scaled', action='store_true')
    parser.add_argument('--ldthresh', required=False, default=0.0,
                        help='Threhold for singular value approximation of '
                             'LD matrix.',
                        type=float)
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random number generation.')
    parser.add_argument('--mmap', dest='mmap', action='store_true')
    parser.add_argument('--learn-scaling', dest='scale_se',
                        action='store_true',
                        help='Whether or not to learn a scaling'
                             'factor for the standard errors.')
    parser.add_argument('--samplesizes', type=str, default='100e3',
                        help='Comma-separated list of values to use '
                             'for the GWAS sample sizes when initializing.')
    parser.add_argument('--init-hg', type=str, default='0.1',
                        help='Comma-separated list of heritabilities '
                             'to use for each population when '
                             'initializing.  These do not need to be '
                             'particular accurate! They are just for '
                             'initializing the optimization algorithm.')
    parser.add_argument('--trait', dest='trait', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)

    logger = logging.getLogger()
    logger.setLevel(10*(2-args.verbose))     # defaults to 'INFO' logging

    if (not args.trait
            and args.ld_schema.count(',') != 1
            and args.ld_schema.count(',') != args.sumstats.count(',')):
        raise IOError('Either need to imput one ld_schema or provide a '
                      'sumstats file for each ld_schema.')
    P = args.sumstats.count(',') + 1
    K = args.components

    names = list(map(str, range(P)))

    if args.names is not None:
        assert args.names.count(',') == args.sumstats.count(',')
        names = args.names.split(',')

    logger.info('Loading variants...')
    variants = load.load_variant_list(args.extract)

    logger.info('Loading annotations...')
    annotations, denylist = load.load_annotations(args.annotations,
                                                  variants=variants)

    combined_ld = []
    combined_betas = []
    combined_errors = []

    stderr_mult = np.zeros(len(args.sumstats.split(',')))
    stderr_mult[:] = list(map(float, args.stderrscale.split(',')))

    gwas_N = np.zeros_like(stderr_mult)
    gwas_N[:] = list(map(float, args.samplesizes.split(',')))

    init_hg = np.zeros_like(gwas_N)
    init_hg[:] = list(map(float, args.init_hg.split(',')))

    # LD matrices
    if args.trait:
        assert len(args.ld_schema.split(',')) == 1
        any_missing = None
        for idx, sumstats_path in enumerate(args.sumstats.split(',')):
            logger.info('Loading sumstats for trait %d...' % (idx+1))
            sumstats, missing = load.load_sumstats(sumstats_path,
                                                   variants=variants)
            if any_missing is None:
                any_missing = missing.tolist()
            else:
                any_missing.extend(missing.tolist())

            combined_betas.append(np.array(sumstats.BETA).reshape((1, -1)))
            logger.info('Largest beta is... %f',
                        np.max(np.abs(np.array(sumstats.BETA))))
            combined_errors.append(np.array(sumstats.SE).reshape((1, -1))
                                   * stderr_mult[idx])
        logger.info('Loading LD')
        any_missing = np.array(list(sorted(set(any_missing))))
        ld = load.load_ld_from_schema(args.ld_schema,
                                      variants=variants,
                                      denylist=any_missing,
                                      t=args.ldthresh,
                                      mmap=args.mmap)
        combined_ld.append(ld)

    else:
        for idx, (ld_schema_path, sumstats_path) in enumerate(
                zip(*(args.ld_schema.split(','),
                    args.sumstats.split(',')))):
            logger.info('Loading sumstats for population %d...' % (idx+1))
            sumstats, missing = load.load_sumstats(sumstats_path,
                                                   variants=variants)
            missing.extend(denylist)
            combined_betas.append(np.array(sumstats.BETA).reshape((1, -1)))
            logger.info('Largest beta is... %f',
                        np.max(np.abs(np.array(sumstats.BETA))))
            combined_errors.append(np.array(sumstats.SE).reshape((1, -1))
                                   * stderr_mult[idx])

            logger.info('Loading LD for population %d...' % (idx+1))
            ld = load.load_ld_from_schema(ld_schema_path,
                                          variants=variants,
                                          denylist=missing,
                                          t=args.ldthresh,
                                          mmap=args.mmap)
            combined_ld.append(ld)

    assert np.all(np.isfinite(combined_betas))
    assert np.all(np.isfinite(combined_errors))
    logger.info('Largest beta is... %f', np.max(np.abs(combined_betas)))

    betas = np.concatenate(combined_betas, axis=0)
    std_errs = np.concatenate(combined_errors, axis=0)

    logger.info('Building cross-population covariances...')
    if args.scaled:
        maxes = np.nanmax((betas/std_errs)**2, axis=1)
        # maxes = np.nanmax((betas/std_errs)**2-1, axis=1) # empirical Bayes
        mins = np.zeros_like(maxes)
        for p in range(len(mins)):
            this_keep = betas[p, :]**2 > 0
            mins[p] = np.nanpercentile(
               (betas[p, this_keep]/std_errs[p, this_keep])**2,
               2.5
            )
    else:
        # maxes = np.nanmax(betas**2, axis=1)   # old version
        maxes = np.zeros(betas.shape[0])
        mins = np.zeros_like(maxes)
        for p in range(len(mins)):
            keep = ~np.isnan(betas[p])
            this_beta = np.abs(betas[p, keep])
            this_se = std_errs[p, keep]
            psi = 1. / len(this_beta)
            probs = 1. / (1.
                          + ((1.-psi)/psi
                             * np.sqrt(this_beta**2/this_se**2)
                             * np.exp(-0.5*this_beta**2/this_se**2 + 0.5)))
            ebayes = np.maximum(this_beta**2 - this_se**2, 1e-10)
            raw_means = this_beta / (1. + this_se**2/ebayes**2)
            maxes[p] = np.max(probs*raw_means)**2
            mins[p] = np.nanpercentile(betas[p, betas[p, :]**2 > 0]**2, 2.5)

    cross_pop_covs = _make_simple(P, K, mins, maxes)

    print(cross_pop_covs)
    print([mc.shape for mc in cross_pop_covs])
    with open("%s.covariance.pkl" % args.output, "wb") as ofile:
        pickle.dump([cross_pop_covs], ofile)

    logger.info('Fitting...')

    if args.method == 'hard':
        raise NotImplementedError('Hard annotation model has not yet been '
                                  'implemented')
    else:
        if args.trait:
            '''
            elbo = Trait_projective_vi(
                marginal_effects=betas,
                std_errs=std_errs,
                ld_mats=combined_ld,
                mixture_covs=cross_pop_covs,
                annotations=annotations,
                checkpoint_freq=2*args.num_its,
                output=args.output,
                scaled=args.scaled,
                scale_se=True,
                gwas_N=gwas_N,
                init_hg=init_hg,
                num_its=args.num_its
            )
            np.save(args.ouput + '.projections',
                    elbo.projections)
            '''
            raise NotImplementedError('blah')
        else:
            elbo = Easy_annotation_smart_vi(
                marginal_effects=betas,
                std_errs=std_errs,
                ld_mats=combined_ld,
                mixture_covs=cross_pop_covs,
                annotations=annotations,
                checkpoint_freq=2*args.num_its,
                output=args.output,
                scaled=args.scaled,
                scale_se=args.scale_se,
                gwas_N=gwas_N,
                init_hg=init_hg,
                num_its=args.num_its,
            )
        # np.save(args.output + '.projections',
        #         elbo.projections)
        params = elbo.optimize()
        np.savez(args.output, **dict(zip(elbo.param_names, params)))
        for n, p in zip(names, elbo._real_posterior_mean(*params)):
            variants['posterior_' + n] = p

        variants.to_csv(args.output + '.estimates.tsv', sep='\t', index=False)


def _make_diag_vals(P, K, mins, maxes):
    diag_vals = []
    # include something that's basically zero
    diag_vals = [[m*1e-6 for m in mins]]
    for k in range(K+1):
        this_diag = []
        for p in range(P):
            this_diag.append(mins[p]
                             * np.exp(np.log(maxes[p]/mins[p]) / K * k))
        diag_vals.append(this_diag)
    return diag_vals


def _make_simple(P, K, mins, maxes):
    cross_pop_covs = []
    diag_vals = _make_diag_vals(P, K, mins, maxes)
    if P == 1:
        return list(np.array(diag_vals).reshape((K+2, P, P)))
    corr_vals = [-.99 + 1.98 * (k + 1) / K for k in range(K)]
    for idx, diag in enumerate(diag_vals):
        for off_diags in itertools.product(*[corr_vals]*((P*(P-1))//2)):
            mat = np.eye(P)
            mat[np.triu_indices_from(mat, k=1)] = off_diags
            mat.T[np.triu_indices_from(mat, k=1)] = off_diags
            mat = mat * np.sqrt(diag)
            mat = mat.T * np.sqrt(diag)
            for k in range(3):
                scale = np.diag(
                    np.sqrt(np.exp(np.random.uniform(-1, 1, P)))
                )
                cross_pop_covs.append(scale.dot(mat.dot(scale)))
        if idx > 0:
            # does population specific causals
            # correlation does not really matter
            for p in range(P):
                single_pop = np.copy(diag_vals[0])
                single_pop[p] = diag[p]
                mat = np.diag(single_pop)
                for k in range(3):
                    scale = np.diag(
                        np.sqrt(np.exp(np.random.uniform(-1, 1, P)))
                    )
                    cross_pop_covs.append(scale.dot(mat.dot(scale)))

    signs, _ = np.linalg.slogdet(cross_pop_covs)
    assert np.all(signs == 1)
    return cross_pop_covs


def _corr_mat(P, xpcor, scale=1):
    to_return = ((1 - xpcor) * np.eye(P) + xpcor * np.ones((P, P)))
    to_return = to_return * np.sqrt(scale)
    to_return = to_return.T * np.sqrt(scale)
    to_return = to_return
    return to_return


def _random_entry(P, cross_population_cor, num_components, scale=1):
    scale_mat = _corr_mat(P, cross_population_cor, scale)
    ret_val = scipy.stats.wishart.rvs(P, scale_mat, num_components) / P
    ret_val = ret_val.reshape((num_components, P, P))
    signs, _ = np.linalg.slogdet(ret_val)
    assert np.all(signs == 1)
    return ret_val


if __name__ == '__main__':
    _main()
