"""Constructs parsers for the command line interface. Calls optimizer"""
import numpy as np
import logging
import argparse
import itertools

from natural_gradient import MultiPopVI
import load
import pickle


def _main():
    # Build command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--logfile', required=False, type=str, default='',
                        help='File to store information about the run. '
                             'To print to stdout use "-". Defaults to '
                             'no logging.')
    parser.add_argument('-K', '--components', default=12, type=int,
                        help='number of mixture components in prior'),
    parser.add_argument('--num-its', default=1000, type=int,
                        help='Maximum number of optimization iterations.'),
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
    parser.add_argument('--names', type=str, required=False,
                        help='Comma-separated names of the '
                             'populations for output. Defaults to '
                             '0, 1,... ')
    parser.add_argument('--extract', required=True, type=str,
                        help='List of SNPs to include in analysis, '
                             'with ID, A1, and A2 columns.')
    parser.add_argument('--scaled', dest='scaled', action='store_true',
                        help='Causes vilma to place a prior on '
                             'frequency-scaled effect sizes intead of '
                             'on effect sizes in their natural scaling.')
    parser.add_argument('--ldthresh', required=False, default=0.0,
                        help='Threhold for singular value approximation of '
                             'LD matrix. Setting --ldthresh x guarantees '
                             'that SNPs with an r^2 of x or larger will be '
                             'linearly independent. So ldthresh of 0 will '
                             'result in no singular value thresholding.',
                        type=float)
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random number generation.')
    parser.add_argument('--mmap', dest='mmap', action='store_true',
                        help='Store the LD matrix on disk instead of '
                             'in memory.  This will result in substantially '
                             'longer runtimes, but lower memory usage.')
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
                             'particularly accurate. They are just for '
                             'initializing the optimization algorithm.')
    parser.add_argument('--trait', dest='trait', action='store_true',
                        help='Consider different sumstats files as '
                             'different traits instead of different '
                             'populations. Currently unimplemented.')
    parser.add_argument('--checkpoint-freq', type=int, default=-1,
                        help='Store the model once every this many '
                             'iterations. Defaults to no checkpointing.')

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Set up logging
    if args.logfile == '-':
        logging.basicConfig(level=50)
    elif args.logfile:
        logging.basicConfig(filename=args.logfile, level=50)

    if (not args.trait
            and args.ld_schema.count(',') != 1
            and args.ld_schema.count(',') != args.sumstats.count(',')):
        raise ValueError('Either need to imput one ld_schema or provide a '
                         'sumstats file for each ld_schema.')

    num_pops = args.sumstats.count(',') + 1
    num_components = args.components

    # Get names, or set to 0, 1, ...
    names = list(map(str, range(num_pops)))
    if args.names is not None:
        if args.names.count(',') != args.sumstats.count(','):
            raise ValueError('If --names are provided, one must be '
                             'provided per sumstat file.')
        names = args.names.split(',')

    logging.info('Loading variants...')
    variants = load.load_variant_list(args.extract)

    logging.info('Loading annotations...')
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

    # Load LD matrices and sumstats
    if args.trait:
        raise NotImplementedError('--trait has not been implemented yet.')
        any_missing = None
        for idx, sumstats_path in enumerate(args.sumstats.split(',')):
            logging.info('Loading sumstats for trait %d...' % (idx+1))
            sumstats, missing = load.load_sumstats(sumstats_path,
                                                   variants=variants)
            if any_missing is None:
                any_missing = missing.tolist()
            else:
                any_missing.extend(missing.tolist())

            combined_betas.append(np.array(sumstats.BETA).reshape((1, -1)))
            logging.info('Largest beta is... %f',
                         np.max(np.abs(np.array(sumstats.BETA))))
            combined_errors.append(np.array(sumstats.SE).reshape((1, -1))
                                   * stderr_mult[idx])
        logging.info('Loading LD')
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
            logging.info('Loading sumstats for population %d...' % (idx+1))
            sumstats, missing = load.load_sumstats(sumstats_path,
                                                   variants=variants)
            missing.extend(denylist)
            combined_betas.append(np.array(sumstats.BETA).reshape((1, -1)))
            logging.info('Largest beta is... %f',
                         np.max(np.abs(np.array(sumstats.BETA))))
            combined_errors.append(np.array(sumstats.SE).reshape((1, -1))
                                   * stderr_mult[idx])

            logging.info('Loading LD for population %d...' % (idx+1))
            ld = load.load_ld_from_schema(ld_schema_path,
                                          variants=variants,
                                          denylist=missing,
                                          t=args.ldthresh,
                                          mmap=args.mmap)
            combined_ld.append(ld)

    logging.info('Largest beta is... %f', np.max(np.abs(combined_betas)))

    betas = np.concatenate(combined_betas, axis=0)
    std_errs = np.concatenate(combined_errors, axis=0)

    logging.info('Building cross-population covariances...')
    if args.scaled:
        maxes = np.nanmax((betas/std_errs)**2, axis=1)
        mins = np.zeros_like(maxes)
        for p in range(len(mins)):
            this_keep = betas[p, :]**2 > 0
            mins[p] = np.nanpercentile(
               (betas[p, this_keep]/std_errs[p, this_keep])**2,
               2.5
            )
    else:
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

    cross_pop_covs = _make_simple(num_pops, num_components, mins, maxes)

    with open("%s.covariance.pkl" % args.output, "wb") as ofile:
        pickle.dump([cross_pop_covs], ofile)

    logging.info('Fitting...')

    if args.trait:
        raise NotImplementedError('--trait has not been implemented yet.')
    else:
        elbo = MultiPopVI(
            marginal_effects=betas,
            std_errs=std_errs,
            ld_mats=combined_ld,
            mixture_covs=cross_pop_covs,
            annotations=annotations,
            checkpoint_freq=args.checkpoint_freq,
            output=args.output,
            scaled=args.scaled,
            scale_se=args.scale_se,
            gwas_N=gwas_N,
            init_hg=init_hg,
            num_its=args.num_its,
        )
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
    return cross_pop_covs


if __name__ == '__main__':
    _main()
