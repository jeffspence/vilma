"""Constructs parsers for the command line interface. Calls optimizer"""
import logging
import itertools
import pickle
import numpy as np
from vilma.variational_inference import MultiPopVI
from vilma import load


def args(super_parser):
    parser = super_parser.add_parser(
        'fit',
        description='Use variational inference to learn '
                    'effect sizes and effect size distribution '
                    'from GWAS summary data.',
        usage='vilma fit <options>',
    )
    parser.add_argument('-K', '--components', default=12, type=int,
                        help='number of mixture components in prior')
    parser.add_argument('--num-its', default=1000, type=int,
                        help='Maximum number of optimization iterations.')
    parser.add_argument('--ld-schema', required=True, type=str,
                        help='Comma-separated paths to LD panel schemas.')
    parser.add_argument('--sumstats', required=True, type=str,
                        help='Comma-separated paths to summary statistics.')
    parser.add_argument('--stderrscale', default='1.0', type=str,
                        required=False,
                        help='Comma separated list of values to multiply'
                             'summary stat stderrs by.')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to annotation file.')
    parser.add_argument('--output', required=True, type=str,
                        help='Output path prefix.')
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
    parser.add_argument('--ldthresh', required=False, default=1.0,
                        help='Threhold for singular value approximation of '
                             'LD matrix. Setting --ldthresh x guarantees '
                             'that SNPs with an r^2 of x or larger will be '
                             'linearly independent. So ldthresh of 1 will '
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
    parser.add_argument('--load-checkpoint', type=str, default='', nargs=2,
                        help='Load a saved checkpoint from which to resume '
                             'optimization. To use, the first argument should '
                             'be a .npz file containing the checkpoint and '
                             'the second should be the .pkl file containing '
                             'the covariance matrices.',
                        metavar=('CHECKPOINT_FILE.npz', 'COVARIANCE_FILE.pkl'))
    return parser


def main(args):
    np.random.seed(args.seed)

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

    missing_annot = np.zeros(len(annotations), dtype=bool)
    missing_annot[denylist] = True

    missing_sumstats = np.zeros((len(annotations), num_pops), dtype=bool)
    missing_ld_info = np.zeros((len(annotations), num_pops), dtype=bool)

    combined_ld = []
    combined_betas = []
    combined_errors = []

    stderr_mult = np.zeros(len(args.sumstats.split(',')))
    stderr_mult[:] = list(map(float, args.stderrscale.split(',')))

    gwas_n = np.zeros_like(stderr_mult)
    gwas_n[:] = list(map(float, args.samplesizes.split(',')))

    init_hg = np.zeros_like(gwas_n)
    init_hg[:] = list(map(float, args.init_hg.split(',')))

    # Load LD matrices and sumstats
    if args.trait:
        raise NotImplementedError('--trait has not been implemented yet.')
        any_missing = None
        for idx, sumstats_path in enumerate(args.sumstats.split(',')):
            logging.info('Loading sumstats for trait %d...', (idx+1))
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
        ld_mat = load.load_ld_from_schema(args.ld_schema,
                                          variants=variants,
                                          denylist=any_missing,
                                          ldthresh=args.ldthresh,
                                          mmap=args.mmap)
        combined_ld.append(ld_mat)

    else:
        for idx, (ld_schema_path, sumstats_path) in enumerate(
                zip(args.ld_schema.split(','),
                    args.sumstats.split(','))):
            logging.info('Loading sumstats for population %d...', (idx+1))
            sumstats, missing = load.load_sumstats(sumstats_path,
                                                   variants=variants)
            missing_sumstats[missing, idx] = True
            missing.extend(denylist)
            combined_betas.append(np.array(sumstats.BETA).reshape((1, -1)))
            logging.info('Largest beta is... %f',
                         np.max(np.abs(np.array(sumstats.BETA))))
            combined_errors.append(np.array(sumstats.SE).reshape((1, -1))
                                   * stderr_mult[idx])

            logging.info('Loading LD for population %d...', (idx+1))
            ld_mat, this_missing_ld = load.load_ld_from_schema(
                ld_schema_path,
                variants=variants,
                denylist=missing,
                ldthresh=args.ldthresh,
                mmap=args.mmap
            )
            combined_ld.append(ld_mat)

            missing_ld_info[this_missing_ld, idx] = True

    logging.info('Largest beta is... %f', np.max(np.abs(combined_betas)))

    betas = np.concatenate(combined_betas, axis=0)
    std_errs = np.concatenate(combined_errors, axis=0)

    if args.load_checkpoint:
        with open(args.load_checkpoint[1], 'rb') as pfile:
            cross_pop_covs = pickle.load(pfile)[0]
    else:
        logging.info('Building cross-population covariances...')
        # First get out plausible maximum and minimum true effect sizes
        if args.scaled:
            maxes = np.nanmax((betas/std_errs)**2, axis=1)
            mins = np.zeros_like(maxes)
            for population in range(len(mins)):
                this_keep = betas[population, :]**2 > 0
                mins[population] = np.nanpercentile(
                   (betas[population, this_keep]
                    / std_errs[population, this_keep])**2,
                   2.5
                )
        else:
            maxes = np.zeros(betas.shape[0])
            mins = np.zeros_like(maxes)
            for population in range(len(mins)):
                keep = ~np.isnan(betas[population])
                this_beta = np.abs(betas[population, keep])
                this_se = std_errs[population, keep]
                psi = 1. / len(this_beta)
                probs = 1. / (1.
                              + ((1.-psi)/psi
                                 * np.sqrt(this_beta**2/this_se**2)
                                 * np.exp(-0.5*this_beta**2/this_se**2 + 0.5)))
                ebayes = np.maximum(this_beta**2 - this_se**2, 1e-10)
                raw_means = this_beta / (1. + this_se**2/ebayes**2)
                maxes[population] = np.max(probs*raw_means)**2
                mins[population] = np.nanpercentile(
                    betas[population, betas[population, :]**2 > 0]**2,
                    2.5
                )

        # build covariance matrices
        cross_pop_covs = _make_simple(num_pops, num_components, mins, maxes)

        # save covariance matrices
        with open('%s.covariance.pkl' % args.output, 'wb') as ofile:
            pickle.dump([cross_pop_covs], ofile)

    # run optimization
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
            checkpoint=(args.checkpoint_freq > 0),
            checkpoint_freq=args.checkpoint_freq,
            output=args.output,
            scaled=args.scaled,
            scale_se=args.scale_se,
            gwas_N=gwas_n,
            init_hg=init_hg,
            num_its=args.num_its,
        )

    checkpoint = None
    if args.load_checkpoint:
        checkpoint = np.load(args.load_checkpoint[0])
    params = elbo.optimize(checkpoint)

    # save model parameters
    to_save = elbo.create_dump_dict(params)
    to_save['vi_sigma'] = elbo.vi_sigma
    np.savez(args.output, **to_save)

    # write posterior means in plink format
    for name, posterior in zip(names, elbo.real_posterior_mean(*params)):
        variants['posterior_' + name] = posterior

    for name, pmv in zip(names, elbo.real_posterior_variance(*params)):
        variants['posterior_variance_' + name] = pmv

    if args.annotations:
        variants['missing_annotation'] = missing_annot

    for idx, name in enumerate(names):
        variants['missing_sumstats_' + name] = missing_sumstats[:, idx]
        variants['missing_LD_' + name] = missing_ld_info[:, idx]

    variants.to_csv(args.output + '.estimates.tsv', sep='\t', index=False)


def _make_diag_vals(num_pops, num_components, mins, maxes):
    """Build a grid of variances across the populations"""
    diag_vals = []
    # include something that's basically zero
    diag_vals = [[m*1e-6 for m in mins]]
    for k in range(num_components+1):
        this_diag = []
        for population in range(num_pops):
            this_diag.append(
                mins[population]
                * np.exp(np.log(maxes[population]/mins[population])
                         / num_components * k)
            )
        diag_vals.append(this_diag)
    return diag_vals


def _make_simple(num_pops, num_components, mins, maxes):
    """Build a grid of covariance matrices"""
    cross_pop_covs = []
    diag_vals = _make_diag_vals(num_pops, num_components, mins, maxes)
    if num_pops == 1:
        return list(np.array(diag_vals).reshape((num_components+2,
                                                 num_pops,
                                                 num_pops)))
    corr_vals = [-.99 + 1.98 * (k + 1) / num_components
                 for k in range(num_components)]
    for idx, diag in enumerate(diag_vals):
        for off_diags in itertools.product(
              *[corr_vals]*((num_pops*(num_pops-1))//2)):
            mat = np.eye(num_pops)
            mat[np.triu_indices_from(mat, k=1)] = off_diags
            mat.T[np.triu_indices_from(mat, k=1)] = off_diags
            mat = mat * np.sqrt(diag)
            mat = mat.T * np.sqrt(diag)
            for _ in range(3):
                scale = np.diag(
                    np.sqrt(np.exp(np.random.uniform(-1, 1, num_pops)))
                )
                cross_pop_covs.append(scale.dot(mat.dot(scale)))
        if idx > 0:
            # does population specific causals
            # correlation does not really matter
            for population in range(num_pops):
                single_pop = np.copy(diag_vals[0])
                single_pop[population] = diag[population]
                mat = np.diag(single_pop)
                for _ in range(3):
                    scale = np.diag(
                        np.sqrt(np.exp(np.random.uniform(-1, 1, num_pops)))
                    )
                    cross_pop_covs.append(scale.dot(mat.dot(scale)))

    return cross_pop_covs
