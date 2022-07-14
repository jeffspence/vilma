"""
Simulate GWAS summary data from a mixture-of-gaussians model
"""
import logging
import numpy as np
import pandas as pd
import vilma
import pickle
import scipy.linalg


def args(super_parser):
    """Build command line arguments"""
    parser = super_parser.add_parser(
        'sim',
        description='Simulate GWAS summary data from a mixture-of-gaussians '
                    'model.',
        usage='vilma sim <options>',
    )
    parser.add_argument('--sumstats',
                        required=True,
                        type=str,
                        help='Comma-separated paths to summary statistics.')
    parser.add_argument('--covariance',
                        required=True,
                        type=str,
                        help='Path to .pkl file containing the covariance '
                             'matrices for each Gaussian component.')
    parser.add_argument('--weights',
                        required=True,
                        type=str,
                        help='Path to a .npy file containing a matrix of '
                             'weights (num_annotations x num_components) '
                             'to assign to each mixture component '
                             'with covariances specified by --covariance. '
                             'Alternatively, can be a .npz file containing '
                             'a fitted vilma model.')
    parser.add_argument('--gwas-n-scaling',
                        required=False,
                        type=str,
                        default='1.',
                        help='Comma-separated list of values to use to scale '
                             'the sample sizes for each cohort.  E.g., '
                             '--gwas-n-scaling 2,2 will simulate data for two '
                             'cohorts with GWAS sample sizes 2x larger than '
                             'the sample sizes used to generate the sumstats '
                             'files provided to --sumstats.')
    parser.add_argument('--annotations',
                        type=str,
                        default='',
                        help='Path to annotations file.')
    parser.add_argument('--output',
                        required=True,
                        type=str,
                        help='Output path prefix.')
    parser.add_argument('--names',
                        type=str,
                        required=False,
                        help='Comma-separated names of the populations for '
                             'the output. Defaults to 0, 1, ...')
    parser.add_argument('--ld-schema',
                        required=True,
                        type=str,
                        help='Comma-separated paths to summary statistics.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Seed for random number generation.')
    return parser


def sim_components(annotations, weights):
    """
    Simulates a one-hot encoded matrix with row-specific weights

    Specifically, row i has a one in column j with probability
    weights[annotations[i], j].

    args:
        annotations: A num_snps x num_annotations numpy array that one-hot
            encodes which annotation each SNP has.
        weights: A num_annotations x num_components numpy array, where each row
            is the distribution over mixture components for an annotation.

    returns:
        A num_snps x num_components numpy array that one-hot encodes which
        mixture component was drawn for each SNP.
    """
    to_return = np.zeros((annotations.shape[0], weights.shape[1]))
    for i in range(annotations.shape[0]):
        this_annotation = np.where(annotations[i] == 1)[0][0]
        comp_idx = np.random.choice(weights.shape[1],
                                    p=weights[this_annotation])
        to_return[i, comp_idx] = 1
    return to_return


def sim_true_effects(annotations, weights, cov_mats):
    """
    Simulates variables from a mixture of multivariate gaussians model

    args:
        annotations: A num_snps x num_annotations numpy array that one-hot
            encodes which annotation each SNP has.
        weights: A num_annotations x num_components numpy array, where each row
            is the distribution over mixture components for an annotation.
        cov_mats: A num_components x num_pops x num_pops numpy array, where
            [k, :, :] is the covariance matrix of the k'th Gaussian mixture
            component.

    returns:
        A num_pops x num_snps numpy array with effect sizes drawn independently
        from the mixture of zero-mean gaussians model, with mixture weights
        determined by each SNP's annotation and that annotation's mixture
        weights.
    """
    num_pops = cov_mats.shape[-1]
    one_hot_components = sim_components(annotations, weights)
    latent_effects = np.random.normal(
        loc=0, scale=1, size=(annotations.shape[0], num_pops)
    )

    # we need "square roots" of the covariance matrices
    # is X is covariance matrix, we can actually do cholesky
    # X = LL^T, then if v ~ N(0, I) we have
    # Lv ~ N(0, LL^T) = N(0, X)
    sqrt_covs = np.array(
        [scipy.linalg.cholesky(mat, lower=True) for mat in cov_mats]
    )
    true_effects = np.einsum('ip,ik,kqp->qi',
                             latent_effects,
                             one_hot_components,
                             sqrt_covs)
    return true_effects


def sim_gwas(true_beta, std_errs, ld_mat):
    """
    Simulate GWAS estimates from true effect sizes

    args:
        true_beta: A num_snps numpy array containing the true, causal effect of
            each SNP.
        std_errs: A num_snps numpy array containing the standard error of the
            GWAS for each SNP.
        ld_mat: A BlockDiagonalMatrix representing the LD matrix

    returns:
        A num_snps numpy array containing the beta hats that would be estimated
        from a GWAS.
    """
    mean = std_errs * (ld_mat.dot(true_beta/std_errs))
    latent_noise = np.random.normal(loc=0,
                                    scale=1,
                                    size=true_beta.shape[0])
    true_noise = std_errs * (ld_mat.matrix_power(0.5)).dot(latent_noise)
    return mean + true_noise


def main(args):
    # Set seed
    np.random.seed(args.seed)

    # Load names
    num_pops = len(args.sumstats.split(','))
    names = list(map(str, range(num_pops)))
    if args.names is not None:
        if args.names.count(',') != args.sumstats.count(','):
            raise ValueError('If --names are provided, one must be '
                             'provided per sumstat file.')
        names = args.names.split(',')

    # Load GWAS N scales
    n_scales = np.ones(num_pops)
    n_scales[:] = np.array(list(map(float, args.gwas_n_scaling.split(','))))
    if not np.all(n_scales > 0):
        raise ValueError('--gwas-n-scaling must be all positive.')

    # Create variants file
    all_vars = []
    for sstats_file in args.sumstats.split(','):
        all_vars.append(vilma.load.load_variant_list(sstats_file))
    all_vars = pd.concat(all_vars, ignore_index=True).drop_duplicates(
        subset='ID', ignore_index=True
    )

    # Load annotations
    annotations, denylist = vilma.load.load_annotations(
        args.annotations, all_vars
    )
    num_annotations = annotations.shape[1]
    # allocate missing annotations proportionally
    annotation_proportions = annotations.sum(axis=0).astype(np.float64)
    annotation_proportions /= annotation_proportions.sum()
    random_annots = np.random.choice(num_annotations,
                                     size=len(denylist),
                                     p=annotation_proportions,
                                     replace=True)
    annotations[denylist, :] = 0
    annotations[denylist, random_annots] = 1
    assert np.all(annotations.sum(axis=1) == 1)

    # Load sumstats (only need standard errors)
    # and LD matrix
    # this will set SE for missing data to be 1e-100
    std_errs = np.ones((num_pops, all_vars.shape[0])) * 1e-100
    ld_mats = []
    for idx, (sstats_file, n_scale, ld_schema_path) in enumerate(
        zip(args.sumstats.split(','),
            n_scales,
            args.ld_schema.split(','))):

        logging.info('Loading sumstats for population %s...', names[idx])
        these_sstats, missing = vilma.load.load_sumstats(
            sstats_file, all_vars
        )

        logging.info('Loading LD for population %s...', names[idx])
        ld_mat, this_missing_ld = vilma.load.load_ld_from_schema(
            ld_schema_path,
            variants=all_vars,
            denylist=missing,
            ldthresh=0.999999,
            mmap=True
        )
        ld_mats.append(ld_mat)
        keep_bool = np.ones(all_vars.shape[0], dtype=bool)
        keep_bool[missing] = False
        keep_bool[this_missing_ld] = False
        std_errs[idx, keep_bool] = (np.sqrt(n_scale)
                                    * these_sstats.SE.loc[keep_bool])

    # Load covariances
    with open(args.covariance, 'rb') as pickle_file:
        cov_mats = np.array(pickle.load(pickle_file)[0])

    # Load weights
    weights = np.load(args.weights)
    try:
        weights.files
        weights = weights['hyper_delta']
    except AttributeError:
        weights = np.array(weights)

    if weights.shape[0] != num_annotations:
        raise ValueError('The shape of the weights does not match the '
                         'number of annotations.')
    if weights.shape[1] != len(cov_mats):
        raise ValueError('The shape of the weights does not match the '
                         'number of covariance matrices.')
    if not np.allclose(weights.sum(axis=1), 1.):
        raise ValueError('weights do not sum to 1 within each annotation.')

    # simulate true effect sizes
    true_effects = sim_true_effects(annotations, weights, cov_mats)
    sim_beta_hat = np.zeros((num_pops, all_vars.shape[0]))
    for p, (ld_mat, beta, std_vec) in enumerate(zip(
        ld_mats, true_effects, std_errs
    )):
        sim_beta_hat[p] = sim_gwas(beta, std_vec, ld_mat)

    # Save results
    for p in range(num_pops):
        logging.info('Saving results for cohort', names[p])
        to_save = all_vars.copy()
        to_save['SE'] = std_errs[p]
        to_save['BETA'] = sim_beta_hat[p]
        to_save['true_beta'] = true_effects[p]
        to_save.loc[to_save.SE < 1e-99, 'SE'] = np.nan
        to_save = to_save.dropna()
        to_save.to_csv(args.output + '.' + names[p] + '.simgwas.tsv',
                       sep='\t',
                       index=False)
