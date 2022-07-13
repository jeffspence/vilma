"""
Utilities for inspecting and analyzing LD schema
"""
import logging
import pandas as pd
import numpy as np
import vilma


def args(super_parser):
    """Build command line arguments"""
    parser = super_parser.add_parser(
        'check_ld_schema',
        description='Utilities for analyzing LD schema.',
        usage='vilma check_ld_schema <options>',
    )
    parser.add_argument('--listvars',
                        required=False,
                        type=str,
                        default='',
                        help='Path at which to print a list of '
                             'all variants present in this schema.')
    parser.add_argument('--trace',
                        required=False,
                        type=str,
                        default='',
                        help='Path at which to print information '
                             'about the trace of the low rank approximation '
                             'of the LD matrix relative to its size. '
                             'See also the options '
                             '--trace-ldthresh and --trace-annotations.')
    parser.add_argument('--trace-ldthresh',
                        required=False,
                        type=float,
                        default=-1.,
                        help='Threshold for singular value approximation of '
                             'LD matrix. Setting --trace-ldthresh x '
                             'guarantees that SNPs with an r^2 of x or larger '
                             'will be linearly independent. So ldthresh of 1 '
                             'will result in no singular value thresholding.')
    parser.add_argument('--trace-annotations',
                        required=False,
                        type=str,
                        default='',
                        help='Path to an annotations file. If provided, only '
                             'SNPs that are both in the annotations file '
                             'and the LD matrix will be considered. If '
                             'provided, then in addtion to the overall trace '
                             'of the low rank approximation to the LD matrix, '
                             'the trace for each annotation will be '
                             'provided.')
    parser.add_argument('--ld-schema',
                        required=True,
                        type=str,
                        help='Path to LD panel schema.')
    parser.add_argument('--trace-mmap',
                        dest='mmap',
                        action='store_true',
                        help='Store the LD matrix on disk instead of in '
                             'memory when computing the trace. '
                             'This will result in substantially '
                             'longer runtimes, but lower memory usage.')
    parser.add_argument('--trace-extract',
                        required=False,
                        type=str,
                        default='',
                        help='List of SNPs to include in trace analysis, '
                             'with ID, A1, and A2 columns.')
    return parser


def compute_trace(block_ld_mat, one_hot_annotations):
    """
    Computes the trace and per-annotation traces of an LD matrix

    If annotations are provided then a trace is additionally computed for the
    submatrix obtained by restricting to each annotation. The results are
    returned as a pandas DataFrame

    args:
        block_ld_matrix: A BlockDiagonalMatrix representing the LD matrix
        one_hot_annotations: A num_snps x num_annotations matrix that one-hot
            encodes the annotation for each SNP (i.e., if row i has a one in
            column k then SNP i has the k^th annotations).

    returns:
        A pandas DataFrame with the various traces and the number of
        SNPs overall (and within each annotation) that are also in the LD
        matrix.
    """
    ld_diags = block_ld_mat.diag()
    total_trace = ld_diags.sum()
    total_snps = block_ld_mat.shape[0] - len(block_ld_mat.missing)

    trace_summary = pd.DataFrame(
        {'annotation': ['all_snps'],
         'trace': [total_trace],
         'num_snps': [total_snps],
         'ratio': [total_trace/total_snps]}
    )

    if not np.all(one_hot_annotations.sum(axis=1) == 1):
        raise ValueError('one_hot_annotations must be '
                         'one-hot encoded.')

    if one_hot_annotations.shape[1] > 1:
        not_missing = np.ones(ld_diags.shape[0])
        not_missing[block_ld_mat.missing] = 0.
        annotation_snps = not_missing.dot(one_hot_annotations)
        annotation_trace = ld_diags.dot(one_hot_annotations)
        annotation_labels = ['annotation_' + str(i) for i in
                             range(one_hot_annotations.shape[1])]
        trace_summary = pd.concat(
            [trace_summary,
             pd.DataFrame({'annotation': annotation_labels,
                           'trace': annotation_trace,
                           'num_snps': annotation_snps,
                           'ratio': annotation_trace/annotation_snps})],
            axis=0,
            ignore_index=True
        )

    return trace_summary


def combine_vars(ld_schema):
    """
    Build a DataFrame with all of the SNPs in an LD schema

    args:
        ld_schema: The path the an LD manifest

    returns:
        A pandas DataFrame with the metadata for all SNPs in an LD schema
    """
    all_vars = []
    for snp_path, _ in vilma.load.schema_iterator(ld_schema):
        all_vars.append(pd.read_csv(snp_path,
                                    header=None,
                                    delim_whitespace=True,
                                    names=['ID', 'CHROM', 'BP',
                                           'CM', 'A1', 'A2']))
    all_vars = pd.concat(all_vars, ignore_index=True)
    return all_vars


def main(args):
    # Check argument compatibility
    if args.trace_annotations and not args.trace:
        raise ValueError('If --trace-annotations is provided then '
                         '--trace must also be provided.')
    if args.trace_ldthresh != -1 and not args.trace:
        raise ValueError('If --trace-ldthresh is provided then '
                         '--trace must also be provided.')
    if not args.trace and not args.listvars:
        raise ValueError('If neither --trace nor --listvars '
                         'are provided, then this command does '
                         'nothing.')

    logging.info('Collecting list of variants in LD Schema.')
    all_vars = combine_vars(args.ld_schema)
    if args.trace:
        logging.info('Computing trace statistics.')

        if args.trace_extract:
            variants = vilma.load.load_variant_list(args.trace_extract)
        else:
            variants = all_vars.copy()

        annotations, denylist = vilma.load.load_annotations(
            args.trace_annotations, variants
        )

        ld_mat = vilma.load.load_ld_from_schema(
            args.ld_schema,
            variants=variants,
            denylist=denylist,
            ldthresh=args.ldthresh,
            mmap=args.mmap
        )
        trace_summary = compute_trace(
            ld_mat,
            annotations
        )
        trace_summary.to_csv(args.trace, sep='\t', index=False)

    if args.listvars:
        logging.info('Saving list of variants')
        all_vars.to_csv(args.listvars, sep='\t', index=False)
