"""
Utilities for building a block LD matrix from genotype data
"""
import os
import logging
import plinkio.plinkfile
from pathlib import Path
import numpy as np
import pandas as pd
from vilma import matrix_structures


def args(super_parser):
    """Build command line arguments"""
    parser = super_parser.add_parser(
        'make_ld_schema',
        description='Build a block diagonal LD matrix from '
                    'genotype data and store it in vilma format.',
        usage='vilma make_ld_schema <options>',
    )

    parser.add_argument('-o', '--out-root', required=True, type=str,
                        help='Path for output schema')
    parser.add_argument('-b', '--block-file', required=True, type=str,
                        help='Bed file containing LD block boundaries')
    parser.add_argument('-p', '--plink-file-list', required=True, type=str,
                        help='A file where each line is the basename of '
                             'plink format genotype data for a single '
                             'chromosome.')
    parser.add_argument('--extract', required=False, type=str, default='',
                        help='A file with a column ID that specifies '
                             'which SNPs to keep. If not specified '
                             'all variants will be included.')
    parser.add_argument('--ldthresh', required=False, type=float, default=-1,
                        help='Threshold for computing SVD. If negative '
                             'then no SVD is performed. If between 0 and 1 '
                             'then setting to x guarantees that SNPs with '
                             'r^2 greater than x will be linearly independent '
                             'in the resulting decomposition.')
    return parser


def _get_ld_blocks(bedfile_name):
    """Load LD block locations from a bed file"""
    ld_table = pd.read_csv(bedfile_name,
                           names=['chrom', 'start', 'end'],
                           comment='#',
                           delim_whitespace=True,
                           header=None,
                           dtype={'chrom': str, 'start': int, 'end': int})
    ld_table_dict = {}
    # split by chromsome
    for chrom in np.unique(ld_table['chrom']):
        sub_table = ld_table.loc[ld_table['chrom'] == chrom]
        # sort
        sub_table = sub_table.sort_values(by='end', ignore_index=True)

        # make sure no overlap
        if not np.all(sub_table.start.to_numpy()[1:]
                      >= sub_table.end.to_numpy()[:-1]):
            raise ValueError('Bedfile contains an overlapping interval')

        ld_table_dict[chrom] = sub_table

    return ld_table_dict


def _process_blocks(blocked_data, outfile_name, ldthresh=-1):
    """Take genetic data and variant IDs, compute correlation and write"""
    outpath = outfile_name + '_{}:{}'
    rel_outpath = outpath.split('/')[-1]
    var_outpath = outfile_name + '_{}:{}.var'
    rel_var_outpath = var_outpath.split('/')[-1]
    legend = []
    for key in blocked_data:
        logging.info('...computing correlations for block %s',
                     key)
        corrmat = blocked_data[key]['SNPs'].corr().to_numpy()
        if ldthresh >= 0:
            trunc_mat = matrix_structures.LowRankMatrix(X=corrmat,
                                                        t=ldthresh)
            corrmat = np.vstack([trunc_mat.u,
                                 trunc_mat.s.reshape((1, -1))])
        np.save(outpath.format(*key.split()), corrmat)
        with open(var_outpath.format(*key.split()), 'w') as ofh:
            for var in blocked_data[key]['IDs']:
                ofh.write('\t'.join(map(str, var)) + '\n')
        legend.append(rel_var_outpath.format(*key.split())
                      + '\t'
                      + (rel_outpath + '.npy').format(*key.split()))

    with open(outfile_name + '.schema', 'a') as ofh:
        ofh.write('\n'.join(legend) + '\n')


def _assign_to_blocks(blocks, plink_data, variants=None):
    """Pull genotype data from `plink_data` and assign SNPs to blocks"""
    blocked_data = {}
    blocked_ids = {}
    chromosome = None
    for locus, row in zip(plink_data.get_loci(), plink_data):
        if chromosome is None:
            chromosome = str(locus.chromosome)
            if chromosome not in blocks.keys():
                raise ValueError('Plink File contains a chromosome '
                                 'that is not in the bedfile.')
        if str(locus.chromosome) != chromosome:
            raise ValueError('Each plink file should contain exactly one '
                             'chromosome.')
        if variants and locus.name not in variants:
            continue
        block_idx = np.searchsorted(blocks[chromosome].start,
                                    locus.bp_position - 1,
                                    side='right') - 1
        # check if SNP is before first block
        if block_idx < 0:
            continue
        # check if past the end of the block
        if locus.bp_position > blocks[chromosome].end[block_idx]:
            continue

        key_str = '{} {}'.format(chromosome, block_idx)
        if key_str not in blocked_data:
            blocked_data[key_str] = []
            blocked_ids[key_str] = []
        blocked_data[key_str].append(
            np.array([[e if e <= 2.1 else np.nan for e in row]])
        )
        blocked_ids[key_str].append(
            [locus.name, chromosome, locus.bp_position,
             locus.position, locus.allele1, locus.allele2]
        )

    # concatenate everything
    for key, value in blocked_data.items():
        block_gts = np.concatenate(value, axis=0).T
        block_gts = np.array(block_gts, dtype=float)
        block_gts[block_gts == 3] = np.nan
        block_gts = pd.DataFrame(block_gts)
        blocked_data[key] = {'SNPs': block_gts,
                             'IDs': blocked_ids[key]}
    return blocked_data


def main(args):
    logging.info('Reading LD blocks from %s', args.block_file)
    ld_blocks = _get_ld_blocks(args.block_file)

    variants = None
    if args.extract:
        logging.info('Loading Variants from %s', args.extract)
        variants = pd.read_csv(args.extract,
                               delim_whitespace=True,
                               header=0)
        if 'ID' not in variants.columns:
            raise ValueError(args.extract + ' must contain '
                             'a column labeled ID')
        variants = set(variants['ID'])
    if os.path.exists(args.out_root + '.schema'):
        raise ValueError(args.out_root + '.schema already exists. '
                         'Please delete before running.')

    plink_path = Path(args.plink_file_list)
    with open(plink_path, 'r') as plink_manifest:
        for idx, line in enumerate(plink_manifest):
            logging.info('Working on plink file %d', idx + 1)
            logging.info('...reading data')
            fname = Path(plink_path.parents[0], line.strip())
            plink_data = plinkio.plinkfile.open(
                str(fname)
            )
            logging.info('...assigning SNPs to blocks')
            blocked_data = _assign_to_blocks(ld_blocks, plink_data, variants)

            logging.info('...processing LD blocks')
            _process_blocks(blocked_data,
                            args.out_root,
                            ldthresh=args.ldthresh)

    logging.info('Done!')
