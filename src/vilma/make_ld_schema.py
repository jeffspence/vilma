"""
Utilities for building a block LD matrix from genotype data
"""
import plinkio.plinkfile
import os
import numpy as np
import pandas as pd
import logging
from argparse import ArgumentParser


def _arguments():
    """Build command line arguments"""
    parser = ArgumentParser()
    parser.add_argument('-o', '--out-root', required=True, type=str,
                        help='Path for output schema')
    parser.add_argument('-b', '--block-file', required=True, type=str,
                        help='Bed file containing LD block boundaries')
    parser.add_argument('-p', '--plink-file-list', required=True, type=str,
                        help='A file where each line is the basename of '
                             'plink format genotype data for a single '
                             'chromosome.')
    return parser.parse_args()


def _get_ld_blocks(bedfile_name):
    """Load LD block locations from a bed file"""
    ld_table = pd.read_csv(bedfile_name, names=['chrom', 'start', 'end'],
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


def _process_blocks(blocked_data, outfile_name):
    """Take genetic data and variant IDs, compute correlation and write"""
    outpath = outfile_name + '_{}:{}'
    var_outpath = outfile_name + '_{}:{}.var'
    legend = []
    for key in blocked_data:
        corrmat = blocked_data[key]['SNPs'].corr().to_numpy()
        np.save(outpath.format(*key.split()), corrmat)
        with open(var_outpath.format(*key.split()), 'w') as ofh:
            for var in blocked_data[key]['IDs']:
                ofh.write('\t'.join(map(str, var)) + '\n')
        legend.append(var_outpath.format(*key.split())
                      + '\t'
                      + (outpath + '.npy').format(*key.split()))

    with open(outfile_name + '.schema', 'a') as ofh:
        ofh.write('\n'.join(legend))


def _assign_to_blocks(blocks, plink_data):
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
        b = np.searchsorted(blocks[chromosome].start,
                            locus.bp_position - 1,   # Plink is 1-indexed
                            side='right') - 1
        # check i SNP is before first block
        if b < 0:
            continue
        # check if past the end of the block
        if locus.bp_position >= blocks[chromosome].end[b]:
            continue

        key_str = '{} {}'.format(chromosome, b)
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
    for key in blocked_data:
        block_gts = np.concatenate(blocked_data[key], axis=0).T
        block_gts = np.array(block_gts, dtype=float)
        block_gts[block_gts == 3] == np.nan
        block_gts = pd.DataFrame(block_gts)
        blocked_data[key] = {'SNPs': block_gts,
                             'IDs': blocked_ids[key]}
    return blocked_data


def _main():
    args = _arguments()

    logging.info('Reading LD blocks from %s' % args.b)
    ld_blocks = _get_ld_blocks(args.b)

    if os.path.exists(args.o + '.schema'):
        raise ValueError(args.o + 'schema already exists. '
                         'Please delete before running.')

    with open(args.p, 'r') as plink_manifest:
        for idx, line in enumerate(plink_manifest):
            logging.info('Working on plink file %d', idx + 1)
            logging.info('...reading data')
            plink_data = plinkio.plinkfile.open(
                plink_manifest
            )
            logging.info('...assigning SNPs to blocks')
            blocked_data = _assign_to_blocks(ld_blocks, plink_data)

            logging.info('...processing LD blocks')
            _process_blocks(blocked_data, args.o)

    logging.info('Done!')


if __name__ == '__main__':
    _main()
