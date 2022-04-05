import plinkio.plinkfile
import numpy as np
import pandas as pd
import logging
from argparse import ArgumentParser


def _arguments():
    parser = ArgumentParser()
    parser.add_argument('-o', '--out-root', required=True, type=str,
                        help='Path for output schema')
    parser.add_argument('-b', '--block-file', required=True, type=str,
                        help='Bed file containing LD block boundaries')
    parser.add_argument('-p', '--plink-file', required=True, type=str,
                        help='Plink format data containing genotypes')
    return parser.parse_args()


def _get_ld_blocks(bedfile_name):
    ld_table = pd.read_table(bedfile_name, names=['chrom', 'start', 'end'],
                             comment='#')
    ld_table_dict = {}
    # split by chromsome
    for chrom in np.unique(ld_table['chrom']):
        sub_table = ld_table.iloc[ld_table['chrom'] == chrom]
        # sort
        sub_table = sub_table.sort_values(by='end', ignore_index=True)

        # make sure no overlap
        assert np.all(sub_table.start.to_numpy()[1:]
                      >= sub_table.end.to_numpy()[:-1])

        ld_table_dict[chrom] = sub_table

    return ld_table_dict


def _process_blocks(blocked_data, blocked_ids, outfile_name):
    outpath = outfile_name + '_%s:%d'
    var_outpath = outfile_name + '_%s:%d.var'
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

    with open(outpath + '.schema', 'w') as ofh:
        ofh.write('\n'.join(legend))


def _assign_to_blocks(blocks, plink_data):
    blocked_data = {}
    blocked_ids = {}
    for locus, row in zip(plink_data, plink_data.loci()):
        if locus.chromosome not in blocks:
            continue
        b = np.searchsorted(blocks[locus.chromosome].start,
                            locus.bp_position - 1,   # Plink is 1-indexed
                            side='right') - 1
        # check i SNP is before first block
        if b < 0:
            continue
        # check if past the end of the block
        if locus.bp_position >= blocks[locus.chromosome].end[b]:
            continue

        key_str = '{} {}'.format(locus.chromosome, b)
        if key_str not in blocked_data:
            blocked_data[key_str] = []
            blocked_ids[key_str] = []
        blocked_data[key_str].append(
            np.array([[e if e <= 2.1 else np.nan for e in row]])
        )
        blocked_ids[key_str].append(
            [locus.name, locus.chromosome, locus.bp_position,
             locus.position, locus.allele1, locus.allele2]
        )

    # concatenate everything
    for key in blocked_data:
        block_gts = np.concatenate(blocked_data[key], axis=0)
        block_gts = pd.DataFrame(block_gts)
        blocked_data[key] = {'SNPs': block_gts,
                             'IDs': blocked_ids[key]}
    return blocked_data


def _main():
    args = _arguments()

    logging.info('Reading LD blocks from %s' % args.b)
    ld_blocks = _get_ld_blocks(args.b)

    logging.info('Reading genetic data from %s' % args.p)
    plink_data = plinkio.plinkfile.open(args.p)

    logging.info('Assigning SNPs to blocks')
    blocked_data = _assign_to_blocks(ld_blocks, plink_data)

    logging.info('Processing LD blocks...')
    _process_blocks(blocked_data, args.o)

    logging.info('Done!')


if __name__ == '__main__':
    _main()
