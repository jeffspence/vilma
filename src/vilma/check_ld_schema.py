"""
Utilities for inspecting and analyzing LD schema
"""
import logging
import pandas as pd
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
                        help='Path at which to print  a list of '
                             'all variants present in this schema.')
    parser.add_argument('--ld-schema',
                        required=True,
                        type=str,
                        help='Path to LD panel schema.')
    return parser


def combine_vars(ld_schema):
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
    if args.listvars:
        logging.info('Collecting list of variants in LD Schema.')
        all_vars = combine_vars(args.ld_schema)
        logging.info('...saving')
        all_vars.to_csv(args.listvars, sep='\t', index=False)
