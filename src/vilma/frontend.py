"""Sets up command line interface."""
import logging
from argparse import ArgumentParser
from vilma import VERSION
from vilma.make_ld_schema import main as make_ld_schema
from vilma.make_ld_schema import args as make_ld_schema_args
from vilma.check_ld_schema import main as check_ld_schema
from vilma.check_ld_schema import args as check_ld_schema_args
from vilma.vi_options import main as fit
from vilma.vi_options import args as fit_args

COMMANDS = {
    'make_ld_schema': {'cmd': make_ld_schema, 'parser': make_ld_schema_args},
    'check_ld_schema': {'cmd': check_ld_schema,
                        'parser': check_ld_schema_args},
    'fit': {'cmd': fit, 'parser': fit_args},
}


def main():
    """
    Takes command line input and calls appropriate vilma command.

    The available commands are:
        make_ld_schema: Build a block diagonal LD matrix and store it in the
            format needed by vilma.
        check_ld_schema: Utilities for inspecting and analyzing LD schema.
        fit: Fit a model to GWAS summary statistics and use that model to build
            polygenic scores.

    Calling vilma <command> --help will show the available options for each
    subcommand.
    """
    parser = ArgumentParser(
        description="""
                    vilma v%s uses variational inference to estimate variant
                    effect sizes from GWAS summary data while simultaneously
                    learning the overall distribution of effects.
                    """ % VERSION,
        usage='vilma <command> <options>'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='command')
    for cmd in COMMANDS:
        cmd_parser = COMMANDS[cmd]['parser'](subparsers)
        cmd_parser.add_argument(
            '--logfile',
            required=False,
            type=str,
            default='',
            help='File to store information about the vilma run. To print to '
                 'stdout use "-". Defaults to no logging.'
        )
        cmd_parser.add_argument(
            '--verbose',
            dest='verbose',
            action='store_true',
            help='Log all information (as opposed to just warnings)'
        )
    args = parser.parse_args()
    try:
        func = COMMANDS[args.command]['cmd']
    except KeyError:
        parser.print_help()
        exit()
    level = 10 if args.verbose else 30
    if args.logfile == '-':
        logging.basicConfig(level=level)
    elif args.logfile:
        logging.basicConfig(filename=args.logfile, level=level)
    func(args)


if __name__ == '__main__':
    main()
