"""
This module is cxflow framework entry point.

The entry point shall be accessed from command line via `cxflow` command.

At the moment cxflow allows to
- split data to x-validation sets with `cxflow split ...`
- train a network with `cxflow train ...`
- resume training with `cxflow resume ...`
- generate model predictions with `cxflow predict ...`

Run `cxflow -h` for details.
"""

import logging
import os
import sys

from cxflow.cli import train, resume, predict, split, grid_search, get_cxflow_arg_parser
from .constants import CXF_LOG_FORMAT, CXF_LOG_DATE_FORMAT


def entry_point() -> None:
    """cxflow entry point for training and dataset splitting."""

    # make sure the path contains the current working directory
    sys.path.insert(0, os.getcwd())

    parser = get_cxflow_arg_parser()

    # parse CLI arguments
    known_args, unknown_args = parser.parse_known_args()

    # show help if no subcommand was specified.
    if not hasattr(known_args, 'subcommand'):
        parser.print_help()
        quit(1)

    # set up global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG if known_args.verbose else logging.INFO)
    logger.handlers = []  # remove default handlers

    # set up STDERR handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter(CXF_LOG_FORMAT, datefmt=CXF_LOG_DATE_FORMAT))
    logger.addHandler(stderr_handler)

    if known_args.subcommand == 'train':
        train(config_file=known_args.config_file, cl_arguments=unknown_args, output_root=known_args.output_root)

    elif known_args.subcommand == 'resume':
        resume(config_path=known_args.config_path, restore_from=known_args.restore_from, cl_arguments=unknown_args,
               output_root=known_args.output_root)

    elif known_args.subcommand == 'predict':
        predict(config_path=known_args.config_path, restore_from=known_args.restore_from, cl_arguments=unknown_args,
                output_root=known_args.output_root)

    elif known_args.subcommand == 'split':
        split(config_file=known_args.config_file, num_splits=known_args.num_splits, train_ratio=known_args.ratio[0],
              valid_ratio=known_args.ratio[1], test_ratio=known_args.ratio[2])

    elif known_args.subcommand == 'gridsearch':
        grid_search(script=known_args.script, params=known_args.params, dry_run=known_args.dry_run)


if __name__ == '__main__':
    entry_point()
