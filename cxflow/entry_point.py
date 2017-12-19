"""
This module is **cxflow** framework entry point.

The entry point shall be accessed from command line via `cxflow` command.

At the moment **cxflow** allows to
- train a model with ``cxflow train ...``
- resume training with ``cxflow resume ...``
- generate model predictions with ``cxflow predict ...``
- invoke dataset method with ``cxflow dataset <method> ...``

Run `cxflow -h` for details.
"""

import logging
import os
import sys

from cxflow.cli import train, resume, predict, grid_search, get_cxflow_arg_parser, invoke_dataset_method, \
    list_train_dirs
from cxflow.cli.prune import prune_train_dirs

from .constants import CXF_LOG_FORMAT, CXF_LOG_DATE_FORMAT


def entry_point() -> None:
    """**cxflow** entry point."""

    # make sure the path contains the current working directory
    sys.path.insert(0, os.getcwd())

    parser = get_cxflow_arg_parser(True)

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
        train(config_path=known_args.config_file, cl_arguments=unknown_args, output_root=known_args.output_root)

    elif known_args.subcommand == 'resume':
        resume(config_path=known_args.config_path, restore_from=known_args.restore_from, cl_arguments=unknown_args,
               output_root=known_args.output_root)

    elif known_args.subcommand == 'predict':
        predict(config_path=known_args.config_path, restore_from=known_args.restore_from, cl_arguments=unknown_args,
                output_root=known_args.output_root)

    elif known_args.subcommand == 'dataset':
        invoke_dataset_method(config_path=known_args.config_file, method_name=known_args.method,
                              cl_arguments=unknown_args, output_root=known_args.output_root)

    elif known_args.subcommand == 'gridsearch':
        grid_search(script=known_args.script, params=known_args.params, dry_run=known_args.dry_run)

    elif known_args.subcommand == 'ls':
        list_train_dirs(known_args.dir, known_args.recursive, known_args.all, known_args.long, known_args.verbose)

    elif known_args.subcommand == 'prune':
        prune_train_dirs(known_args.dir, known_args.epochs, known_args.subdirs)


if __name__ == '__main__':
    entry_point()
