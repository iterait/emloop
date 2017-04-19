#!/usr/bin/python3 -mgrid_search
"""
This module is deprecated. See issue #50 for details.
"""
import argparse
import ast

import itertools
import logging
import os
import sys

from cxflow.entry_point import CXF_LOG_FORMAT


def init_grid_search() -> None:
    """
    This method is deprecated. See issue #50 for details.
    """
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(format=CXF_LOG_FORMAT, level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('script', help='script to be grid-searched')
    parser.add_argument('--dry-run', action='store_true', help='print instead execute')
    known, unknown = parser.parse_known_args()

    param_space = {}
    for arg in unknown:
        assert '=' in arg

        name = arg[:arg.index('=')]
        options = arg[arg.index('=') + 1:]
        options = ast.literal_eval(options)
        assert isinstance(options, list), options

        param_space[name] = options

    param_names = param_space.keys()
    commands = []
    for values in itertools.product(*[param_space[name] for name in param_names]):
        command = str(known.script) + ' '
        for name, value in zip(param_names, values):
            command += str(name) + '=' + str(value) + ' '
        commands.append(command[:-1])

    if known.dry_run:
        logging.warning('Dry run')
        for command in commands:
            print(command)
    else:
        for command in commands:
            try:
                return_code = os.system(command)
            except Exception as _:  # pylint: disable=broad-except
                logging.error('Command `%s` failed with exit code %s.', command, return_code)


if __name__ == '__main__':
    init_grid_search()
