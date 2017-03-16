#!/usr/bin/python3 -mgrid_search

import argparse
import ast

import itertools
import logging
import os
import sys


def grid_search() -> None:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
                ret_code = os.system(command)
            except:
                logging.error('Command failed: %s', command)


def init_grid_search() -> None:
    sys.path.insert(0, os.getcwd())
    grid_search()


if __name__ == '__main__':
    init_grid_search()
