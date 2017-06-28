"""
This module handles grid search,
"""

import ast
from collections import OrderedDict
import itertools
import logging
import subprocess
import typing


def _build_grid_search_commands(script: str, params: typing.Iterable[str]) -> typing.Iterable[typing.List[str]]:
    """
    Build all grid search parameter configurations.

    :param script: String of command prefix, e.g. `cxflow train -v -o log`.
    :param params: Iterable collection of strings in standard cxflow param form, e.g. 'numerical_param:int=[1, 2]' or
                   'text_param:str=["hello", "cio"]'.
    """

    param_space = OrderedDict()
    for arg in params:
        assert '=' in arg

        name = arg[:arg.index('=')]
        options = arg[arg.index('=') + 1:]
        options = ast.literal_eval(options)
        assert isinstance(options, list), options

        param_space[name] = options

    param_names = param_space.keys()
    commands = []
    for values in itertools.product(*[param_space[name] for name in param_names]):
        command = str(script).split()
        for name, value in zip(param_names, values):
            command.append(str(name) + '="' + str(value) + '"')
        commands.append(command)

    return commands


def grid_search(script: str, params: typing.Iterable[str], dry_run: bool=False) -> None:
    """
    Build all grid search parameter configurations and optionally run them.

    :param script: String of command prefix, e.g. `cxflow train -v -o log`.
    :param params: Iterable collection of strings in standard cxflow param form, e.g. 'numerical_param:int=[1, 2]' or
                   'text_param:str=["hello", "cio"]'.
    :param dry_run: If set to true, the built commands will only printed instead of executed.
    """

    commands = _build_grid_search_commands(script=script, params=params)

    if dry_run:
        logging.warning('Dry run')
        for command in commands:
            logging.info(command)
    else:
        for command in commands:
            try:
                completed_process = subprocess.run(command)
                logging.info('Command `%s` completed with exit code %d', command, completed_process.returncode)
            except Exception as _:  # pylint: disable=broad-except
                logging.error('Command `%s` failed.', command)
