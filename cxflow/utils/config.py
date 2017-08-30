"""
Config module provides util functions for loading and dumping yaml configurations.
"""
import typing
from os import path

import yaml
import ruamel.yaml  # pylint: disable=import-error

from ..constants import CXF_CONFIG_FILE


def parse_arg(arg: str) -> typing.Tuple[str, typing.Any]:
    """
    Parse CLI argument in format ``key=value`` to ``(key, value)``

    :param arg: CLI argument string
    :return: tuple (key, value)
    :raise: yaml.ParserError: on yaml parse error
    """
    assert '=' in arg, 'Unrecognized argument `{}`. [name]=[value] expected.'.format(arg)

    key = arg[:arg.index('=')]
    value = yaml.load(arg[arg.index('=') + 1:])

    return key, value


def load_config(config_file: str, additional_args: typing.Iterable[str]) -> dict:
    """
    Load config from YAML ``config_file`` and extend/override it with the given ``additional_args``.

    :param config_file: path the YAML config file to be loaded
    :param additional_args: additional args which may extend or override the config loaded from the file.
    :return: configuration as dict
    """

    with open(config_file, 'r') as file:
        config = ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)

    for key_full, value in [parse_arg(arg) for arg in additional_args]:
        key_split = key_full.split('.')
        key_prefix = key_split[:-1]
        key = key_split[-1]

        conf = config
        for key_part in key_prefix:
            conf = conf[key_part]
        conf[key] = value

    config = yaml.load(ruamel.yaml.dump(config, Dumper=ruamel.yaml.RoundTripDumper))
    return config


def config_to_file(config, output_dir: str, name: str=CXF_CONFIG_FILE) -> str:
    """
    Save the given config to the given path in YAML.

    :param config: configuration dict
    :param output_dir: target output directory
    :param name: target filename
    :return: target path
    """
    dumped_config_f = path.join(output_dir, name)
    with open(dumped_config_f, 'w') as file:
        yaml.dump(config, file)
    return dumped_config_f


def config_to_str(config: dict) -> str:
    """
    Return the given given config as YAML str.

    :param config: configuration dict
    :return: given configuration as yaml str
    """
    return yaml.dump(config)

__all__ = []
