"""
YAML module providing util functions for handling YAMLs.
"""
from os import path
import typing

import ruamel.yaml  # pylint: disable=import-error
import yaml


def load_yaml(yaml_file: str) -> typing.Any:
    """
    Load YAML from file.

    :param yaml_file: path to YAML file
    :return: content of the YAML as dict/list
    """
    with open(yaml_file, 'r') as file:
        return ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)


def yaml_to_file(data, output_dir: str, name: str) -> str:
    """
    Save the given object to the given path in YAML.

    :param data: dict/list to be dumped
    :param output_dir: target output directory
    :param name: target filename
    :return: target path
    """
    dumped_config_f = path.join(output_dir, name)
    with open(dumped_config_f, 'w') as file:
        yaml.dump(data, file, Dumper=ruamel.yaml.RoundTripDumper)
    return dumped_config_f


def yaml_to_str(data: dict) -> str:
    """
    Return the given given config as YAML str.

    :param data: configuration dict
    :return: given configuration as yaml str
    """
    return yaml.dump(data, Dumper=ruamel.yaml.RoundTripDumper)
