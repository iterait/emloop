import sys
import logging
import os.path as path

from ..constants import CXF_CONFIG_FILE


def find_config(config_path: str) -> str:
    """
    Derive configuration file path from the given path and check its existence.

    The given path is expected to be either

    1. path to the file
    2. path to a dir, in such case the path is joined with ``CXF_CONFIG_FILE``

    :param config_path: path to the configuration file or its parent directory
    :return: validated configuration file path
    """
    if path.isdir(config_path):  # dir specified instead of config file
        config_path = path.join(config_path, CXF_CONFIG_FILE)
    assert path.exists(config_path), '`{}` does not exist'.format(config_path)
    return config_path


def validate_config(config: dict) -> None:
    """
    Assert the config contains both ``model`` and ``dataset`` sections.

    Additionally, warn if no hooks specified.

    :param config: configuration object
    """
    assert 'model' in config, '`model` not present in the config'
    assert 'dataset' in config, '`dataset` not present in the config'
    if 'hooks' not in config:
        logging.warning('\tNo hooks found in config')


def fallback(message: str, ex: Exception) -> None:
    """
    Fallback procedure when a cli command fails.

    :param message: message to be logged
    :param ex: Exception which caused the failure
    """
    logging.error('%s', message)
    logging.exception('%s', ex)
    sys.exit(1)
