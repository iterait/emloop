import logging

from typing import Iterable

from .util import fallback, validate_config, find_config
from .common import run
from ..utils.config import load_config


def train(config_path: str, cl_arguments: Iterable[str], output_root: str) -> None:
    """
    Load config and start the training.

    :param config_path: path to configuration file
    :param cl_arguments: additional command line arguments which will update the configuration
    :param output_root: output root in which the training directory will be created
    """
    config = None

    try:
        config_path = find_config(config_path)
        config = load_config(config_file=config_path, additional_args=cl_arguments)
        validate_config(config)
        logging.debug('\tLoaded config: %s', config)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Loading config failed', ex)

    run(config=config, output_root=output_root)
