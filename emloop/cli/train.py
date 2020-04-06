import logging

from typing import Iterable

from ..api import create_emloop_training, clean_output_dir
from .util import fallback, validate_config, find_config, print_delete_warning
from ..utils.config import load_config


def train(config_path: str, cl_arguments: Iterable[str], output_root: str, delete_dir: bool) -> None:
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
        emloop_training = create_emloop_training(config, output_root)
        if delete_dir:
            print_delete_warning()
        emloop_training.main_loop.run_training()
        clean_output_dir(emloop_training.output_dir, delete_dir)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Training failed', ex)
