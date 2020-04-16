import sys
import logging
from typing import Iterable

from ..api import create_emloop_training, delete_output_dir
from .util import validate_config, find_config, print_delete_warning
from ..utils.config import load_config


def train(config_path: str, cl_arguments: Iterable[str], output_root: str, delete_dir: bool) -> int:
    """
    Load config and start the training.

    :param config_path: path to configuration file
    :param cl_arguments: additional command line arguments which will update the configuration
    :param output_root: output root in which the training directory will be created
    :param delete_dir: if True, delete output directory after training finishes
    :return: exit code of the run
    """
    emloop_training = None
    exit_code = 0
    try:
        if delete_dir:
            print_delete_warning()
        config_path = find_config(config_path)
        config = load_config(config_file=config_path, additional_args=cl_arguments)
        validate_config(config)
        logging.debug('\tLoaded config: %s', config)
        emloop_training = create_emloop_training(config, output_root)
        emloop_training.main_loop.run_training()
    except (Exception, AssertionError) as ex:  # pylint: disable=broad-except
        logging.error('Training failed')
        logging.exception('%s', ex)
        exit_code = 1
    finally:
        if delete_dir:
            delete_output_dir(emloop_training.output_dir)

    return exit_code
