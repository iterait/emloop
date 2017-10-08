import logging
import os.path as path

from typing import Iterable

from .util import fallback, validate_config, find_config
from .common import create_dataset, create_output_dir
from ..utils.config import load_config


def invoke_dataset_method(config_path: str, method_name: str, output_root: str, cl_arguments: Iterable[str]) -> None:
    """
    Create the specified dataset and invoke its specified method.

    :param config_path: path to the config file or the directory in which it is stored
    :param method_name: name of the method to be invoked on the specified dataset
    :param cl_arguments: additional command line arguments which will update the configuration
    :param output_root: output root in which the training directory will be created
    """

    config = dataset = method = output_dir = None

    try:
        config_path = find_config(config_path)
        config = load_config(config_file=config_path, additional_args=cl_arguments)
        assert 'dataset' in config, '`dataset` section not present in the config'
        logging.debug('\tLoaded config: %s', config)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Loading config failed', ex)

    try:
        dataset = create_dataset(config)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating dataset failed', ex)

    try:
        method = getattr(dataset, method_name)
    except AttributeError as ex:
        fallback('Method `{}` not found in the dataset'.format(method_name), ex)

    try:
        method()
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Exception occurred during method `{}` invocation'.format(method_name), ex)
