import logging
import os.path as path

from typing import Optional, Iterable

from ..api import create_main_loop
from .util import fallback, validate_config, find_config
from ..utils.config import load_config


def evaluate(model_path: str, stream_name: str, config_path: Optional[str], cl_arguments: Iterable[str],
             output_root: str) -> None:
    """
    Evaluate the given model on the specified data stream.

    Configuration is updated by the respective predict.stream_name section, in particular:
        - hooks section is entirely replaced
        - model and dataset sections are updated

    :param model_path: path to the model to be evaluated
    :param stream_name: data stream name to be evaluated
    :param config_path: path to the config to be used, if not specified infer the path from ``model_path``
    :param cl_arguments: additional command line arguments which will update the configuration
    :param output_root: output root in which the training directory will be created
    """
    config = None

    try:
        model_dir = path.dirname(model_path) if not path.isdir(model_path) else model_path
        config_path = find_config(model_dir if config_path is None else config_path)
        config = load_config(config_file=config_path, additional_args=cl_arguments, override_stream=stream_name)

        validate_config(config)

        logging.debug('\tLoaded config: %s', config)
        
        main_loop = create_main_loop(config=config, output_root=output_root, restore_from=model_path)
        main_loop.run_evaluation(stream_name)

    except Exception as ex:  # pylint: disable=broad-except
        fallback('Evaluation failed', ex)
