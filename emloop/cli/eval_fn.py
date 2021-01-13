import logging
import os.path as path

from typing import Optional, Iterable

from ..api import create_emloop_training, delete_output_dir
from .util import validate_config, find_config, print_delete_warning
from ..utils.config import load_config


def evaluate(model_path: str, eval_target: str, config_path: Optional[str], cl_arguments: Iterable[str],
             output_root: str, delete_dir: bool, output_dir: Optional[str]=None) -> int:
    """
    Evaluate the given model.
    If `stream_names` are stated in config, then evaluation runs on those data streams.
    Otherwise evaluation runs on data stream specified by `eval_target`.

    Configuration is updated by the respective predict.stream_name section, in particular:
        - hooks section is entirely replaced
        - model and dataset sections are updated

    :param model_path: path to the model to be evaluated
    :param eval_target: evaluation target name
    :param config_path: path to the config to be used, if not specified infer the path from ``model_path``
    :param cl_arguments: additional command line arguments which will update the configuration
    :param output_root: output root in which the training directory will be created
    :param delete_dir: if True, delete output directory after evaluation finishes
    :param output_dir: if specified new dir will be created with path `output_root`/`output_dir`
    :return: exit code of the run
    """
    emloop_training = None
    exit_code = 0

    try:
        if delete_dir:
            print_delete_warning()
        if model_path.endswith('.yaml'):
            config_path = model_path
            model_dir = path.dirname(model_path)
        else:
            model_dir = path.dirname(model_path) if not path.isdir(model_path) else model_path
            config_path = find_config(model_dir if config_path is None else config_path)

        config = load_config(config_file=config_path, additional_args=cl_arguments,
                             override_stream=eval_target)
        stream_names = [eval_target]
        if 'eval' in config and eval_target in config['eval'] and 'stream_names' in config['eval'][eval_target]:
            stream_names = config['eval'][eval_target]['stream_names']
        logging.info('Will evaluate on streams `%s`', stream_names)

        validate_config(config)

        logging.debug('\tLoaded config: %s', config)
        emloop_training = create_emloop_training(config, output_root, model_path, output_dir)
        for stream_name in stream_names:
            logging.info('Starting evalutation on `%s`', stream_name)
            emloop_training.main_loop.run_evaluation(stream_name)
    except (Exception, AssertionError) as ex:  # pylint: disable=broad-except
        logging.error('Evaluation failed')
        logging.exception('%s', ex)
        exit_code = 1
    finally:
        if delete_dir:
            delete_output_dir(emloop_training.output_dir)

    return exit_code
