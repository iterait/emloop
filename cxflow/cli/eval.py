import logging
import os.path as path

from typing import Optional, Iterable

from .util import fallback, validate_config, find_config
from .common import run
from ..utils.config import load_config
from ..constants import CXF_PREDICT_STREAM


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
        config = load_config(config_file=config_path, additional_args=cl_arguments)

        if stream_name == CXF_PREDICT_STREAM and stream_name in config:  # old style ``cxflow predict ...``
            logging.warning('Old style ``predict`` configuration section is deprecated and will not be supported, '
                            'use ``eval.predict`` section instead.')
            config['eval'] = {'predict': config['predict']}

        if 'eval' in config and stream_name in config['eval']:
            update_section = config['eval'][stream_name]
            for subsection in ['dataset', 'model', 'main_loop']:
                if subsection in update_section:
                    config[subsection].update(update_section[subsection])
            if 'hooks' in update_section:
                config['hooks'] = update_section['hooks']
            else:
                logging.warning('Config does not contain `eval.%s.hooks` section. '
                                'No hook will be employed during the evaluation.', stream_name)
                config['hooks'] = []

        validate_config(config)

        logging.debug('\tLoaded config: %s', config)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Loading config failed', ex)

    run(config=config, output_root=output_root, restore_from=model_path, eval=stream_name)


def predict(config_path: str, restore_from: Optional[str], cl_arguments: Iterable[str], output_root: str) -> None:
    """
    Run prediction from the specified config path.

    If the config contains a ``predict`` section:
        - override hooks with ``predict.hooks`` if present
        - update dataset, model and main loop sections if the respective sections are present

    :param config_path: path to the config file or the directory in which it is stored
    :param restore_from: backend-specific path to the already trained model to be restored from.
                         If ``None`` is passed, it is inferred from the configuration file location as the directory
                         it is located in.
    :param cl_arguments: additional command line arguments which will update the configuration
    :param output_root: output root in which the training directory will be created
    """
    config = None

    try:
        config_path = find_config(config_path)
        restore_from = restore_from or path.dirname(config_path)
        config = load_config(config_file=config_path, additional_args=cl_arguments)

        if 'predict' in config:
            for section in ['dataset', 'model', 'main_loop']:
                if section in config['predict']:
                    config[section].update(config['predict'][section])
            if 'hooks' in config['predict']:
                config['hooks'] = config['predict']['hooks']
            else:
                logging.warning('Config does not contain `predict.hooks` section. '
                                'No hook will be employed during the prediction.')
                config['hooks'] = []

        validate_config(config)

        logging.debug('\tLoaded config: %s', config)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Loading config failed', ex)

    run(config=config, output_root=output_root, restore_from=restore_from, eval='predict')
