import os
import errno
import logging
import os.path as path
from datetime import datetime

from typing import Optional, Iterable

from .util import fallback
from ..datasets import AbstractDataset
from ..models import AbstractModel
from ..hooks import AbstractHook
from ..constants import CXF_LOG_FILE, CXF_HOOKS_MODULE, CXF_CONFIG_FILE, CXF_LOG_DATE_FORMAT, CXF_LOG_FORMAT
from ..utils.reflection import get_class_module, parse_fully_qualified_name, create_object
from ..utils.yaml import yaml_to_str, yaml_to_file
from ..utils import get_random_name
from ..utils.training_trace import TrainingTrace, TrainingTraceKeys
from ..main_loop import MainLoop


def create_output_dir(config: dict, output_root: str, default_model_name: str='Unnamed') -> str:
    """
    Create output_dir under the given ``output_root`` and
        - dump the given config to YAML file under this dir
        - register a file logger logging to a file under this dir

    :param config: config to be dumped
    :param output_root: dir wherein output_dir shall be created
    :param default_model_name: name to be used when `model.name` is not found in the config
    :return: path to the created output_dir
    """
    logging.info('Creating output dir')

    # create output dir
    model_name = default_model_name
    if 'name' not in config['model']:
        logging.warning('\tmodel.name not found in config, defaulting to: %s', model_name)
    else:
        model_name = config['model']['name']

    if not os.path.exists(output_root):
        logging.info('\tOutput root folder "%s" does not exist and will be created', output_root)
        os.makedirs(output_root)

    # keep trying to create new output dir until it succeeds
    # this is neccessary due to improbable yet possible output dir name conflicts
    while True:
        try:
            output_dir = path.join(output_root, '{}_{}_{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                                                                  model_name, get_random_name()))
            os.mkdir(output_dir)
            break
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise ex
    logging.info('\tOutput dir: %s', output_dir)

    # create file logger
    file_handler = logging.FileHandler(path.join(output_dir, CXF_LOG_FILE))
    file_handler.setFormatter(logging.Formatter(CXF_LOG_FORMAT, datefmt=CXF_LOG_DATE_FORMAT))
    logging.getLogger().addHandler(file_handler)

    return output_dir


def create_dataset(config: dict, output_dir: Optional[str]=None) -> AbstractDataset:
    """
    Create a dataset object according to the given config.

    Dataset config section and the `output_dir` are passed to the constructor in a single YAML-encoded string.

    :param config: config dict with dataset config
    :param output_dir: path to the training output dir or None
    :return: dataset object
    """
    logging.info('Creating dataset')

    dataset_config = config['dataset']
    assert 'class' in dataset_config, '`dataset.class` not present in the config'
    dataset_module, dataset_class = parse_fully_qualified_name(dataset_config['class'])

    if 'output_dir' in dataset_config:
        raise ValueError('The `output_dir` key is reserved and can not be used in dataset configuration.')

    dataset_config = {'output_dir': output_dir, **config['dataset']}
    del dataset_config['class']

    dataset = create_object(dataset_module, dataset_class, args=(yaml_to_str(dataset_config),))
    logging.info('\t%s created', type(dataset).__name__)
    return dataset


def create_model(config: dict, output_dir: str, dataset: AbstractDataset,
                 restore_from: Optional[str]=None) -> AbstractModel:
    """
    Create a model object either from scratch of from the checkpoint in ``resume_dir``.

    Cxflow allows the following scenarios

    1. Create model: leave ``restore_from=None`` and specify ``class``;
    2. Restore model: specify ``restore_from`` which is a backend-specific path to (a directory with) the saved model.

    :param config: config dict with model config
    :param output_dir: path to the training output dir
    :param dataset: dataset object implementing the :py:class:`cxflow.datasets.AbstractDataset` concept
    :param restore_from: from whence the model should be restored (backend-specific information)
    :return: model object
    """

    logging.info('Creating a model')

    model_config = config['model']
    assert 'class' in model_config, '`model.class` not present in the config'
    model_module, model_class = parse_fully_qualified_name(model_config['class'])

    # create model kwargs (without `class` and `name`)
    model_kwargs = {'dataset': dataset, 'log_dir': output_dir, 'restore_from': restore_from, **model_config}
    del model_kwargs['class']
    if 'name' in model_kwargs:
        del model_kwargs['name']

    try:
        model = create_object(model_module, model_class, kwargs=model_kwargs)
    except (ImportError, AttributeError) as ex:
        if restore_from is None:  # training case
            raise ImportError('Cannot create model from the specified model module `{}` and class `{}`.'.format(
                model_module, model_class)) from ex

        else:  # restore cases (resume, predict)
            logging.warning('Cannot create model from the specified model class `%s`.', model_config['class'])
            assert 'restore_fallback' in model_config, '`model.restore_fallback` not present in the config'
            logging.info('Trying to restore with fallback `%s` instead.', model_config['restore_fallback'])

            try:  # try fallback class
                fallback_module, fallback_class = parse_fully_qualified_name(model_config['restore_fallback'])
                model = create_object(fallback_module, fallback_class, kwargs=model_kwargs)
            except (ImportError, AttributeError) as ex:  # if fallback module/class specified but it fails
                raise ImportError('Cannot create model from the specified restore_fallback `{}`.'.format(
                    model_config['restore_fallback'],)) from ex

    logging.info('\t%s created', type(model).__name__)
    return model


def create_hooks(config: dict, model: AbstractModel,
                 dataset: AbstractDataset, output_dir: str) -> Iterable[AbstractHook]:
    """
    Create hooks specified in ``config['hooks']`` list.

    Hook config entries may be one of the following types:

    .. code-block:: yaml
        :caption: A hook with default args specified only by its name as a string; e.g.

        hooks:
          - LogVariables
          - cxflow_tensorflow.WriteTensorBoard

    .. code-block:: yaml
        :caption: A hook with custom args as a dict name -> args; e.g.

        hooks:
          - StopAfter:
              n_epochs: 10

    :param config: config dict
    :param model: model object to be passed to the hooks
    :param dataset: dataset object to be passed to hooks
    :param output_dir: training output dir available to the hooks
    :return: list of hook objects
    """
    logging.info('Creating hooks')
    hooks = []
    if 'hooks' in config:
        for hook_config in config['hooks']:
            if isinstance(hook_config, str):
                hook_config = {hook_config: {}}
            assert len(hook_config) == 1, 'Hook configuration must have exactly one key (fully qualified name).'

            hook_path, hook_params = next(iter(hook_config.items()))
            if hook_params is None:
                logging.warning('\t\t Empty config of `%s` hook', hook_path)
                hook_params = {}

            hook_module, hook_class = parse_fully_qualified_name(hook_path)
            # find the hook module if not specified
            if hook_module is None:
                hook_module = get_class_module(CXF_HOOKS_MODULE, hook_class)
                logging.debug('\tFound hook module `%s` for class `%s`', hook_module, hook_class)
                if hook_module is None:
                    raise ValueError('Can`t find hook module for hook class `{}`. '
                                     'Make sure it is defined under `{}` sub-modules.'
                                     .format(hook_class, CXF_HOOKS_MODULE))
            # create hook kwargs
            hook_kwargs = {'dataset': dataset, 'model': model, 'output_dir': output_dir, **hook_params}

            # create new hook
            try:
                hook = create_object(hook_module, hook_class, kwargs=hook_kwargs)
                hooks.append(hook)
                logging.info('\t%s created', type(hooks[-1]).__name__)
            except (ValueError, KeyError, TypeError, NameError, AttributeError, AssertionError, ImportError) as ex:
                logging.error('\tFailed to create a hook from config `%s`', hook_config)
                raise ex
    return hooks


def run(config: dict, output_root: str, restore_from: str=None, predict: bool=False) -> None:
    """
    Run **cxflow** training configured by the passed `config`.

    Unique ``output_dir`` for this training is created under the given ``output_root`` dir
    wherein all the training outputs are saved. The output dir name will be roughly ``[model.name]_[time]``.

    The training procedure consists of the following steps:

    1.  Set up (create output dir and file logger, dump the loaded config into the output dir)
    2.  Create dataset (YAML string with ``dataset`` and ``log_dir`` configs are passed to the dataset constructor)
    3.  Create (or restore) model (dataset, ``log_dir`` and model config is passed to the constructor)
    4.  Create all the training hooks
    5.  Create the ``MainLoop`` object
    6.  Run the main loop

    If any of the steps fails, the training is terminated.

    After the training procedure finishes, the output dir will contain the following:
        - ``train_log.txt`` with entry point and main loop logs (same as the stderr)
        - dumped YAML config

    Additional outputs created by hooks, dataset or tensorflow may include:
        - ``dataset_log.txt`` with info about dataset/stream creation
        - model checkpoint(s)
        - TensorBoard log file
        - TensorFlow event log

    :param config: configuration
    :param output_root: dir under which output_dir shall be created
    :param restore_from: from whence the model should be restored (backend-specific information)
    """

    output_dir = dataset = model = hooks = main_loop = None

    try:
        output_dir = create_output_dir(config=config, output_root=output_root)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Failed to create output dir', ex)

    try:
        dataset = create_dataset(config=config, output_dir=output_dir)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating dataset failed', ex)

    try:
        model = create_model(config=config, output_dir=output_dir, dataset=dataset, restore_from=restore_from)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating model failed', ex)

    try:  # save the config to file
        # modify the config so that it contains fallback information
        config['model']['restore_fallback'] = model.restore_fallback
        yaml_to_file(data=config, output_dir=output_dir, name=CXF_CONFIG_FILE)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Saving config failed', ex)

    try:
        hooks = create_hooks(config=config, model=model, dataset=dataset, output_dir=output_dir)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating hooks failed', ex)

    try:
        logging.info('Creating main loop')
        kwargs = config['main_loop'] if 'main_loop' in config else {}
        if predict:
            kwargs['extra_streams'] = []
        main_loop = MainLoop(model=model, dataset=dataset, hooks=hooks, **kwargs)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating main loop failed', ex)

    if predict:
        try:
            with main_loop:
                logging.info('Running the prediction')
                main_loop.run_prediction()
        except Exception as ex:  # pylint: disable=broad-except
            fallback('Running the prediction failed', ex)
    else:
        trace = TrainingTrace(output_dir)
        try:
            with main_loop:
                logging.info('Running the training')
                trace[TrainingTraceKeys.TRAIN_BEGIN] = datetime.now()
                main_loop.run_training(trace)
                trace[TrainingTraceKeys.EXIT_STATUS] = 0
        except Exception as ex:  # pylint: disable=broad-except
            trace[TrainingTraceKeys.EXIT_STATUS] = 1
            fallback('Running the training failed', ex)
        except SystemExit as ex:
            trace[TrainingTraceKeys.EXIT_STATUS] = ex.code
        finally:
            trace[TrainingTraceKeys.EPOCHS_DONE] = main_loop.epochs_done
            trace[TrainingTraceKeys.TRAIN_END] = datetime.now()
