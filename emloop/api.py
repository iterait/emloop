import os
import logging
import os.path as path
from datetime import datetime
import copy
from typing import Optional, Iterable

from .datasets import AbstractDataset
from .models import AbstractModel
from .hooks import AbstractHook
from .hooks.training_trace import TrainingTrace
from .constants import EL_LOG_FILE, EL_HOOKS_MODULE, EL_CONFIG_FILE, EL_LOG_DATE_FORMAT, EL_LOG_FORMAT
from .utils.reflection import get_class_module, parse_fully_qualified_name, create_object
from .utils.yaml import yaml_to_str, yaml_to_file
from .utils import get_random_name
from .main_loop import MainLoop


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

    model_name = default_model_name
    if 'name' not in config['model']:
        logging.warning('\tmodel.name not found in config, defaulting to: %s', model_name)
    else:
        model_name = config['model']['name']

    if not os.path.exists(output_root):
        logging.info('\tOutput root folder "%s" does not exist and will be created', output_root)
        os.makedirs(output_root)

    output_dir_format = os.environ.get('EMLOOP_OUTPUT_DIR_FORMAT', '{model_name}_{random_name}')
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    random_name = get_random_name()
    output_dir_name = output_dir_format.format(timestamp=timestamp, model_name=model_name, random_name=random_name)

    suffix = 0
    while True:
        if suffix > 0:
            output_dir_name = output_dir_name + f'_{suffix}'

        output_dir = os.path.join(output_root, output_dir_name)
        try:
            os.mkdir(output_dir)
            break
        except FileExistsError:
            suffix += 1

    logging.info(f'Created output directory with name {output_dir}')
    logging.info('\tOutput dir: %s', output_dir)
    yaml_to_file(data=config, output_dir=output_dir, name=EL_CONFIG_FILE)

    # create file logger
    file_handler = logging.FileHandler(path.join(output_dir, EL_LOG_FILE))
    file_handler.setFormatter(logging.Formatter(EL_LOG_FORMAT, datefmt=EL_LOG_DATE_FORMAT))
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
    config = copy.deepcopy(config)
 
    dataset_config = dict(config)['dataset']
    assert 'class' in dataset_config, '`dataset.class` not present in the config'
    dataset_module, dataset_class = parse_fully_qualified_name(dataset_config['class'])

    if 'output_dir' in dataset_config:
        raise ValueError('The `output_dir` key is reserved and can not be used in dataset configuration.')

    dataset_config = {'output_dir': output_dir, **config['dataset']}
    del dataset_config['class']

    dataset = create_object(dataset_module, dataset_class, args=(yaml_to_str(dataset_config),))
    logging.info('\t%s created', type(dataset).__name__)

    return dataset


def create_model(config: dict, output_dir: Optional[str]=None, dataset: Optional[AbstractDataset]=None,
                 restore_from: Optional[str]=None) -> AbstractModel:
    """
    Create a model object either from scratch or from the checkpoint specified by ``restore_from``.

    Emloop allows the following scenarios

    1. Create model: leave ``restore_from=None`` and specify ``class``;
    2. Restore model: specify ``restore_from`` which is a backend-specific path to (a directory with) the saved model.

    :param config: config dict with model config
    :param output_dir: path to the training output dir
    :param dataset: dataset object implementing the :py:class:`emloop.datasets.AbstractDataset` concept
    :param restore_from: from whence the model should be restored (backend-specific information)
    :return: model object
    """
    logging.info('Creating a model')
    config = copy.deepcopy(config)
    model_config = config['model']

    assert 'class' in model_config, '`model.class` not present in the config'
    model_module, model_class = parse_fully_qualified_name(model_config['class'])

    # create model kwargs (without `class` and `name`)
    model_kwargs = {'dataset': dataset, 'log_dir': output_dir, 'restore_from': restore_from, **model_config}
    del model_kwargs['class']
    if 'name' in model_kwargs:
        del model_kwargs['name']

    model = create_object(model_module, model_class, kwargs=model_kwargs)
    logging.info('\t%s created', type(model).__name__)
    return model


def create_hooks(config: dict, model: Optional[AbstractModel]=None, dataset: Optional[AbstractDataset]=None,
                 output_dir: Optional[str]=None) -> Iterable[AbstractHook]:
    """
    Create hooks specified in ``config['hooks']`` list.

    Hook config entries may be one of the following types:

    .. code-block:: yaml
        :caption: A hook with default args specified only by its name as a string; e.g.

        hooks:
          - LogVariables
          - emloop_tensorflow.WriteTensorBoard

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
    hooks_config = copy.deepcopy(config.get('hooks', {}))
    hooks = []

    training_trace_created = False

    for hook_config in hooks_config:
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
            hook_module = get_class_module(EL_HOOKS_MODULE, hook_class)
            logging.debug('\tFound hook module `%s` for class `%s`', hook_module, hook_class)
            if hook_module is None:
                raise ModuleNotFoundError('Can`t find hook module for hook class `{}`. '
                                          'Make sure it is defined under `{}` sub-modules.'
                                          .format(hook_class, EL_HOOKS_MODULE))
        # create hook kwargs
        hook_kwargs = {'dataset': dataset, 'model': model, 'output_dir': output_dir, **hook_params}

        # create new hook
        try:
            hook = create_object(hook_module, hook_class, kwargs=hook_kwargs)
            hooks.append(hook)
            if isinstance(hook, TrainingTrace):
                training_trace_created = True

            logging.info('\t%s created', type(hooks[-1]).__name__)
        except (ValueError, KeyError, TypeError, NameError, AttributeError, AssertionError, ImportError) as ex:
            logging.error('\tFailed to create a hook from config `%s`', hook_config)
            raise ex

    if not training_trace_created:
        logging.warning('TrainingTrace hook added between hooks. Add it to your config.yaml to suppress this warning.')
        hooks.append(TrainingTrace(output_dir=output_dir))

    return hooks


def create_main_loop(config: dict, output_root: str, restore_from: str=None) -> MainLoop:
    """
    Creates :py:class:`MainLoop` with model, dataset and hooks according to config.

    :param config: config dict 
    :param output_root: dir where output_dir shall be created
    :param restore_from: if not None, from whence the model should be restored (backend-specific information)

    :return: main loop object
    """
    output_dir = dataset = model = hooks = main_loop = None

    output_dir = create_output_dir(config=config, output_root=output_root)
    dataset = create_dataset(config=config, output_dir=output_dir)
    model = create_model(config=config, output_dir=output_dir, dataset=dataset, restore_from=restore_from)
    hooks = create_hooks(config=config, model=model, dataset=dataset, output_dir=output_dir)
    logging.info('Creating main loop')
    main_loop_kwargs = copy.deepcopy(config.get('main_loop', {}))
    main_loop = MainLoop(model=model, dataset=dataset, hooks=hooks, **main_loop_kwargs)

    return main_loop
