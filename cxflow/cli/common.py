import os
import logging
import tempfile
import os.path as path
from datetime import datetime

from typing import Optional, Iterable

from .util import fallback
from ..datasets import AbstractDataset
from ..nets import AbstractNet
from ..hooks import AbstractHook, CXF_HOOK_INIT_ARGS
from ..constants import CXF_LOG_FILE, CXF_HOOKS_MODULE, CXF_CONFIG_FILE, CXF_LOG_DATE_FORMAT, CXF_LOG_FORMAT
from ..utils.reflection import create_object_from_config, get_class_module
from ..utils.config import config_to_str, config_to_file
from ..main_loop import MainLoop


def create_output_dir(config: dict, output_root: str, default_net_name: str='NonameNet') -> str:
    """
    Create output_dir under the given output_root and
        - dump the given config to yaml file under this dir
        - register a file logger logging to a file under this dir
    :param config: config to be dumped
    :param output_root: dir wherein output_dir shall be created
    :param default_net_name: name to be used when `net.name` is not found in the config
    :return: path to the created output_dir
    """
    logging.info('Creating output dir')

    # create output dir
    net_name = default_net_name
    if 'name' not in config['net']:
        logging.warning('\tnet.name not found in config, defaulting to: %s', net_name)
    else:
        net_name = config['net']['name']

    if not os.path.exists(output_root):
        logging.info('\tOutput root folder "%s" does not exist and will be created', output_root)
        os.makedirs(output_root)

    output_dir = tempfile.mkdtemp(prefix='{}_{}_'.format(net_name, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),
                                  dir=output_root)
    logging.info('\tOutput dir: %s', output_dir)

    # create file logger
    file_handler = logging.FileHandler(path.join(output_dir, CXF_LOG_FILE))
    file_handler.setFormatter(logging.Formatter(CXF_LOG_FORMAT, datefmt=CXF_LOG_DATE_FORMAT))
    logging.getLogger().addHandler(file_handler)

    return output_dir


def create_dataset(config: dict, output_dir: Optional[str]=None) -> AbstractDataset:
    """
    Create a dataset object according to the given config.

    Dataset and output_dir configs are passed to the constructor in a single YAML-encoded string.
    :param config: config dict with dataset config
    :param output_dir: path to the training output dir or None
    :return: dataset object
    """
    logging.info('Creating dataset')
    dataset_config = config['dataset']
    if 'output_dir' in dataset_config:
        raise ValueError('The `output_dir` key is reserved and can not be used in dataset configuration.')
    dataset_config['output_dir'] = output_dir
    dataset = create_object_from_config(config['dataset'], args=(config_to_str(dataset_config),))
    logging.info('\t%s created', type(dataset).__name__)
    return dataset


def create_net(config: dict, output_dir: str, dataset: AbstractDataset,
               restore_from: Optional[str]=None) -> AbstractNet:
    """
    Create a net object either from scratch of from the checkpoint in `resume_dir`.

    -------------------------------------------------------
    cxflow allows the following scenarios
    -------------------------------------------------------
    1. Create net: leave `restore_from` to `None` and specify `module` and `class`;
    2. Restore net: specify `resume_dir` a backend-specific path to (a directory with) the saved model.
    -------------------------------------------------------

    :param config: config dict with net config
    :param output_dir: path to the training output dir
    :param dataset: AbstractDataset object
    :param restore_from: from whence the model should be restored (backend-specific information)
    :return: net object
    """

    logging.info('Creating a net')

    net_config = config['net']
    assert 'module' in net_config, '`net.module` not present in the config'
    assert 'class' in net_config, '`net.module` not present in the config'

    # create net kwargs (without `module` and `class`)
    net_kwargs = {'dataset': dataset, 'log_dir': output_dir, 'restore_from': restore_from, **net_config}
    del net_kwargs['module']
    del net_kwargs['class']

    try:
        net = create_object_from_config(net_config, kwargs=net_kwargs, key_prefix='')
    except (ImportError, AttributeError) as ex:
        if restore_from is None:  # training case
            raise ImportError('Cannot create net from the specified net module `{}` and class `{}`.'.format(
                net_config['module'], net_config['class'])) from ex

        else:  # restore cases (resume, predict)
            logging.warning('Cannot create net from the specified net module `%s` and class `%s`.',
                            net_config['module'], net_config['class'])
            assert 'restore_fallback_module' in net_config, '`net.restore_fallback_module` not present in the config'
            assert 'restore_fallback_class' in net_config, '`net.restore_fallback_class` not present in the config'
            logging.info('Trying to restore with fallback module `{}` and class `{}` instead.'.format(
                net_config['restore_fallback_module'], net_config['restore_fallback_class']))

            try:  # try fallback class
                net = create_object_from_config(net_config, kwargs=net_kwargs, key_prefix='restore_fallback_')
            except (ImportError, AttributeError) as ex:  # if fallback module/class specified but it fails
                raise ImportError('Cannot create net from the specified restore_module `{}` and net_class `{}`.'.format(
                    net_config['restore_fallback_module'], net_config['restore_fallback_class'])) from ex

    logging.info('\t%s created', type(net).__name__)
    return net


def create_hooks(config: dict, net: AbstractNet, dataset: AbstractDataset, output_dir: str) -> Iterable[AbstractHook]:
    """
    Create hooks specified in config['hooks'] list.
    :param config: config dict
    :param net: net object to be passed to the hooks
    :param dataset: AbstractDataset object
    :param output_dir: training output dir available to the hooks
    :return: list of hook objects
    """
    logging.info('Creating hooks')
    hooks = []
    if 'hooks' in config:
        for hook_config in config['hooks']:
            assert 'class' in hook_config
            for key in CXF_HOOK_INIT_ARGS:
                if key in hook_config:
                    raise KeyError('Name `{}` is reserved in the hook config. Use a different name.'.format(key))

            # find the hook module if not specified
            if 'module' not in hook_config:
                hook_module = get_class_module(CXF_HOOKS_MODULE, hook_config['class'])
                if hook_module is not None:
                    logging.debug('\tFound hook module `%s` for class `%s`', hook_module, hook_config['class'])
                    hook_config['module'] = hook_module
                else:
                    raise ValueError('Can`t find hook module for hook class `{}`. '
                                     'Make sure it is defined under `{}` sub-modules.'
                                     .format(hook_config['class'], CXF_HOOKS_MODULE))
            # create hook kwargs
            hook_config_to_pass = hook_config.copy()
            hook_config_to_pass.pop('module')
            hook_config_to_pass.pop('class')
            hook_kwargs = {'dataset': dataset, 'net': net, 'output_dir': output_dir, **hook_config_to_pass}
            for key in CXF_HOOK_INIT_ARGS:
                assert key in hook_kwargs

            # create new hook
            try:
                hook = create_object_from_config(hook_config, kwargs=hook_kwargs, key_prefix='')
            except (ValueError, KeyError, TypeError, NameError, AttributeError, AssertionError, ImportError) as ex:
                logging.error('\tFailed to create a hook from config `%s`', hook_config)
                raise ex
            hooks.append(hook)
            logging.info('\t%s created', type(hooks[-1]).__name__)
    return hooks


def run(config: dict, output_root: str, restore_from: str=None, predict: bool=False) -> None:
    """
    Run cxflow training configured by the passed `config`.

    Unique `output_dir` for this training is created under the given `output_root` dir
    wherein all the training outputs are saved. The output dir name will be roughly `[net.name]_[time]`.

    -------------------------------------------------------
    The training procedure consists of the following steps:
    -------------------------------------------------------
    Step 1:
        - Create output dir
        - Create file logger under the output dir
        - Dump loaded config to the output dir
    Step 2:
        - Create dataset
            - YAML string with `dataset` and `log_dir` configs are passed to the dataset constructor
    Step 3:
        - Create network
            - Dataset, `log_dir` and net config is passed to the constructor
            - In case the network is about to resume the training, it does so.
    Step 4:
        - Create all the training hooks
    Step 5:
        - Create the MainLoop object
    Step 6:
        - Run the main loop
    -------------------------------------------------------
    If any of the steps fails, the training is terminated.
    -------------------------------------------------------

    After the training procedure finishes, the output dir will contain the following:
        - train_log.txt with entry_point and main_loop logs (same as the stderr)
        - dumped yaml config

    Additional outputs created by hooks, dataset or tensorflow may include:
        - dataset_log.txt with info about dataset/stream creation
        - model checkpoint(s)
        - tensorboard log file
        - tensorflow event log


    :param config: configuration
    :param output_root: dir under which output_dir shall be created
    :param restore_from: from whence the model should be restored (backend-specific information)
    """

    output_dir = dataset = net = hooks = main_loop = None

    try:
        output_dir = create_output_dir(config=config, output_root=output_root)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Failed to create output dir', ex)

    try:
        dataset = create_dataset(config=config, output_dir=output_dir)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating dataset failed', ex)

    try:
        net = create_net(config=config, output_dir=output_dir, dataset=dataset, restore_from=restore_from)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating network failed', ex)

    try:  # save the config to file
        # modify the config so that it contains fallback information
        config['net']['restore_fallback_module'] = net.restore_fallback_module
        config['net']['restore_fallback_class'] = net.restore_fallback_class
        config_to_file(config=config, output_dir=output_dir, name=CXF_CONFIG_FILE)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Saving config failed', ex)

    try:
        hooks = create_hooks(config=config, net=net, dataset=dataset, output_dir=output_dir)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating hooks failed', ex)

    try:
        logging.info('Creating main loop')
        kwargs = config['main_loop'] if 'main_loop' in config else {}
        main_loop = MainLoop(net=net, dataset=dataset, hooks=hooks, **kwargs)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating main loop failed', ex)

    try:
        logging.info('Running the main loop')
        if predict:
            main_loop.run_prediction()
        else:
            main_loop.run_training()
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Running the main loop failed', ex)
