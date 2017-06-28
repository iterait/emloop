"""
This module is cxflow framework entry point.

The entry point shall be accessed from command line via `cxflow` command.

At the moment cxflow allows to
- train a network with `cxflow train ...`
- split data to x-validation sets with `cxflow split ...`

Run `cxflow -h` for details.
"""

import logging
import os
from os import path
import sys
import tempfile
import traceback
from argparse import ArgumentParser
from datetime import datetime
from typing import Iterable, Optional

from .datasets import AbstractDataset
from .hooks.abstract_hook import AbstractHook, CXF_HOOK_INIT_ARGS
from .main_loop import MainLoop
from .nets.abstract_net import AbstractNet
from .utils.config import load_config, config_to_str, config_to_file
from .utils.grid_search import grid_search
from .utils.reflection import create_object_from_config, get_class_module

# cxflow logging formats and formatter.
CXF_LOG_FORMAT = '%(asctime)s: %(levelname)-8s@%(module)-15s: %(message)s'
CXF_LOG_DATE_FORMAT = '%H:%M:%S'
CXF_LOG_FORMATTER = logging.Formatter(CXF_LOG_FORMAT, datefmt=CXF_LOG_DATE_FORMAT)

# Module with standard cxflow hooks (as would be used in import).
CXF_HOOKS_MODULE = 'cxflow.hooks'


def train_load_config(config_file: str, cli_options: Iterable[str]) -> dict:
    """
    Load config from the given yaml file and extend/override it with the given CLI args.
    :param config_file: path to the config yaml file
    :param cli_options: additional args to extend/override the config
    :return: config dict
    """
    logging.info('Loading config')
    config = load_config(config_file=config_file, additional_args=cli_options)
    logging.debug('\tLoaded config: %s', config)

    assert 'net' in config
    assert 'dataset' in config
    if 'hooks' not in config:
        logging.warning('\tNo hooks found in config')

    return config


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
    file_handler = logging.FileHandler(path.join(output_dir, 'train.log'))
    file_handler.setFormatter(CXF_LOG_FORMATTER)
    logging.getLogger().addHandler(file_handler)

    # dump config including CLI args
    config_to_file(config=config, output_dir=output_dir)

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
    dataset_config = {'dataset': config['dataset']}
    if output_dir:
        dataset_config['output_dir'] = output_dir
    dataset = create_object_from_config(config['dataset'], args=(config_to_str(dataset_config),))
    logging.info('\t%s created', type(dataset).__name__)
    return dataset


def create_net(config: dict, output_dir: str, dataset: AbstractDataset) -> AbstractNet:
    """
    Create a net object either from scratch of from the specified checkpoint.

    -------------------------------------------------------
    cxflow allows the following scenarios
    -------------------------------------------------------
    1. Create net: specify net_module and net_class, do not specify restore_from
    2. Restore net: specify restore_from, net_module and net_class is ignored
    3. Custom restore net (e.g. fine-tunning): specify restore_from, restore_module and restore_class
    -------------------------------------------------------

    :param config: config dict with net config
    :param output_dir: path to the training output dir
    :param dataset: AbstractDataset object
    :return: net object
    """
    net_config = config['net']
    net_kwargs = {'dataset': dataset, 'log_dir': output_dir, **net_config}
    if 'restore_from' in net_config:
        logging.info('Restoring net')
        try:
            net = create_object_from_config(net_config, kwargs=net_kwargs, key_prefix='restore_')
            logging.info('\tNet restored with custom class')
        except (AssertionError, ValueError, AttributeError, ImportError, TypeError) as _:
            logging.error('Cannot restore without net module and class specification.')
            # See issue #50 and #51
            # net = BaseTFNetRestore(**net_kwargs)
            # logging.info('\tNet restored with generic BaseTFNetRestore')
    else:
        logging.info('Creating net')
        net = create_object_from_config(net_config, kwargs=net_kwargs)
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


def fallback(message: str, ex: Exception) -> None:
    """
    Fallback procedure when a training step fails.
    :param message: message to be logged
    :param ex: Exception which caused the failure
    """
    logging.error('%s: %s\n%s', message, ex, traceback.format_exc())
    sys.exit(1)


def train(config_file: str, cli_options: Iterable[str], output_root: str) -> None:
    """
    Run cxflow training configured from the given file and cli_options.

    Unique output dir for this training is created under the given output_root dir
    wherein all the training outputs are saved. The output dir name will be roughly [net.name]_[time].

    -------------------------------------------------------
    The training procedure consists of the following steps:
    -------------------------------------------------------
    Step 1:
        - Load yaml configuration and override or extend it with parameters passed in CLI arguments
        - Check if `net` and `dataset` configs are present
    Step 2:
        - Create output dir
        - Create file logger under the output dir
        - Dump loaded config to the output dir
    Step 3:
        - Create dataset
            - yaml string with `dataset` and `log_dir` configs is passed to the dataset constructor
    Step 4:
        - Create network
            - Dataset, `log_dir` and net config is passed to the constructor
    Step 5:
        - Create all the training hooks
    Step 6:
        - Create the MainLoop object
    Step 7:
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


    :param config_file: path to the training yaml config
    :param cli_options: additional CLI arguments to override or extend the yaml config
    :param output_root: dir under which output_dir shall be created
    """

    config = output_dir = dataset = net = hooks = main_loop = None

    try:
        config = train_load_config(config_file=config_file, cli_options=cli_options)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Loading config failed', ex)

    try:
        output_dir = create_output_dir(config=config, output_root=output_root)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Failed to create output dir', ex)

    try:
        dataset = create_dataset(config=config, output_dir=output_dir)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating dataset failed', ex)

    try:
        net = create_net(config=config, output_dir=output_dir, dataset=dataset)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating network failed', ex)

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
        main_loop.run()
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Running the main loop failed', ex)


def split(config_file: str, num_splits: int, train_ratio: float, valid_ratio: float, test_ratio: float=0) -> None:
    """
    Create dataset and call the split method with the given args.
    :param config_file: path to the training yaml config
    :param num_splits: number of x-val splits to be created
    :param train_ratio: portion of data to be split to the train set
    :param valid_ratio: portion of data to be split to the valid set
    :param test_ratio: portion of data to be split to the test set
    """
    logging.info('Splitting to %d splits with ratios %f:%f:%f', num_splits, train_ratio, valid_ratio, test_ratio)

    config = dataset = None

    try:
        logging.info('Loading config')
        config = load_config(config_file=config_file, additional_args=[])
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Loading config failed', ex)

    try:
        logging.info('Creating dataset')
        dataset = create_dataset(config)
    except Exception as ex:  # pylint: disable=broad-except
        fallback('Creating dataset failed', ex)

    logging.info('Splitting')
    dataset.split(num_splits, train_ratio, valid_ratio, test_ratio)


def entry_point() -> None:
    """
    cxflow entry point for training and dataset splitting.
    """

    # make sure the path contains the current working directory
    sys.path.insert(0, os.getcwd())

    # create parser
    main_parser = ArgumentParser('cxflow')
    subparsers = main_parser.add_subparsers(help='cxflow modes')

    # create train subparser
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(subcommand='train')
    train_parser.add_argument('config_file', help='path to the config file')

    # create split subparser
    split_parser = subparsers.add_parser('split')
    split_parser.set_defaults(subcommand='split')
    split_parser.add_argument('config_file', help='path to the config file')
    split_parser.add_argument('-n', '--num-splits', type=int, default=1, help='number of splits')
    split_parser.add_argument('-r', '--ratio', type=int, nargs=3, required=True, help='train, valid and test ratios')

    # add common arguments
    for parser in [main_parser, train_parser, split_parser]:
        parser.add_argument('-v', '--verbose', action='store_true', help='increase verbosity do level DEBUG')
        parser.add_argument('-o', '--output-root', default='log', help='output directory')

    # create grid-search subparser
    gridsearch_parser = subparsers.add_parser('gridsearch')
    gridsearch_parser.set_defaults(subcommand='gridsearch')
    gridsearch_parser.add_argument('script', help='Script to be grid-searched')
    gridsearch_parser.add_argument('params', nargs='*', help='Params to be tested. Format: name:type=[value1,value2].'
                                                             'Type is optional')
    gridsearch_parser.add_argument('--dry-run', action='store_true', help='Print commands instead of executing them'
                                                                          'right away')

    # parse CLI arguments
    known_args, unknown_args = main_parser.parse_known_args()

    # show help if no subcommand was specified.
    if not hasattr(known_args, 'subcommand'):
        main_parser.print_help()
        quit(1)

    # set up global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG if known_args.verbose else logging.INFO)
    logger.handlers = []  # remove default handlers

    # set up STDERR handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(CXF_LOG_FORMATTER)
    logger.addHandler(stderr_handler)

    if known_args.subcommand == 'train':
        train(config_file=known_args.config_file,
              cli_options=unknown_args,
              output_root=known_args.output_root)

    elif known_args.subcommand == 'split':
        split(config_file=known_args.config_file,
              num_splits=known_args.num_splits,
              train_ratio=known_args.ratio[0],
              valid_ratio=known_args.ratio[1],
              test_ratio=known_args.ratio[2])

    elif known_args.subcommand == 'gridsearch':
        grid_search(script=known_args.script,
                    params=known_args.params,
                    dry_run=known_args.dry_run)


if __name__ == '__main__':
    entry_point()
