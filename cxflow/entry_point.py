#!/usr/bin/python3 -mentry_point

from .main_loop import MainLoop
from .nets.abstract_net import AbstractNet
from .datasets.abstract_dataset import AbstractDataset
from .hooks.abstract_hook import AbstractHook
from .utils.config import load_config, config_to_str, config_to_file
from .utils.reflection import create_object_from_config

from argparse import ArgumentParser
from datetime import datetime
from os import path
import logging
import os
import sys
import tempfile
import traceback
import typing


_cxflow_log_formatter = logging.Formatter('%(asctime)s: %(levelname)-8s@%(module)-15s: %(message)s', datefmt='%H:%M:%S')


def _train_load_config(config_file: str, cli_options: typing.Iterable[str]) -> dict:
    """
    Load config from the given yaml file and extend/override it with the given CLI args.
    :param config_file: path to the config yaml file
    :param cli_options: additional args to extend/override the config
    :return: config dict
    """
    logging.info('Loading config')
    config = load_config(config_file=config_file, additional_args=cli_options)
    logging.debug('\tLoaded config: %s', config)

    assert ('net' in config)
    assert ('dataset' in config)
    if 'hooks' not in config:
        logging.warning('\tNo hooks found in config')

    return config


def _train_create_output_dir(config: dict, output_root: str, default_net_name: str='NonameNet') -> str:
    """
    Create output_dir under the given output_root and
        - dump the given config to yaml file under this dir
        - register a file logger logging to a file under this dir
    :param config: config to be dumped
    :param output_root: dir wherein output_dir shall be created
    :return: path to the created output_dir
    """
    logging.info('Creating output dir')

    # create output dir
    net_name = default_net_name
    if 'name' not in config['net']:
        logging.warning('\tnet.name not found in config, defaulting to: %s', net_name)
    else:
        net_name = config['net']['name']

    output_dir = tempfile.mkdtemp(prefix='{}_{}_'.format(net_name, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),
                                  dir=output_root)
    logging.info('\tOutput dir: %s', output_dir)

    # create file logger
    file_handler = logging.FileHandler(path.join(output_dir, 'train.log'))
    file_handler.setFormatter(_cxflow_log_formatter)
    logging.getLogger().addHandler(file_handler)

    # dump config including CLI args
    config_to_file(config=config, output_dir=output_dir)

    return output_dir


def _train_create_dataset(config: dict, output_dir: str) -> AbstractDataset:
    """
    Create a dataset object according to the given config.

    Dataset, stream and output_dir configs are passed to the constructor in a single YAML-encoded string.
    :param config: config dict with dataset and stream configs
    :param output_dir: path to the training output dir
    :return: dataset object
    """
    logging.info('Creating dataset')
    config_str = config_to_str({'dataset': config['dataset'],
                                'stream': config['stream'],
                                'output_dir': output_dir})

    dataset = create_object_from_config(config['dataset'], args=(config_str,))
    logging.info('\t%s created', type(dataset).__name__)
    return dataset


def _train_create_net(config: dict, output_dir: str, dataset: AbstractDataset) -> AbstractNet:
    """
    Create a net object either from scratch of from the specified checkpoint.

    -------------------------------------------------------
    To restore a net from a checkpoint, one must provide config['net']['restore_from'] parameter
    with a path to the checkpoint.
    -------------------------------------------------------

    :param config: config dict with net config
    :param output_dir: path to the training output dir
    :param dataset: AbstractDataset object
    :return: net object
    """
    net_config = config['net']
    if 'restore_from' in net_config:
        logging.info('Restoring net from: "%s"', net_config['restore_from'])
        if 'net_module' in net_config or 'net_class' in net_config:
            logging.warning('`net_module` and `net_class` config parameters are provided yet ignored')
        net = AbstractNet(dataset=dataset, log_dir=output_dir, **net_config)
    else:
        logging.info('Creating new net')

        net = create_object_from_config(net_config, kwargs={'dataset': dataset, 'log_dir': output_dir, **net_config})
        logging.info('\t%s created', type(net).__name__)
    return net


def _train_create_hooks(config: dict, net: AbstractNet, dataset: AbstractDataset) -> typing.Iterable[AbstractHook]:
    """
    Create hooks specified in config['hooks'] list.
    :param config: config dict
    :param net: net object to be passed to the hooks
    :param dataset: AbstractDataset object
    :return: list of hook objects
    """
    logging.info('Creating hooks')
    hooks = []
    if 'hooks' in config:
        for hook_config in config['hooks']:
            hooks.append(create_object_from_config(hook_config, kwargs={'dataset': dataset, 'net': net,
                                                                        'config': config, **hook_config}))
            logging.info('\t%s created', type(hooks[-1]).__name__)
    return hooks


def _fallback(message: str, e: Exception) -> None:
    """
    Fallback procedure when a training step fails.
    :param message: message to be logged
    :param e: Exception which caused the failure
    """
    logging.error('%s: %s\n%s', message, e, traceback.format_exc())
    sys.exit(1)


def train(config_file: str, cli_options: typing.Iterable[str], output_root: str) -> None:
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
            - yaml string with `dataset`, `stream` and `log_dir` configs is passed to the dataset constructor
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
        config = _train_load_config(config_file=config_file, cli_options=cli_options)
    except Exception as e:
        _fallback('Loading config failed', e)

    try:
        output_dir = _train_create_output_dir(config=config, output_root=output_root)
    except Exception as e:
        _fallback('Failed to create output dir', e)

    try:
        dataset = _train_create_dataset(config=config, output_dir=output_dir)
    except Exception as e:
        _fallback('Creating dataset failed', e)

    try:
        net = _train_create_net(config=config, output_dir=output_dir, dataset=dataset)
    except Exception as e:
        _fallback('Creating network failed', e)

    try:
        hooks = _train_create_hooks(config=config, net=net, dataset=dataset)
    except Exception as e:
        _fallback('Creating hooks failed', e)

    try:
        logging.info('Creating main loop')
        main_loop = MainLoop(net=net, dataset=dataset, hooks=hooks)
    except Exception as e:
        _fallback('Creating main loop failed', e)

    try:
        logging.info('Running the main loop')
        main_loop.run(run_test_stream=('test' in config['stream']))
    except Exception as e:
        _fallback('Running the main loop failed', e)


def split(config_file: str, num_splits: int, train_ratio: float, valid_ratio: float, test_ratio: float=0):
    logging.info('Splitting to %d splits with ratios %f:%f:%f', num_splits, train_ratio, valid_ratio, test_ratio)

    try:
        logging.info('Loading config')
        config = load_config(config_file=config_file, additional_args=[])
    except Exception as e:
        _fallback('Loading config failed', e)

    try:
        logging.info('Creating dataset')
        config_str = config_to_str({'dataset': config['dataset'], 'stream': config['stream']})
        dataset = create_object_from_config(config['dataset'], args=(config_str,))
    except Exception as e:
        _fallback('Creating dataset failed', e)

    logging.info('Splitting')
    dataset.split(num_splits, train_ratio, valid_ratio, test_ratio)


def entry_point() -> None:
    """
    cxflow entry point for training and dataset splitting.
    """

    # make sure the path contains the current working directory
    sys.path.insert(0, os.getcwd())

    # create parser
    parser = ArgumentParser('cxflow')
    subparsers = parser.add_subparsers(help='cxflow modes')

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
    for p in [parser, train_parser, split_parser]:
        p.add_argument('-v', '--verbose', action='store_true', help='increase verbosity do level DEBUG')
        p.add_argument('-o', '--output-root', default='log', help='output directory')

    # parse CLI arguments
    known_args, unknown_args = parser.parse_known_args()

    # show help if no subcommand was specified.
    if not hasattr(known_args, 'subcommand'):
        parser.print_help()
        quit(1)

    # set up global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG if known_args.verbose else logging.INFO)
    logger.handlers = []  # remove default handlers

    # set up STDERR handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(_cxflow_log_formatter)
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


if __name__ == '__main__':
    entry_point()
