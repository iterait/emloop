#!/usr/bin/python3 -mentry_point

from .main_loop import MainLoop
from .datasets.abstract_dataset import AbstractDataset
from .hooks.abstract_hook import AbstractHook
from .nets.abstract_net import AbstractNet
from .utils.config import load_config, config_to_str, config_to_file

import numpy.random as npr

from argparse import ArgumentParser
from datetime import datetime
import importlib
import logging
import os
import sys
import typing
import traceback
from os import path

# set up custom logging format
_cxflow_log_formatter = logging.Formatter('%(asctime)s: %(levelname)-8s@%(module)s: %(message)s', datefmt='%H:%M:%S')


class EntryPoint:
    """Entry point of the whole training. Should be used only via `cxflow` command."""

    @staticmethod
    def create_output_dir(output_root: str, net_name: str) -> str:
        """Create output directory with proper name (if specified in the net config section)."""
        output_dir = path.join(output_root,'{}_{}'.format(net_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir

    @staticmethod
    def create_dataset(dataset_config: dict, stream_config: dict) -> AbstractDataset:
        """Use `DatasetLoader` in order to load the proper dataset."""
        logging.debug('Loading dataset module')
        dataset_module = importlib.import_module(dataset_config['dataset_module'])

        logging.debug('Loading dataset class')
        dataset_class = getattr(dataset_module, dataset_config['dataset_class'])

        return dataset_class(config_str=config_to_str({'dataset':dataset_config, 'stream':stream_config}))

    @staticmethod
    def create_network(net_config: dict, dataset: AbstractDataset, output_dir: str) -> AbstractNet:
        """
        Create network according to the given config.
        Either create a new one, or load it from the checkpoint specified in net_config['restore_from'].
        :param net_config: net configuration dict
        :param dataset: AbstractDataset
        :param output_dir: output directory to use for dumping checkpoints
        :return: net
        """

        if 'restore_from' in net_config:
            logging.info('Restoring net from: "%s"', net_config['restore_from'])
            if 'net_module' in net_config or 'net_class' in net_config:
                logging.warning('`net_module` or `net_class` provided even though the net is restoring from "%s".'
                                'Restoring anyway while ignoring these parameters. Consider removing them from config'
                                'file.', net_config['restore_from'])

            net = AbstractNet(dataset=dataset, log_dir=output_dir, **net_config)
        else:
            logging.info('Creating new net')
            logging.debug('Loading net module')
            net_module = importlib.import_module(net_config['net_module'])

            logging.debug('Loading net class')
            net_class = getattr(net_module, net_config['net_class'])

            logging.debug('Constructing net instance')
            net = net_class(dataset=dataset, log_dir=output_dir, **net_config)
        return net

    @staticmethod
    def create_hook(config, net, hook_module: str, hook_class: str, **kwargs) -> AbstractHook:
        """
        Construct a hook.

        This is equivalent to
        ```
        from <hook_module> import <hook_class>
        return <hook_class>(**kwargs)
        ```
        """

        logging.debug('Loading hook module %s', hook_module)
        hook_module = importlib.import_module(hook_module)

        logging.debug('Loading hook class %s', hook_class)
        hook_class = getattr(hook_module, hook_class)

        logging.debug('Constructing hook')
        hook = hook_class(net=net, config=config, **kwargs)
        return hook

    @staticmethod
    def create_hooks(config: dict, net: AbstractNet) -> typing.Iterable[AbstractHook]:
        """Construct hooks from the saved config. file."""
        hooks = []
        if 'hooks' in config:
            for hook_conf in config['hooks']:
                hook = EntryPoint.create_hook(config, net, **hook_conf)
                hooks.append(hook)

        return hooks

    @staticmethod
    def train(config_file: str, cli_options: typing.Iterable[str], output_root: str) -> None:
        """
        Train method resposible for constring all required objects and training itself.

        All arguments are passed via CLI arguments which should be in form of `key[:type]=value`.
        Then the life cycle is as follows:
        1. configuration file is loaded
        2. CLI arguments are applied
        3. final configuration is dumped
        4. dataset is loaded
        5. network is created
        6. main loop hooks are created
        7. main loop is created
        8. main loop is run
        """

        try:
            logging.info('Loading config')
            config = load_config(config_file=config_file, additional_args=cli_options)
            logging.debug('Loaded config: %s', config)
        except Exception as e:
            logging.error('Loading config failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            logging.info('Creating output dir')

            # create output dir
            net_name = 'UnknownNetName'
            if 'name' not in config['net']:
                logging.warning('net.name not found in config, defaulting to: %s', net_name)
            else:
                net_name = config['net']['name']
            output_dir = EntryPoint.create_output_dir(output_root=output_root, net_name=net_name)

            # create file logger
            file_handler = logging.FileHandler(path.join(output_dir, 'train_log.txt'))
            file_handler.setFormatter(_cxflow_log_formatter)
            logging.getLogger().addHandler(file_handler)

            # dump config including CLI args
            config_to_file(config=config, output_dir=output_dir)

        except Exception as e:
            logging.error('Failed to create output dir: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            logging.info('Creating dataset')
            dataset = EntryPoint.create_dataset(config['dataset'], config['stream'])
        except Exception as e:
            logging.error('Creating dataset failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            logging.info('Creating network')
            net = EntryPoint.create_network(net_config=config['net'], dataset=dataset, output_dir=output_dir)
        except Exception as e:
            logging.error('Creating network failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            logging.info('Creating hooks')
            if 'hooks' not in config:
                logging.warning('No hooks found')
            hooks = EntryPoint.create_hooks(config=config, net=net)
        except Exception as e:
            logging.error('Creating hooks failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            logging.info('Creating main loop')
            main_loop = MainLoop(net=net, dataset=dataset, hooks=hooks)
        except Exception as e:
            logging.error('Creating main loop failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            logging.info('Running the main loop')
            main_loop.run(run_test_stream=('test' in config['stream']))
        except Exception as e:
            logging.error('Running the main loop failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

    @staticmethod
    def split(config_file: str, num_splits: int, train_ratio: float, valid_ratio: float, test_ratio: float=0):
        logging.info('Splitting to %d splits with ratios %f:%f:%f', num_splits, train_ratio, valid_ratio, test_ratio)

        logging.debug('Loading config')
        try:
            config = load_config(config_file=config_file, additional_args=[])
        except Exception as e:
            logging.error('Loading config failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        logging.debug('Creating dataset')
        try:
            dataset = EntryPoint.create_dataset(config['dataset'], config['stream'])
        except Exception as e:
            logging.error('Creating dataset failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        logging.debug('Splitting')
        dataset.split(num_splits, train_ratio, valid_ratio, test_ratio)


def init_entry_point() -> None:
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
    split_parser.add_argument('-n', '--num-splits', type=int, help='number of splits')
    split_parser.add_argument('-r', '--ratio', type=int, nargs=3, help='train, valid and test ratios')

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
        EntryPoint.train(config_file=known_args.config_file,
                         cli_options=unknown_args,
                         output_root=known_args.output_root)

    elif known_args.subcommand == 'split':
        EntryPoint.split(config_file=known_args.config_file,
                         num_splits=known_args.num_splits,
                         train_ratio=known_args.ratio[0],
                         valid_ratio=known_args.ratio[1],
                         test_ratio=known_args.ratio[2])


if __name__ == '__main__':
    init_entry_point()
