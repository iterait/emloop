#!/usr/bin/python3 -mentry_point

from .main_loop import MainLoop
from .utils.arg_parser import parse_arg
from .datasets.abstract_dataset import AbstractDataset
from .hooks.abstract_hook import AbstractHook
from .nets.abstract_net import AbstractNet

import numpy.random as npr

from argparse import ArgumentParser
from datetime import datetime
import importlib
import logging
import os
from os import path
import sys
import typing
import yaml
import ruamel.yaml
import traceback

# set up custom logging format
_cxflow_log_formatter = logging.Formatter('%(asctime)s: %(levelname)-8s@%(module)s: %(message)s', datefmt='%H:%M:%S')


class EntryPoint:
    """Entry point of the whole training. Should be used only via `cxflow` command."""

    @staticmethod
    def load_config(config_file: str, additional_args: typing.Iterable[str]) -> dict:
        """
        Load config from `config_file` and apply CLI args `additional_args`.
        :param additional_args: Additional args which may extend or override the config from file.
        :return: configuration as dict
        """

        with open(config_file, 'r') as f:
            config = ruamel.yaml.load(f, ruamel.yaml.RoundTripLoader)

        for key_full, value in [parse_arg(arg) for arg in additional_args]:
            key_split = key_full.split('.')
            key_prefix = key_split[:-1]
            key = key_split[-1]

            conf = self._config
            for key_part in key_prefix:
                conf = conf[key_part]
            conf[key] = value

        config = yaml.load(ruamel.yaml.dump(config, Dumper=ruamel.yaml.RoundTripDumper))
        logging.debug('Loaded config: %s', config)
        return config

    @staticmethod
    def create_output_dir(output_root: str, config: dict) -> str:
        """Create output directory with proper name (if specified in the net config section)."""

        name = 'UnknownNetName'
        try:
            name = config['net']['name']
        except:
            logging.warning('Net name not found in the config')

        output_dir = path.join(output_root,'{}_{}_{}'.format(name,
                                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                                                             npr.random_integers(10000, 99999)))
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

        return dataset_class(config_str=EntryPoint.config_to_str({'dataset':dataset_config, 'stream':stream_config}))

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
    def config_to_file(config, output_dir: str, name: str= 'config.yaml') -> str:
        """
        Save the given config to the given path in yaml.
        :param config: configuration dict
        :param output_dir: target output directory
        :param name: target filename
        :return: target path
        """
        dumped_config_f = path.join(output_dir, name)
        with open(dumped_config_f, 'w') as f:
            yaml.dump(config, f)
        return dumped_config_f

    @staticmethod
    def config_to_str(config: dict) -> str:
        """
        Return the given given config as yaml str.
        :param config: configuration dict
        :return: given configuration as yaml str
        """
        return yaml.dump(config)

    @staticmethod
    def construct_hook(config, net, hook_module: str, hook_class: str, **kwargs) -> AbstractHook:
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
    def construct_hooks(config: dict, net: AbstractNet) -> typing.Iterable[AbstractHook]:
        """Construct hooks from the saved config. file."""
        hooks = []
        if 'hooks' in config:
            logging.info('Creating hooks')
            for hook_conf in config['hooks']:
                try:
                    hook = EntryPoint.construct_hook(config, net, **hook_conf)
                    hooks.append(hook)
                except Exception as e:
                    logging.error('Unexpected exception occurred when constructing hook with config: "%s".'
                                  'Exception: %"', hook_conf, e)
                    raise e
        else:
            logging.warning('No hooks found')

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
            config = EntryPoint.load_config(config_file=config_file, additional_args=cli_options)
        except Exception as e:
            logging.error('Loading config failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        output_dir = EntryPoint.create_output_dir(output_root=output_root, config=config)

        # setup file handler
        file_handler = logging.FileHandler(path.join(output_dir, 'train_log.txt'))
        file_handler.setFormatter(_cxflow_log_formatter)
        logging.getLogger().addHandler(file_handler)

        try:
            dataset = EntryPoint.create_dataset(config['dataset'], config['stream'])
        except Exception as e:
            logging.error('Creating dataset failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            net = EntryPoint.create_network(net_config=config['net'], dataset=dataset, output_dir=output_dir)
        except Exception as e:
            logging.error('Creating network failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            hooks = EntryPoint.construct_hooks(config=config, net=net)
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
            config = EntryPoint.load_config(config_file=config_file, additional_args=[])
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
    # make sure the path contains the current working directory
    sys.path.insert(0, os.getcwd())

    # create parser
    parser = ArgumentParser('cxflow')
    subparsers = parser.add_subparsers(help='cxflow modes')

    # create train subparser
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(subcommand='train')
    train_parser.add_argument('config_file', help='path to the config file')

    # create crossval subparser
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

    # run entry-point method according to the proper subcommand
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
