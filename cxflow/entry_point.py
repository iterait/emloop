#!/usr/bin/python3 -mentry_point

from .main_loop import MainLoop
from .utils.arg_parser import parse_arg
from .dataset_loader import DatasetLoader
from .hooks.abstract_hook import AbstractHook
from .nets.abstract_net import AbstractNet

import numpy.random as npr

from argparse import ArgumentParser
from datetime import datetime
import importlib
import logging
from logging.handlers import RotatingFileHandler
import os
from os import path
import sys
import typing
import yaml
import ruamel.yaml
import traceback


class EntryPoint:
    """Entry point of the whole training. Should be used only via `cxflow` command."""

    def _load_config(self, config_file: str, additional_args: typing.Iterable[str]) -> dict:
        """Load config from `config_file` and apply CLI args `additional_args`. The result is saved as `self.config`"""

        with open(config_file, 'r') as f:
            self._config = ruamel.yaml.load(f, ruamel.yaml.RoundTripLoader)

        for key_full, value in [parse_arg(arg) for arg in additional_args]:
            key_split = key_full.split('.')
            key_prefix = key_split[:-1]
            key = key_split[-1]

            conf = self._config
            for key_part in key_prefix:
                conf = conf[key_part]
            conf[key] = value

        self._config = yaml.load(ruamel.yaml.dump(self._config, Dumper=ruamel.yaml.RoundTripDumper))
        logging.debug('Loaded config: %s', self._config)
        return self._config

    def create_output_dir(self, output_root: str) -> str:
        """Create output directory with proper name (if specified in the net config section)."""

        name = 'UnknownNetName'
        try:
            name = self._config['net']['name']
        except:
            logging.warning('Net name not found in the config')

        output_dir = path.join(output_root,'{}_{}_{}'.format(name,
                                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                                                             npr.random_integers(10000, 99999)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        return output_dir

    def _create_dataset(self) -> None:
        """Use `DatasetLoader` in order to load the proper dataset."""
        data_loader = DatasetLoader(self._config, self.dumped_config_file)
        self._dataset = data_loader.load_dataset()

    def _create_network(self) -> None:
        """
        Use python reflection to construct the net.

        Note that config must contain `net_module` and `net_class`.
        """

        logging.info('Creating net')

        if 'restore_from' in self._config['net']:
            logging.info('Restoring net from: "%s"', self._config['net']['restore_from'])
            if 'net_module' in self._config['net'] or 'net_class' in self._config['net']:
                logging.warning('`net_module` or `net_class` provided even though the net is restoring from "%s".'
                                'Restoring anyway while ignoring these parameters. Consider removing them from config'
                                'file.', self._config['net']['restore_from'])

            self._net = AbstractNet(dataset=self._dataset, log_dir=self.output_dir, **self._config['net'])
        else:
            logging.info('Creating new net')
            logging.debug('Loading net module')
            net_module = importlib.import_module(self._config['net']['net_module'])

            logging.debug('Loading net class')
            net_class = getattr(net_module, self._config['net']['net_class'])

            logging.debug('Constructing net instance')
            self._net = net_class(dataset=self._dataset, log_dir=self.output_dir, **self._config['net'])

    def _dump_config(self, name: str='config.yaml') -> str:
        """Save the YAML file."""

        dumped_config_f = path.join(self.output_dir, name)
        with open(dumped_config_f, 'w') as f:
            yaml.dump(self._config, f)
        return dumped_config_f

    def _construct_hook(self, hook_module: str, hook_class: str, **kwargs) -> AbstractHook:
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
        hook = hook_class(net=self._net, config=self._config, **kwargs)
        return hook

    def _construct_hooks(self) -> typing.Iterable[AbstractHook]:
        """Construct hooks from the saved config. file."""

        hooks = []
        if 'hooks' in self._config:
            logging.info('Creating hooks')
            for hook_conf in self._config['hooks']:
                try:
                    hook = self._construct_hook(**hook_conf)
                    hooks.append(hook)
                except Exception as e:
                    logging.error('Unexpected exception occurred when constructing hook with config: "%s".'
                                  'Exception: %"', hook_conf, e)
                    raise e
            return hooks
        else:
            logging.warning('No hooks found')
            return []

    def train(self, config_file: str, cli_options: typing.Iterable[str]) -> None:
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

        self._net = None

        try:
            self._load_config(config_file=config_file, additional_args=cli_options)
        except Exception as e:
            logging.error('Loading config failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            self.dumped_config_file = self._dump_config()
        except Exception as e:
            logging.error('Saving modified config failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            self._create_dataset()
        except Exception as e:
            logging.error('Creating dataset failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            self._create_network()
        except Exception as e:
            logging.error('Creating network failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            hooks = self._construct_hooks()
        except Exception as e:
            logging.error('Creating hooks failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            logging.info('Creating main loop')
            main_loop = MainLoop(net=self._net,
                                 dataset=self._dataset,
                                 hooks=hooks)
        except Exception as e:
            logging.error('Creating main loop failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            logging.info('Running the main loop')
            main_loop.run(run_test_stream=('test' in self._config['stream']))
        except Exception as e:
            logging.error('Running the main loop failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

    def split(self, config_file: str, num_splits: int, train_ratio: float, valid_ratio: float, test_ratio: float=0):
        logging.info('Splitting to %d splits with ratios %f:%f:%f', num_splits, train_ratio, valid_ratio, test_ratio)

        logging.debug('Loading config')
        try:
            self._load_config(config_file=config_file, additional_args=[])
        except Exception as e:
            logging.error('Loading config failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        try:
            self.dumped_config_file = self._dump_config()
        except Exception as e:
            logging.error('Saving modified config failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        logging.debug('Creating dataset')
        try:
            self._create_dataset()
        except Exception as e:
            logging.error('Creating dataset failed: %s\n%s', e, traceback.format_exc())
            sys.exit(1)

        logging.debug('Splitting')
        self._dataset.split(num_splits, train_ratio, valid_ratio, test_ratio)


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

    # set up global logger
    logger = logging.getLogger('')
    logger.handlers = []  # remove default handlers
    if known_args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # set up custom logging format
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s@%(module)s: %(message)s', datefmt='%H:%M:%S')

    # set up STDERR handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # create entry point - this cannot be done earlier as the logger wasn't set up and this cannot be done later as
    # the logdir is required for file logger
    entry_point = EntryPoint()
    output_dir = entry_point.create_output_dir(known_args.output_root)

    # setup file handler
    file_handler = RotatingFileHandler(path.join(output_dir, 'stderr.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # run entry-point method according to the proper subcommand
    if not hasattr(known_args, 'subcommand'):
        parser.print_help()
        quit(1)
    if known_args.subcommand == 'train':
        entry_point.train(config_file=known_args.config_file,
                          cli_options=unknown_args)

    elif known_args.subcommand == 'split':
        entry_point.split(config_file=known_args.config_file, num_splits=known_args.num_splits,
                          train_ratio=known_args.ratio[0], valid_ratio=known_args.ratio[1],
                          test_ratio=known_args.ratio[2])

    else:
        logging.error('Unrecognized subcommand: "%s". Please run `cxflow -h` for more info.', known_args.subcommand)
        sys.exit(1)


if __name__ == '__main__':
    init_entry_point()
