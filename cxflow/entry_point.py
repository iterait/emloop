#!/usr/bin/python3 -mentry_point

from .network_manager import NetworkManager
from .utils.arg_parser import parse_arg
from .dataset_loader import DatasetLoader
from .hooks.abstract_hook import AbstractHook
from .nets.abstract_net import AbstractNet

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


class EntryPoint:
    """Entry point of the whole training. Should be used only via `cxflow` command."""

    def _load_config(self, config_file: str, additional_args: typing.Iterable[str]) -> dict:
        """Load config from `config_file` and apply CLI args `additional_args`. The result is saved as `self.config`"""

        with open(config_file, 'r') as f:
            self.config = ruamel.yaml.load(f, ruamel.yaml.RoundTripLoader)

        for key_full, value in [parse_arg(arg) for arg in additional_args]:
            key_split = key_full.split('.')
            key_prefix = key_split[:-1]
            key = key_split[-1]

            conf = self.config
            for key_part in key_prefix:
                conf = conf[key_part]
            conf[key] = value

        self.config = yaml.load(ruamel.yaml.dump(self.config, Dumper=ruamel.yaml.RoundTripDumper))
        logging.debug('Loaded config: %s', self.config)
        return self.config

    def _create_output_dir(self) -> str:
        """Create output directory with proper name (if specified in the net config section)."""

        name = 'UnknownNetName'
        try:
            name = self.config['net']['name']
        except:
            logging.warning('Net name not found in the config')

        output_dir = path.join(self._output_root,'{}_{}'.format(name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def _create_dataset(self) -> None:
        """Use `DatasetLoader` in order to load the proper dataset."""
        data_loader = DatasetLoader(self.config, self.dumped_config_file)
        self.dataset = data_loader.load_dataset()

    def _create_network(self) -> None:
        """
        Use python reflection to construct the net.

        Note that config must contain `net_module` and `net_class`.
        """

        logging.info('Creating net')

        if 'restore_from' in self.config['net']:
            logging.info('Restoring net from: "%s"', self.config['net']['restore_from'])
            if 'net_module' in self.config['net'] or 'net_class' in self.config['net']:
                logging.warning('`net_module` or `net_class` provided even though the net is restoring from "%s".'
                                'Restoring anyway while ignoring these parameters. Consider removing them from config'
                                'file.', self.config['net']['restore_from'])

            self.net = AbstractNet(dataset=self.dataset, log_dir=self.output_dir, **self.config['net'])
        else:
            logging.info('Creating new net')
            logging.debug('Loading net module')
            net_module = importlib.import_module(self.config['net']['net_module'])

            logging.debug('Loading net class')
            net_class = getattr(net_module, self.config['net']['net_class'])

            logging.debug('Constructing net instance')
            self.net = net_class(dataset=self.dataset, log_dir=self.output_dir, **self.config['net'])

    def _dump_config(self, name='config.yaml') -> str:
        """Save the YAML file."""

        dumped_config_f = path.join(self.output_dir, name)
        with open(dumped_config_f, 'w') as f:
            yaml.dump(self.config, f)
        return dumped_config_f

    def _construct_hook(self, hook_module: str, hook_class, **kwargs) -> AbstractHook:
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
        hook = hook_class(net=self.net, config=self.config, **kwargs)
        return hook

    def _construct_hooks(self) -> typing.Iterable[AbstractHook]:
        """Construct hooks from the saved config. file."""

        hooks = []
        if 'hooks' in self.config:
            logging.info('Creating hooks')
            for hook_conf in self.config['hooks']:
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

    def train(self, config_file: str, output_root: str, cli_options: typing.Iterable[str]) -> None:
        """
        Train method resposible for constring all required objects and training itself.

        All arguments are passed via CLI arguments which should be in form of `key[:type]=value`.
        Then the life cycle is as follows:
        1. configuration file is loaded
        2. CLI arguments are applied
        3. training directory is created
        4. final configuration is dumped
        5. dataset is loaded
        6. network is created
        7. mainloop hooks are created
        8. NetworkManager is created
        9. mainloop is initiated
        """

        self.net = None
        self._output_root = output_root

        try:
            self._load_config(config_file=config_file, additional_args=cli_options)
        except Exception as e:
            logging.error('Loading config failed: %s', e)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        try:
            self.output_dir = self._create_output_dir()
        except Exception as e:
            logging.error('Creating output dir failed: %s', e)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        try:
            self.dumped_config_file = self._dump_config()
        except Exception as e:
            logging.error('Saving modified config failed: %s', e)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        try:
            self._create_dataset()
        except Exception as e:
            logging.error('Creating dataset failed: %s', e)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        try:
            self._create_network()
        except Exception as e:
            logging.error('Creating network failed: %s', e)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        try:
            hooks = self._construct_hooks()
        except Exception as e:
            logging.error('Creating hooks failed: %s', e)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        try:
            logging.info('Creating NetworkManager')
            manager = NetworkManager(net=self.net,
                                     dataset=self.dataset,
                                     hooks=hooks)
        except Exception as e:
            logging.error('Creating NetworkManager failed: %s', e)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        try:
            logging.info('Running main loop')
            manager.run_main_loop(run_test_stream='test' in self.config['stream'])
        except Exception as e:
            logging.error('Running the main loop failed: %s', e)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)


def init_entry_point() -> None:
    sys.path.insert(0, os.getcwd())

    parser = ArgumentParser('cxflow')
    subparsers = parser.add_subparsers(help='cxflow modes')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase verbosity do level DEBUG')

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(subcommand='train')
    train_parser.add_argument('config_file', help='path to the config file')
    train_parser.add_argument('-o', '--output-root', default='log', help='output directory')

    crossval_split_parser = subparsers.add_parser('xval-init')
    crossval_split_parser.set_defaults(subcommand='xval-init')
    crossval_split_parser.add_argument('-s', '--seed', type=int, default=100003, help='split seed')

    known_args, unknown_args = parser.parse_known_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG if known_args.verbose else logging.INFO)

    entry_point = EntryPoint()
    if known_args.subcommand == 'train':
        entry_point.train(config_file=known_args.config_file,
                          output_root=known_args.output_root,
                          cli_options=unknown_args)

    elif known_args.subcommand == 'xval-init':
        raise NotImplementedError()

    else:
        logging.error('Unrecognized subcommand: "%s". Please run `cxflow -h` for more info.', known_args.subcommand)
        sys.exit(1)


if __name__ == '__main__':
    init_entry_point()
