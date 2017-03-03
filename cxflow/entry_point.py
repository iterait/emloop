#!/usr/bin/python3 -msrc.entry_point

from .network_manager import NetworkManager
from .utils.arg_parser import parse_arg
from .utils.stream_types import StreamTypes
from .dataset_loader import DatasetLoader

from argparse import ArgumentParser
from collections import defaultdict
import importlib
import logging
import os
from os import path
import sys
import typing
import yaml


class EntryPoint:

    def __init__(self):
        parser = ArgumentParser('cxflow')
        parser.add_argument('config_file', help='relative path to the config file')
        known_args, unknown_args = parser.parse_known_args()

        self.net = None

        self._load_config(config_file=known_args.config_file, additional_args=unknown_args)
        self._create_dataset()
        self._create_network()
        self._dump_config()

    def _load_config(self, config_file: str, additional_args: typing.Iterable[str]):
        with open(config_file, 'r') as f:
            # self.config = defaultdict(defaultdict, yaml.load(f))
            self.config = yaml.load(f)

        # TODO: fix CLI: http://yaml.readthedocs.io/en/latest/detail.html#adding-replacing-comments
        for key_full, value in [parse_arg(arg) for arg in additional_args]:
            key_split = key_full.split('.')
            key_prefix = key_split[:-1]
            key = key_split[-1]

            conf = self.config
            for key_part in key_prefix:
                conf = conf[key_part]
            conf[key] = value

    def _create_dataset(self):
        data_loader = DatasetLoader(self.config)
        self.dataset = data_loader.load_dataset()

    def _create_network(self):
        logging.info('Creating net')

        logging.debug('Loading net module')
        net_module = importlib.import_module(self.config['net']['net_module'])

        logging.debug('Loading net class')
        net_class = getattr(net_module, self.config['net']['net_class'])

        logging.debug('Constructing net')
        self.net = net_class(dataset=self.dataset, **self.config['net'])

    def _dump_config(self):
        with open(path.join(self.net.log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)

    def run(self):
        hooks = []
        if 'hooks' in self.config:
            logging.info('Creating hooks')
            for hook_conf in self.config['hooks']:
                if 'hook_module' not in hook_conf or 'hook_class' not in hook_conf:
                    logging.error('All hook configs must contain hook_module and hook_class')
                    quit()

                logging.debug('Loading hook module %s', hook_conf['hook_module'])
                hook_module = importlib.import_module(hook_conf['hook_module'])

                logging.debug('Loading hook class %s', hook_conf['hook_class'])
                hook_class = getattr(hook_module, hook_conf['hook_class'])

                hooks.append(hook_class(net=self.net, config=self.config, **hook_conf))
        else:
            logging.warning('No hooks found')

        logging.info('Creating NetworkManager')
        manager = NetworkManager(net=self.net,
                                 dataset=self.dataset,
                                 hooks=hooks)
        logging.info('Running main loop')
        manager.run_main_loop(batch_size=self.config['net']['batch_size'])


def init_entry_point():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    sys.path.insert(0, os.getcwd())

    entry_point = EntryPoint()
    entry_point.run()


if __name__ == '__main__':
    init_entry_point()
