#!/usr/bin/python3 -msrc.entry_point

from .network_manager import NetworkManager
from .utils.arg_parser import parse_arg
from .dataset_loader import DatasetLoader

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


class EntryPoint:

    def __init__(self):
        parser = ArgumentParser('cxflow')
        parser.add_argument('config_file', help='path to the config file')
        parser.add_argument('-o',  '--output-root', default='log', help='output directory')
        parser.add_argument('-v',  '--verbose', action='store_true', help='incease verbosity do level DEBUG')
        known_args, unknown_args = parser.parse_known_args()

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.DEBUG if known_args.verbose else logging.INFO)

        self.net = None
        self._output_root = known_args.output_root

        self._load_config(config_file=known_args.config_file, additional_args=unknown_args)
        self.output_dir = self._create_output_dir()
        self.dumped_config_file = self._dump_config()
        self._create_dataset()
        self._create_network()

    def _load_config(self, config_file: str, additional_args: typing.Iterable[str]) -> dict:
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
        return self.config

    def _create_output_dir(self) -> str:
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
        data_loader = DatasetLoader(self.config, self.dumped_config_file)
        self.dataset = data_loader.load_dataset()

    def _create_network(self) -> None:
        logging.info('Creating net')

        logging.debug('Loading net module')
        net_module = importlib.import_module(self.config['net']['net_module'])

        logging.debug('Loading net class')
        net_class = getattr(net_module, self.config['net']['net_class'])

        logging.debug('Constructing net')
        self.net = net_class(dataset=self.dataset, log_dir=self.output_dir, **self.config['net'])

    def _dump_config(self) -> str:
        dumped_config_f = path.join(self.output_dir, 'config.yaml')
        with open(dumped_config_f, 'w') as f:
            yaml.dump(self.config, f)
        return dumped_config_f

    def run(self) -> None:
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

                logging.debug('Constructing hook')
                hook = hook_class(net=self.net, config=self.config, **hook_conf)

                hooks.append(hook)
        else:
            logging.warning('No hooks found')

        logging.info('Creating NetworkManager')
        manager = NetworkManager(net=self.net,
                                 dataset=self.dataset,
                                 hooks=hooks)
        logging.info('Running main loop')
        manager.run_main_loop(batch_size=self.config['net']['batch_size'])


def init_entry_point() -> None:
    sys.path.insert(0, os.getcwd())

    entry_point = EntryPoint()
    entry_point.run()


if __name__ == '__main__':
    init_entry_point()
