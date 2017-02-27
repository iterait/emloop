#!/usr/bin/python3 -msrc.entry_point

from .network_manager import NetworkManager
from .utils.stream_types import StreamTypes

import ast
from argparse import ArgumentParser
from collections import defaultdict
import importlib
import json
import logging
import os
from os import path
import sys
import typing


class EntryPoint:

    @staticmethod
    def _parse_arg(arg: str):
        assert '=' in arg

        if ':' in arg:
            key = arg[:arg.index(':')]
            typee = arg[arg.index(':')+1:arg.index('=')]
            value = arg[arg.index('=')+1:]
        else:
            key = arg[:arg.index('=')]
            typee = 'str'
            value = arg[arg.index('=')+1:]

        try:
            if typee == 'ast':
                value = ast.literal_eval(value)
            elif typee == 'int':
                value = int(float(value))
            elif typee == 'bool':
                value = bool(int(value))
            else:
                value = eval(typee)(value)
        except (Exception, AssertionError) as e:
            logging.error('Couldn\'t convert argument %s of value %s to type %s. Original argument: "%s". Exception: %s',
                          key, value, typee, arg, e)
            raise AttributeError('Couldn\'t convert argument {} of value {} to type {}. Original argument: "{}". Exception: {}'.format(
                                 key, value, typee, arg, e))

        return key, value

    def __init__(self):
        parser = ArgumentParser('cxflow')
        parser.add_argument('config_file', help='relative path to the config file')
        known_args, unknown_args = parser.parse_known_args()

        self.net = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self._load_config(config_file=known_args.config_file, additional_args=unknown_args)
        self._create_streams()
        self._create_network()
        self._dump_config()

    def _load_config(self, config_file: str, additional_args: typing.Iterable[str]):
        with open(config_file, 'r') as f:
            self.config = defaultdict(defaultdict, json.load(f))
        for key_full, value in [EntryPoint._parse_arg(arg) for arg in additional_args]:
            key_split = key_full.split('.')
            key_prefix = key_split[:-1]
            key = key_split[-1]

            conf = self.config
            for key_part in key_prefix:
                conf = conf[key_part]
            conf[key] = value

    def _create_streams(self):
        logging.info('Creating datasets')

        logging.debug('Loading dataset module')
        dataset_module = importlib.import_module(self.config['stream']['common']['dataset_module'])

        logging.debug('Loading dataset class')
        dataset_class = getattr(dataset_module, self.config['stream']['common']['dataset_class'])

        if 'train' in self.config['stream']:
            logging.debug('Constructing train dataset')
            train_config = {**dict(),
                            **(self.config['common'] if 'common' in self.config else dict()),
                            **(self.config['stream']['common'] if 'common' in self.config['stream'] else dict()),
                            **(self.config['stream']['train'] if 'train' in self.config['stream'] else dict()),
                           }
            self.train_dataset = dataset_class(stream_type=StreamTypes.TRAIN, **train_config)

        if 'valid' in self.config['stream']:
            logging.debug('Constructing valid dataset')
            valid_config = {**dict(),
                            **(self.config['common'] if 'common' in self.config else dict()),
                            **(self.config['stream']['common'] if 'common' in self.config['stream'] else dict()),
                            **(self.config['stream']['valid'] if 'valid' in self.config['stream'] else dict()),
                           }
            if self.train_dataset:
                if hasattr(self.train_dataset, 'REUSE_FROM_TRAIN'):
                    for key in self.train_dataset.REUSE_FROM_TRAIN:
                        valid_config[key] = getattr(self.train_dataset, key)

            self.valid_dataset = dataset_class(stream_type=StreamTypes.VALID, **valid_config)

        if 'test' in self.config['stream']:
            logging.debug('Constructing test dataset')
            test_config = {**dict(),
                           **(self.config['common'] if 'common' in self.config else dict()),
                           **(self.config['stream']['common'] if 'common' in self.config['stream'] else dict()),
                           **(self.config['stream']['test'] if 'test' in self.config['stream'] else dict()),
                          }
            if self.train_dataset:
                if hasattr(self.train_dataset, 'REUSE_FROM_TRAIN'):
                    for key in self.train_dataset.REUSE_FROM_TRAIN:
                        test_config[key] = getattr(self.train_dataset, key)

            self.test_dataset = dataset_class(stream_type=StreamTypes.TEST, **test_config)

    def _create_network(self):
        logging.debug('Creating net')
        try:
            self.config['net']['vocabulary_size'] = len(self.train_dataset.token2id)
        except:
            logging.warning('Train dataset does not contain token2id')

        logging.debug('Loading net module')
        net_module = importlib.import_module(self.config['net']['net_module'])

        logging.debug('Loading net class')
        net_class = getattr(net_module, self.config['net']['net_class'])

        logging.debug('Constructing net')
        net_config = {**dict(),
                      **(self.config['common'] if 'common' in self.config else dict()),
                      **(self.config['net'] if 'net' in self.config else dict())
                      }
        self.net = net_class(**net_config)

    def _dump_config(self):
        with open(path.join(self.net.log_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f)

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

                hook_processed_config = {
                    **dict(),
                    **(self.config['common'] if 'common' in self.config else dict()),
                    **hook_conf
                }

                hooks.append(hook_class(net=self.net, config=self.config, **hook_processed_config))
        else:
            logging.warning('No hooks found')

        logging.info('Creating NetworkManager')
        manager = NetworkManager(net=self.net,
                                 train_dataset=self.train_dataset,
                                 valid_dataset=self.valid_dataset,
                                 test_dataset=self.test_dataset,
                                 hooks=hooks)
        logging.info('Running main loop')
        manager.run_main_loop(batch_size=self.config['common']['batch_size'])


def init_entry_point():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    sys.path.insert(0, os.getcwd())

    entry_point = EntryPoint()
    entry_point.run()


if __name__ == '__main__':
    init_entry_point()
