from .datasets.abstract_dataset import AbstractDataset

import logging
import importlib


class DatasetLoader:
    def __init__(self, config: dict, dumped_cofig_path: str):
        self.config = config
        self.dumped_cofig_path = dumped_cofig_path

        logging.debug('Loading dataset module')
        dataset_module = importlib.import_module(self.config['dataset']['dataset_module'])

        logging.debug('Loading dataset class')
        self.dataset_class = getattr(dataset_module, self.config['dataset']['dataset_class'])

        self.load_f = getattr(self, '_load_{}'.format(self.config['dataset']['backend']))

    def load_dataset(self) -> AbstractDataset:
        return self.load_f()

    def _load_fuel(self) -> AbstractDataset:
        logging.debug('load fuel')
        return self.dataset_class(**self.config['dataset'], **self.config['stream'])

    def _load_cxtream(self) -> AbstractDataset:
        logging.debug('Using cxtream backend')
        logging.debug('Dumped config path: %s', self.dumped_cofig_path)
        logging.error('Not implemented yet')
        quit()
