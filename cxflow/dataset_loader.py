import logging
import importlib


class DatasetLoader:
    def __init__(self, config):
        self.config = config

        logging.debug('Loading dataset module')
        dataset_module = importlib.import_module(self.config['dataset']['dataset_module'])

        logging.debug('Loading dataset class')
        self.dataset_class = getattr(dataset_module, self.config['dataset']['dataset_class'])

        self.load_f = getattr(self, '_load_{}'.format(self.config['dataset']['backend']))

    def load_dataset(self):
        return self.load_f()

    def _load_fuel(self):
        logging.debug('load fuel')
        dataset =  self.dataset_class(**self.config['dataset'], **self.config['stream'])
        return dataset

    def _load_cxtream(self):
        logging.debug('cxtream')
