from .datasets.abstract_dataset import AbstractDataset

import logging
import importlib


class DatasetLoader:
    """
    Entity responsible for correct dataset loading.

    For each possible backend there must be implemented a method named `_load_<backend_name>`.
    """

    def __init__(self, config: dict, dumped_cofig_path: str):
        """
        Create the loader.
        :param config: config dict
        :param dumped_cofig_path: path to the dumped config YAML
        """

        self.config = config
        self.dumped_cofig_path = dumped_cofig_path

        logging.debug('Loading dataset module')
        dataset_module = importlib.import_module(self.config['dataset']['dataset_module'])

        logging.debug('Loading dataset class')
        self.dataset_class = getattr(dataset_module, self.config['dataset']['dataset_class'])

        self.load_f = getattr(self, '_load_{}'.format(self.config['dataset']['backend']))

    @staticmethod
    def _verify_dataset(dataset):
        """Verify the passed dataset implements the interface of AbstractDataset."""

        for method_name in dir(AbstractDataset):
            if callable(getattr(AbstractDataset, method_name)):
                try:
                    method = getattr(dataset, method_name)
                    if not callable(method):
                        raise ValueError()
                except:
                    logging.error('Dataset does not contain callable method "%s"', method_name)
                    raise ValueError()

    def load_dataset(self) -> AbstractDataset:
        """Load the dataset by using the proper backend."""

        logging.debug('Loading the dataset')
        dataset =  self.load_f()

        logging.debug('Checking the dataset')
        DatasetLoader._verify_dataset(dataset)

        return dataset

    def _load_fuel(self) -> AbstractDataset:
        """Load the dataset by using Fuel (Python) backend"""

        logging.debug('Using fuel backend')
        return self.dataset_class(**self.config['dataset'], **self.config['stream'])

    def _load_cxtream(self) -> AbstractDataset:
        """Load the dataset by using cxtream (C++) backend"""

        logging.debug('Using cxtream backend')
        return self.dataset_class(self.dumped_cofig_path)
