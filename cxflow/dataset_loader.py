from .datasets.abstract_dataset import AbstractDataset

import logging
import importlib
import typing


class DatasetLoader:
    """
    Entity responsible for correct dataset loading.

    For each possible backend there must be implemented a method named `_load_<backend_name>`.
    """

    def __init__(self, config: typing.Mapping[str, typing.Any], dumped_cofig_path: str):
        """
        Create the loader.
        :param config: config dict
        :param dumped_cofig_path: path to the dumped config YAML
        """

        self._config = config
        self._dumped_cofig_path = dumped_cofig_path

        try:
            self._config['dataset']['dataset_module']
            self._config['dataset']['dataset_class']
            self._config['dataset']['backend']
        except KeyError as e:
            logging.error('Dataset does not contain `dataset_module` or `dataset_class` or `backend`.')
            raise e

        logging.debug('Loading dataset module')
        dataset_module = importlib.import_module(self._config['dataset']['dataset_module'])

        logging.debug('Loading dataset class')
        self._dataset_class = getattr(dataset_module, self._config['dataset']['dataset_class'])

        self._load_f = getattr(self, '_load_{}'.format(self._config['dataset']['backend']))

    def load_dataset(self) -> AbstractDataset:
        """Load the dataset by using the proper backend."""

        logging.debug('Loading the dataset')
        return self._load_f()

    def _load_fuel(self) -> AbstractDataset:
        """Load the dataset by using Fuel (Python) backend"""

        logging.debug('Using fuel backend')
        return self._dataset_class(**self._config['dataset'], **self._config['stream'])

    def _load_cxtream(self) -> AbstractDataset:
        """Load the dataset by using cxtream (C++) backend"""

        logging.debug('Using cxtream backend')
        return self._dataset_class(self._dumped_cofig_path)
