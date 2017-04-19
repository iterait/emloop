"""
This module contains BaseDataset which might be used as a base class for your dataset written in python.
"""
from abc import abstractmethod, ABCMeta

import yaml

from .abstract_dataset import AbstractDataset


class BaseDataset(metaclass=ABCMeta):
    """
    Base class for datasets written in python.

    In the inherited class, one should:
        - override the _init_with_kwargs method insead of __init__
        - override the create_train_stream method
        - override any additional create_[stream_name]_stream method in order to make [stream_name] stream available
    """

    def __init__(self, config_str: str):
        """
        Create new dataset.
        Decode the given yaml config string and pass the obtained kwargs to the _init_with_kwargs method.
        :param config_str: dataset configuration as yaml string
        """
        config = yaml.load(config_str)

        assert 'dataset' in config

        output_dir = config['output_dir'] if 'output_dir' in config else None

        self._init_with_kwargs(output_dir=output_dir, **config['dataset'])

    @abstractmethod
    def _init_with_kwargs(self, output_dir, **kwargs):
        """
        Initialize the dataset with kwargs.

        :param output_dir: output_dir for logging and any additional outputs (None if no output dir is available)
        :param kwargs: dataset configuration as **kwargs parsed from config['dataset']
        """
        raise NotImplementedError('Dataset does not implement obligatory _init_with_kwargs method.')

    @abstractmethod
    def create_train_stream(self) -> AbstractDataset.Stream:
        """Get the train stream iterator."""
        raise NotImplementedError('Dataset does not implement obligatory create_train_stream method.')

    def create_valid_stream(self) -> AbstractDataset.Stream:
        """Get the valid stream iterator."""
        raise NotImplementedError('Dataset does not implement create_valid_stream method although it is now required.')

    def create_test_stream(self) -> AbstractDataset.Stream:
        """Get the test stream iterator."""
        raise NotImplementedError('Dataset does not implement create_test_stream method although it is now required.')

    def split(self, num_splits: int, train: float, valid: float, test: float) -> None:
        """
        Perform cross-validation split with the given parameters.
        :param num_splits: the number of train-valid splits to be created (sharing the test set)
        :param train: portion of data to be split to train
        :param valid: portion of data to be split to valid
        :param test: portion of data to be split to test
        """
        raise NotImplementedError('Dataset does not implement split method although it is now required.')
