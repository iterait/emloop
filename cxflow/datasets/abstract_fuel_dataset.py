from .abstract_dataset import AbstractDataset

import yaml

from abc import abstractmethod, ABCMeta


class AbstractFuelDataset(metaclass=ABCMeta):
    """
    Fuel dataset ancestor which automatically handles train/valid/test config.
    Note that `create_train_stream`, `create_valid_stream` and `create_test_stream` must not be modified. Instead, their
    underscored versions should be used.
    """

    def __init__(self, config_str: str):
        """Save all kwargs"""

        config = yaml.load(config_str)

        assert('dataset' in config)
        assert('stream' in config)

        output_dir = config['output_dir'] if 'output_dir' in config else None

        self._init_with_kwargs(output_dir=output_dir, **config['dataset'], **config['stream'])

        assert('train' in config['stream'])
        assert('valid' in config['stream'])

        self._train_config = config['stream']['train']
        self._valid_config = config['stream']['valid']

        if 'test' in config['stream']:
            self._test_config = config['stream']['test']

    def create_train_stream(self) -> AbstractDataset.Stream:
        if not hasattr(self, '_train_config'):
            raise ValueError('Fuel dataset does not have train config initialized.')
        return self._create_train_stream(**self._train_config)

    def create_valid_stream(self) -> AbstractDataset.Stream:
        if not hasattr(self, '_valid_config'):
            raise ValueError('Fuel dataset does not have valid config initialized.')
        return self._create_valid_stream(**self._valid_config)

    def create_test_stream(self) -> AbstractDataset.Stream:
        if not hasattr(self, '_test_config'):
            raise ValueError('Fuel dataset does not have test config initialized.')
        return self._create_test_stream(**self._test_config)

    @abstractmethod
    def _init_with_kwargs(self, **kwargs) -> None:
        """
        Dataset initialization with kwargs parsed from the yaml config string.
        :param kwargs: kwargs parsed from the yaml config
        """
        pass

    @abstractmethod
    def _create_train_stream(self, **kwargs) -> AbstractDataset.Stream:
        """
        Return a train stream iterator parametrized with kwargs.
        :param kwargs: train stream kwargs
        :return: train stream iterator
        """
        pass

    @abstractmethod
    def _create_valid_stream(self, **kwargs) -> AbstractDataset.Stream:
        """
        Return a valid stream iterator parametrized with kwargs.
        :param kwargs: valid stream kwargs
        :return: valid stream iterator
        """
        pass

    def _create_test_stream(self, **kwargs) -> AbstractDataset.Stream:
        """
        Return a test stream iterator parametrized with kwargs.
        :param kwargs: test stream kwargs
        :return: test stream iterator
        """
        raise NotImplementedError('Dataset does not implement create test stream method although it is now required.')

    def split(self, num_splits: int, train: float, valid: float, test: float) -> None:
        """
        Perform cross-validation split with the given parameters.
        :param num_splits: the number of train-valid splits to be created (sharing the test set)
        :param train: portion of data to be split to train
        :param valid: portion of data to be split to valid
        :param test: portion of data to be split to test
        """
        raise NotImplementedError('Dataset does not implement split method although it is now required.')
