from .abstract_dataset import AbstractDataset

from abc import abstractmethod, ABCMeta


class AbstractFuelDataset(metaclass=ABCMeta):
    """
    Fuel dataset ancestor which automatically handles train/valid/test config.
    Note that `create_train_stream`, `create_valid_stream` and `create_test_stream` must not be modified. Instead, their
    underscored versions should be used.
    """

    def __init__(self, train: dict, valid: dict, **kwargs):
        """Save all kwargs"""

        self.train = train
        self.valid = valid

        for name, value in kwargs.items():
            setattr(self, name, value)

    def create_train_stream(self) -> AbstractDataset.Stream:
        return self._create_train_stream(**self.train)

    def create_valid_stream(self) -> AbstractDataset.Stream:
        return self._create_valid_stream(**self.valid)


    @abstractmethod
    def _create_train_stream(self, **kwargs) -> AbstractDataset.Stream:
        """Return a train iterator which is parametrized by kwargs"""
        pass

    @abstractmethod
    def _create_valid_stream(self, **kwargs) -> AbstractDataset.Stream:
        """Return a valid iterator which is parametrized by kwargs"""
        pass


class AbstractFuelDatasetWithTest(AbstractFuelDataset, metaclass=ABCMeta):
    """Same as `AbstractFuelDataset` with test stream methods."""

    def __init__(self, test: dict, **kwargs):
        super().__init__(**kwargs)
        self.test = test

    def create_test_stream(self) -> AbstractDataset.Stream:
        return self._create_test_stream(**self.test)

    @abstractmethod
    def _create_test_stream(self, **kwargs) -> AbstractDataset.Stream:
        """Return a test iterator which is parametrized by kwargs"""
        pass
