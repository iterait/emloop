from .abstract_dataset import AbstractDataset

from abc import abstractmethod


class AbstractFuelDataset(AbstractDataset):
    """
    Fuel dataset ancestor which automatically handles train/valid/test config.
    Note that `create_train_stream`, `create_valid_stream` and `create_test_stream` must not be modified. Instead, their
    underscored versions should be used.
    """

    def __init__(self, **kwargs):
        """Save all kwargs"""

        for name, value in kwargs.items():
            setattr(self, name, value)

    def create_train_stream(self):
        return self._create_train_stream(**self.train)

    def create_valid_stream(self):
        return self._create_valid_stream(**self.valid)

    def create_test_stream(self):
        return self._create_test_stream(**self.test)

    @abstractmethod
    def _create_train_stream(self, **kwargs):
        """Return a train iterator which is parametrized by kwargs"""
        pass

    @abstractmethod
    def _create_valid_stream(self, **kwargs):
        """Return a valid iterator which is parametrized by kwargs"""
        pass

    @abstractmethod
    def _create_test_stream(self, **kwargs):
        """Return a test iterator which is parametrized by kwargs"""
        pass
