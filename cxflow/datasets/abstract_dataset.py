import numpy as np

from abc import abstractmethod
import typing


class AbstractDataset:
    """cxflow dataset interface."""

    @abstractmethod
    def create_train_stream(self) -> typing.Iterable[typing.Mapping[str, np.ndarray]]:
        """Return a train iterator which provides a mapping (source name -> value)"""
        pass

    @abstractmethod
    def create_valid_stream(self) -> typing.Iterable[typing.Mapping[str, np.ndarray]]:
        """Return a valid iterator which provides a mapping (source name -> value)"""
        pass

    @abstractmethod
    def create_test_stream(self) -> typing.Iterable[typing.Mapping[str, np.ndarray]]:
        """Return a test iterator which provides a mapping (source name -> value)"""
        pass
