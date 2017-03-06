import numpy as np

from abc import abstractmethod
import typing


class AbstractDataset:
    """cxflow dataset interface."""

    Stream = typing.NewType('Stream', typing.Iterable[typing.Mapping[str, typing.Any]])

    @abstractmethod
    def create_train_stream(self) -> Stream:
        """Return a train iterator which provides a mapping (source name -> value)"""
        pass

    @abstractmethod
    def create_valid_stream(self) -> Stream:
        """Return a valid iterator which provides a mapping (source name -> value)"""
        pass

    @abstractmethod
    def create_test_stream(self) -> Stream:
        """Return a test iterator which provides a mapping (source name -> value)"""
        pass
