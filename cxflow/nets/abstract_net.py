"""
This module contains the definition of a net trainable in cxflow framework.
"""
from abc import abstractmethod, ABCMeta
from typing import Iterable, Mapping


class AbstractNet(metaclass=ABCMeta):
    """
    Abstract neural network which exposes input and output names, run and save methods.
    AbstractNet implementations are trainable with cxflow main_loop.
    """

    @property
    @abstractmethod
    def input_names(self) -> Iterable[str]:
        """
        List of net input names.
        """
        pass

    @property
    @abstractmethod
    def output_names(self) -> Iterable[str]:
        """
        List of net output names.
        """
        pass

    @abstractmethod
    def run(self, batch: Mapping[str, object], train: bool) -> Mapping[str, object]:
        """
        Run feed-forward pass with the given batch and return the results as dict.

        When train=True, also update parameters.
        :return: results dict
        """
        pass

    @abstractmethod
    def save(self, name_suffix: str) -> str:
        """
        Save the net parameters with the given name_suffix.
        :return: path to the saved file/dir
        """
        pass
