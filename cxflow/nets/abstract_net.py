"""
This module contains the definition of a net trainable in cxflow framework.
"""
from abc import abstractmethod, ABCMeta
from typing import Iterable, Mapping, Optional

from ..datasets import AbstractDataset

class AbstractNet(metaclass=ABCMeta):
    """
    Abstract neural network which exposes input and output names, run and save methods.
    AbstractNet implementations are trainable with cxflow main_loop.
    """

    @abstractmethod
    def __init__(self, dataset: Optional[AbstractDataset], log_dir: str, restore_from: Optional[str]=None, **kwargs):
        pass

    @property
    @abstractmethod
    def input_names(self) -> Iterable[str]:
        """List of net input names."""
        pass

    @property
    @abstractmethod
    def output_names(self) -> Iterable[str]:
        """List of net output names."""
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

    @property
    @abstractmethod
    def restore_fallback_module(self) -> str:
        """
        Return the module name with fallback restore class.

        When restoring a model, cxflow tries to use the fallback class if the specified `net.class` fails to do so.
        :return: fallback restore module name
        """
        pass

    @property
    @abstractmethod
    def restore_fallback_class(self) -> str:
        """
        Return the fallback restore class name.

        When restoring a model, cxflow tries to use the fallback class if the specified `net.class` fails to do so.
        :return: fallback restore class name
        """
        pass
