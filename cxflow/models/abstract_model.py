"""
This module contains the definition of a model trainable in **cxflow** framework.
"""
from abc import abstractmethod, ABCMeta
from typing import Iterable, Optional

from ..datasets import AbstractDataset, StreamWrapper
from ..types import Batch


class AbstractModel(metaclass=ABCMeta):
    """
    Abstract machine learning model which exposes input and output names, run and save methods.
    `AbstractModel` implementations are trainable with :py:class:`cxflow.MainLoop`.
    """

    @abstractmethod
    def __init__(self, dataset: Optional[AbstractDataset], log_dir: str, restore_from: Optional[str]=None, **kwargs):
        """
        Model constructor interface.

        Additional parameters (currently covered by ``**kwargs``) are passed
        according to the configuration ``model`` section.

        :param dataset: dataset object
        :param log_dir: existing directory in which all output files should be stored
        :param restore_from: information passed to the model constructor (backend-specific);
                             usually a directory in which the trained model is stored
        :param kwargs: configuration section ``model``
        """
        pass

    @property
    @abstractmethod
    def input_names(self) -> Iterable[str]:
        """List of model input names."""
        pass

    @property
    @abstractmethod
    def output_names(self) -> Iterable[str]:
        """List of model output names."""
        pass

    @abstractmethod
    def run(self, batch: Batch, train: bool, stream: StreamWrapper) -> Batch:
        """
        Run feed-forward pass with the given batch and return the results as dict.

        When ``train=True``, also update parameters.

        :param batch: batch to be processed
        :param train: ``True`` if this batch should be used for model update, ``False`` otherwise
        :param stream: stream wrapper (useful for precise buffer management)
        :return: results dict
        """
        pass

    @abstractmethod
    def save(self, name_suffix: str) -> str:
        """
        Save the model parameters with the given ``name_suffix``.

        :param name_suffix: name suffix to be appended to the saved model
        :return: path to the saved file/dir
        """
        pass

    @property
    @abstractmethod
    def restore_fallback(self) -> str:
        """
        Return the fully-qualified name of the fallback restore class (e.g. ``module.submodule.BaseClass``).

        When restoring a model, **cxflow** tries to use the fallback class if the construction of the model
        object specified in `model` configuration section fails.

        :return: fully-qualified name of the fallback restore class
        """
        pass
