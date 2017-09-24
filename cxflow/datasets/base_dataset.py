"""
This module contains :py:class:`cxflow.datasets.BaseDataset` which might be used as a base class for your
dataset implemented in Python.
"""

from abc import abstractmethod, ABCMeta
from typing import Optional

import yaml

from .abstract_dataset import AbstractDataset


class BaseDataset(AbstractDataset, metaclass=ABCMeta):
    """
    Base class for datasets written in python.

    In the inherited class, one should:
        - override the ``_configure_dataset``
        - (optional) implement ``train_stream`` method if intended to be used with ``cxflow train ...``
        - (optional) implement ``predict_stream`` method if intended to be used with ``cxflow predict ...``
        - (optional) implement ``<stream_name>_stream`` method in order to make ``<stream_name>`` stream available

    """

    def __init__(self, config_str: str):
        """
        Create new dataset.

        Decode the given YAML config string and pass the obtained ``**kwargs`` to :py:meth:`_configure_dataset`.

        :param config_str: dataset configuration as YAML string
        """
        super().__init__(config_str)

        config = yaml.load(config_str)
        self._configure_dataset(**config)

    @abstractmethod
    def _configure_dataset(self, output_dir: Optional[str], **kwargs):
        """
        Configure the dataset with ``**kwargs`` decoded from YAML configuration.

        :param output_dir: output directory for logging and any additional outputs (None if no output dir is available)
        :param kwargs: dataset configuration as ``**kwargs`` parsed from ``config['dataset']``
        :raise NotImplementedError: if not overridden
        """
