"""
This module contains `BaseDataset` which might be used as a base class for your dataset written in Python.
"""

from abc import abstractmethod, ABCMeta

import yaml


class BaseDataset(metaclass=ABCMeta):
    """
    Base class for datasets written in python.

    In the inherited class, one should:
        - override the ``_init_with_kwargs`` method instead of ``__init__``
        - override the ``train_stream`` method
        - add any additional ``<stream_name>_stream` method in order to make `<stream_name>` stream available
    """

    def __init__(self, config_str: str):
        """
        Create new dataset.
        Decode the given YAML config string and pass the obtained kwargs to the ``_init_with_kwargs`` method.

        :param config_str: dataset configuration as yaml string
        """
        config = yaml.load(config_str)
        self._init_with_kwargs(**config)

    @abstractmethod
    def _init_with_kwargs(self, output_dir, **kwargs):
        """
        Initialize the dataset with ``kwargs``.

        :param output_dir: output directory for logging and any additional outputs (None if no output dir is available)
        :param kwargs: dataset configuration as ``**kwargs`` parsed from ``config['dataset']``
        """
        raise NotImplementedError('Dataset does not implement obligatory `_init_with_kwargs` method.')
