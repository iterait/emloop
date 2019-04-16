"""
This module contains :py:class:`emloop.datasets.BaseDataset` which might be used as a base class for your
dataset implemented in Python.
"""
import logging
from abc import abstractmethod, ABCMeta
from typing import Optional
from collections import namedtuple
import traceback

import ruamel.yaml
import tabulate
import numpy as np

from .abstract_dataset import AbstractDataset


class BaseDataset(AbstractDataset, metaclass=ABCMeta):
    """
    Base class for datasets written in python.

    In the inherited class, one should:
        - override the ``_configure_dataset``
        - (optional) implement ``train_stream`` method if intended to be used with ``emloop train ...``
        - (optional) implement ``<stream_name>_stream`` method in order to make ``<stream_name>`` stream available

    """

    def __init__(self, config_str: str):
        """
        Create new dataset.

        Decode the given YAML config string and pass the obtained ``**kwargs`` to :py:meth:`_configure_dataset`.

        :param config_str: dataset configuration as YAML string
        """
        super().__init__(config_str)

        config = ruamel.yaml.load(config_str, ruamel.yaml.RoundTripLoader)
        self._configure_dataset(**config)

    @abstractmethod
    def _configure_dataset(self, output_dir: Optional[str], **kwargs):
        """
        Configure the dataset with ``**kwargs`` decoded from YAML configuration.

        :param output_dir: output directory for logging and any additional outputs (None if no output dir is available)
        :param kwargs: dataset configuration as ``**kwargs`` parsed from ``config['dataset']``
        :raise NotImplementedError: if not overridden
        """

    def stream_info(self) -> None:
        """Check and report source names, dtypes and shapes of all the streams available."""
        stream_names = [stream_name for stream_name in dir(self)
                        if 'stream' in stream_name and stream_name != 'stream_info']
        logging.info('Found %s stream candidates: %s', len(stream_names), stream_names)
        for stream_name in stream_names:
            try:
                stream_fn = getattr(self, stream_name)
                logging.info(stream_name)
                for batch in stream_fn():
                    rows = []
                    for key, value in batch.items():
                        try:
                            value_arr = np.array(value)
                            row = [key, value_arr.dtype, value_arr.shape]
                            if value_arr.dtype.kind in 'bui':  # boolean, unsigned, integer
                                row.append('{} - {}'.format(value_arr.min(), value_arr.max()))
                            elif value_arr.dtype.kind is 'f':
                                row.append('{0:.2f} - {1:.2f}'.format(value_arr.min(), value_arr.max()))
                        except ValueError:  # np broadcasting failed (ragged array)
                            value_arr = None
                            row = [key, '{}'.format(type(value[0]).__name__), '({},)'.format(len(list(value)))]

                        if value_arr is None or \
                                (value_arr.ndim > 0 and value_arr.shape[1:] != np.array(value_arr[0]).shape):
                            logging.warning('*** stream source `%s` appears to be ragged (non-rectangular) ***', key)

                        rows.append(row)
                    for line in tabulate.tabulate(rows, headers=['name', 'dtype', 'shape', 'range'],
                                                  tablefmt='grid').split('\n'):
                        logging.info(line)
                    break
            except Exception:
                logging.warning('Exception was raised during checking stream `%s`, '
                                '(stack trace is displayed only with --verbose flag)', stream_name)
                logging.debug(traceback.format_exc())
