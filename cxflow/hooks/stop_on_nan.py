"""
Module with StopOnNaN hook.
"""
import logging
from typing import Iterable, Optional

import numpy as np

from cxflow.types import EpochData
from cxflow.hooks import AbstractHook, TrainingTerminated


class StopOnNaN(AbstractHook):
    """
    Stop the training when any of the specified variables contain NaN.

    .. code-block:: yaml
        :caption: stop as soon as any variable contains NaN

        hooks:
          - StopOnNaN

    .. code-block:: yaml
        :caption: stop on NaN in loss variable

        hooks:
          - StopOnNan:
              variables: [loss]

    """

    UNKNOWN_TYPE_ACTIONS = ['error', 'warn', 'ignore']
    """Posible actions to take on unknown variable type."""

    def __init__(self, variables: Optional[Iterable[str]]=None, on_unknown_type: str='ignore', stop_on_inf: bool=False,
                 after_batch: bool=False, after_epoch: bool=True, **kwargs):
        """
        Create new StopOnNaN hook.

        :param variables: variable names to be checked; check all variables in ``epoch_data`` by default
        :param on_unkown_type: option for handling unknown data types, possible options are ``'warn'``, ``'error'`` and \
               default ``'ignore'``
        :param stop_on_inf: if `True` consider infinity values as NaN, default is `False`
        :param after_batch: check data after each batch? default is `False`
        :param after_epoch: check data after each epoch? default is `True`
        :raise AssertionError: for undefined ``on_unknown_type``
        :raise AssertionError: if both ``after_batch`` and ``after_epoch`` are `False`
        """

        assert after_batch or after_epoch
        assert on_unknown_type in StopOnNaN.UNKNOWN_TYPE_ACTIONS

        self._variables = variables
        self._on_unkown_type = on_unknown_type
        self._stop_on_inf = stop_on_inf
        self._after_batch = after_batch
        self._after_epoch = after_epoch
        super().__init__(**kwargs)

    def _is_nan(self, variable: str, data) -> bool:
        """
        Recursively search passed data and find NaNs.

        :param variable: name of variable to be checked
        :param data: data object (dict, list, scalar)
        :return: `True` if there is a NaN value in the data; `False` otherwise.
        :raise ValueError: if the variable value is of unsupported type and ``on_unknown_type`` is set to ``error``
        """
        if isinstance(data, np.ndarray) or isinstance(data, list):
            return any(np.isnan(data)) or (self._stop_on_inf and any(np.isinf(data)))
        elif np.isscalar(data):
            return np.isnan(data) or (self._stop_on_inf and np.isinf(data))
        elif isinstance(data, dict):
            return any([self._is_nan(key, value) for key, value in data.items()])
        else:
            message = 'Variable `{}` of type `{}` can not be checked for NaNs.'.format(variable, type(data))
            if self._on_unkown_type == 'warn':
                logging.warning(message)
            elif self._on_unkown_type == 'error':
                raise ValueError(message)
            return False

    def _check_nan(self, epoch_data: EpochData) -> None:
        """
        Raise an exception when some of the monitored data is NaN.

        :param epoch_data: epoch data checked
        :raise KeyError: if the specified variable is not found in the stream
        :raise ValueError: if the variable value is of unsupported type and ``self._on_unknown_type`` is set to ``error``
        """
        for stream_name in epoch_data.keys():
            stream_data = epoch_data[stream_name]
            variables = self._variables if self._variables is not None else stream_data.keys()
            for variable in variables:
                if variable not in stream_data:
                    raise KeyError('Variable `{}` to be nan-checked was not found in the batch data for stream `{}`. '
                                   'Available variables are `{}`.'.format(variable, stream_name, stream_data.keys()))

                value = stream_data[variable]
                if self._is_nan(variable, value):
                    raise TrainingTerminated('Variable `{}` is NaN.'.format(variable))

    def after_epoch(self, epoch_data: EpochData, **kwargs) -> None:
        """
        If initialized to check after each epoch, stop the training once the epoch data contains a monitored
        variable equal to NaN.

        :param epoch_data: epoch data to be checked
        """

        if self._after_epoch:
            self._check_nan(epoch_data)

    def after_batch(self, stream_name: str, batch_data) -> None:
        """
        If initialized to check after each batch, stop the training once the batch data contains a monitored
        variable equal to NaN.

        :param stream_name: name of the stream to be checked
        :param batch_data: batch data to be checked
        """

        if self._after_batch:
            self._check_nan({stream_name: batch_data})
