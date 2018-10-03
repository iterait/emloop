"""
Module with a hook that checks if the specified variable reached the expected level.
"""
import numpy as np

from . import AbstractHook, TrainingTerminated
from ..types import EpochData


class Check(AbstractHook):
    """
    Terminate training if the given stream variable exceeds the threshold in at most specified number of epochs.

    Raise :py:class:`ValueError` if the threshold was not exceeded in given number of epochs

    .. code-block:: yaml
        :caption: exceed 95% accuracy on valid (default) stream within at most 10 epochs

        hooks:
          - Check:
              variable: accuracy
              required_min_value: 0.93
              max_epoch: 10

    """

    def __init__(self, variable: str, required_min_value: float, max_epoch: int, stream: str='valid', **kwargs):
        """
        Create new Check hook.

        :param variable: variable to be checked
        :param required_min_value: threshold to be exceeded
        :param max_epoch: maximum epochs to be run
        :param stream: stream to be checked
        """
        super().__init__(**kwargs)

        self._stream = stream
        self._variable = variable
        self._required_min_value = required_min_value
        self._max_epoch = max_epoch

    def after_epoch(self, epoch_id: int, epoch_data: EpochData):
        """
        Check termination conditions.

        :param epoch_id: number of the processed epoch
        :param epoch_data: epoch data to be checked
        :raise KeyError: if the stream of variable was not found in ``epoch_data``
        :raise TypeError: if the monitored variable is not a scalar or scalar ``mean`` aggregation
        :raise ValueError: if the specified number of epochs exceeded
        :raise TrainingTerminated: if the monitor variable is above the required level
        """

        if self._stream not in epoch_data:
            raise KeyError('The hook could not determine whether the threshold was exceeded as the stream `{}`'
                           'was not found in the epoch data'.format(self._stream))

        if self._variable not in epoch_data[self._stream]:
            raise KeyError('The hook could not determine whether the threshold was exceeded as the variable `{}`'
                           'was not found in the epoch data stream `{}`'.format(self._variable, self._stream))

        value = epoch_data[self._stream][self._variable]
        if isinstance(value, dict) and 'mean' in value:
            value = value['mean']

        if not np.isscalar(value):
            raise TypeError('The value to be checked has to be either a scalar or a dict with `mean` key. '
                            'Got `{}` instead.'.format(type(value).__name__))

        if value > self._required_min_value:
            raise TrainingTerminated('{} {} level matched (current {} is greater than required {}).'
                                     .format(self._stream, self._variable, value, self._required_min_value))
        elif epoch_id >= self._max_epoch:
            raise ValueError('{} {} was only {} in epoch {}, but {} was required. Training failed.'
                             .format(self._stream, self._variable, value, epoch_id, self._required_min_value))
