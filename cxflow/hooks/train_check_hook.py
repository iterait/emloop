"""
Module with a hook that checks if the specified variable reached the expected level.
"""
import numpy as np

from cxflow.hooks.abstract_hook import AbstractHook, TrainingTerminated


class TrainCheckHook(AbstractHook):
    """
    Terminate training if the given stream variable exceeds the threshold in at most specified number of epochs.

    Raise `ValueError` on error or when the threshold was not exceeded in given number of epochs

    -------------------------------------------------------
    Example usage in config
    -------------------------------------------------------
    # exceed 95% accuracy on valid (default) stream within at most 10 epochs
    hooks:
      - class: TrainCheckHook
        variable: accuracy
        required_min_value: 0.93
        max_epoch: 10
    -------------------------------------------------------
    """

    def __init__(self, variable: str, required_min_value: float, max_epoch: int, stream: str='valid', **kwargs):
        """
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

    def after_epoch(self, epoch_id: int, epoch_data: AbstractHook.EpochData):
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
            raise ValueError('The value to be checked has to be either a scalar or a dict with `mean` key. '
                             'Got `{}` instead.'.format(type(value).__name__))

        if value > self._required_min_value:
            raise TrainingTerminated('{} {} level matched (current {} is greater than required {}).'
                                     .format(self._stream, self._variable, value, self._required_min_value))
        elif epoch_id >= self._max_epoch:
            raise ValueError('{} {} was only {} in epoch {}, but {} was required. Training failed.'
                             .format(self._stream, self._variable, value, epoch_id, self._required_min_value))
