"""
Module with batch data accumulating hook.
"""
import typing
from collections import defaultdict

from .abstract_hook import AbstractHook
from ..datasets import AbstractDataset


class AccumulatingHook(AbstractHook):
    """
    This hook accumulates the specified variables allowing their aggregation after each epoch.

    The hook itself does not utilize the accumulated variables. It is meant to be inherited from. The child hook
    will have the accumulated variables available in `self._accumulator` after each epoch.

    The data are accumulated in a form of nested mapping 'stream_name' -> 'variable_name' -> [iterable values].

    -------------------------------------------------------
    This hook should not be used directly as it does nothing on its own.
    -------------------------------------------------------
    """

    def __init__(self, variables: typing.Iterable['str'], **kwargs):
        super().__init__(**kwargs)
        self._variables = variables
        self._accumulator = None
        self._reset_accumulator()

    def _reset_accumulator(self):
        """Set the accumulator to an empty double-index defaultdict."""
        self._accumulator = defaultdict(lambda: defaultdict(list))

    def after_batch(self, stream_name: str, batch_data: AbstractDataset.Batch):
        """
        Extend the accumulated variables with the given batch data.

        :param stream_name: stream name; `train` or any other...
        :param batch_data: batch data = stream sources + net outputs

        Raise:
            KeyError: if the variables to be aggregated are missing
            TypeError: if the variable value is not iterable (e.g. it is only a scalar)
        """
        for variable in self._variables:
            if variable in batch_data:
                value = batch_data[variable]
                if not hasattr(value, '__iter__'):
                    raise TypeError('Variable `{}` to be accumulated is not iterable.'.format(variable))
                self._accumulator[stream_name][variable] += list(value)
            else:
                raise KeyError('Variable `{}` to be accumulated was not found in the batch data. '
                               'Available variables are `{}`.'.format(variable, batch_data.keys()))

    def after_epoch(self, **_):
        """Reset the accumulator."""
        self._reset_accumulator()
