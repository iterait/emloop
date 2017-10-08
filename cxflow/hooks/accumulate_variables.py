"""
Module with batch data accumulating hook.
"""
import typing
from collections import defaultdict

from . import AbstractHook
from ..types import Batch


class AccumulateVariables(AbstractHook):
    """
    Accumulate the specified variables allowing their aggregation after each epoch.

    The hook itself does not utilize the accumulated variables. It is meant to be inherited from. The child hook
    will have the accumulated variables available in ``self._accumulator`` after each epoch.

    The data are accumulated in a form of nested mapping
    ``stream_name`` -> ``variable_name`` -> ``Iterable``[``values``].

    .. warning::
        This hook should not be used directly as it does nothing on its own.
    """

    def __init__(self, variables: typing.Iterable[str], **kwargs):
        """
        Create new AccumulateVariables hook.

        :param variables: collection of variable names to be logged
        """
        super().__init__(**kwargs)
        self._variables = variables
        self._accumulator = None
        self._reset_accumulator()

    def _reset_accumulator(self):
        """Set the accumulator to an empty double-index :py:class:`collections.defaultdict`."""
        self._accumulator = defaultdict(lambda: defaultdict(list))

    def after_batch(self, stream_name: str, batch_data: Batch):
        """
        Extend the accumulated variables with the given batch data.

        :param stream_name: stream name; e.g. ``train`` or any other...
        :param batch_data: batch data = stream sources + model outputs
        :raise KeyError: if the variables to be aggregated are missing
        :raise TypeError: if the variable value is not iterable (e.g. it is only a scalar)
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
        """Reset the accumulator after each epoch."""
        self._reset_accumulator()
