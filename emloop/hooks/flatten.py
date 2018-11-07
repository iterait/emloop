"""
Hook for flattening variables.
"""
from typing import Iterable, Mapping, Optional
import more_itertools

from . import AbstractHook
from ..types import Batch


class Flatten(AbstractHook):
    """
    Flatten a stream variable.
    

    .. code-block:: yaml
        :caption: Example: Flatten `xs` variable in test stream and save the result into
                  variable `xs_flat` to be able to feed it into :py:class:`SaveConfusionMatrix` hook.

        hooks:
          - Flatten:
              variables: {xs: xs_flat}
              streams: [test]
          - SaveConfusionMatrix:
              variables: [xs_flat]
              streams: [test]
    """

    def __init__(self, variables: Mapping[str, str], streams: Optional[Iterable[str]]=None, **kwargs):
        """
        Hook constructor.

        :param variables: names of the variables to be flattened
        :param streams: list of stream names to be considered;
                        if None, the hook will be applied to all the available streams
        """
        assert len(variables) > 0, 'You have to specify at least one variable.'

        super().__init__(**kwargs)
        self._variables = variables
        self._streams = streams

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        """Flatten given variables."""
        if self._streams is not None and stream_name not in self._streams:
            return

        for variable in self._variables:
            if variable not in batch_data:
                raise KeyError('Variable `{}` to be flattened was not found in the batch data for stream `{}`. '
                               'Available variables are `{}`.'.format(variable, stream_name, batch_data.keys()))
            batch_data[self._variables[variable]] = list(more_itertools.collapse(batch_data[variable]))
