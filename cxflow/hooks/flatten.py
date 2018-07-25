"""
Hook for flattening variables.
"""
from typing import Iterable, Mapping, Optional

import numpy as np

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
        super().__init__(**kwargs)
        self._variables = variables
        self._streams = streams

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        if self._streams is None or stream_name in self._streams:
            for variable in self._variables:
                assert variable in batch_data
            for src, dst in self._variables.items():
                batch_data[dst] = np.array(batch_data[src]).flatten()
