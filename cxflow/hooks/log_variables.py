"""
Module with standard logging hook.
"""
import logging
from typing import Iterable

import numpy as np

from . import AbstractHook
from ..types import EpochData


class LogVariables(AbstractHook):
    """
    Log the training results to stderr via standard :py:mod:`logging` module.


    .. code-block:: yaml
        :caption: log all the variables

        hooks:
          - LogVariables

    .. code-block:: yaml
        :caption: log only certain variables

        hooks:
          - LogVariables:
              variables: [loss]

    .. code-block:: yaml
        :caption: warn about unsupported variables

        hooks:
          - LogVariables:
              on_unknown_type: warn

    """

    UNKNOWN_TYPE_ACTIONS = ['error', 'warn', 'str', 'ignore']
    """Posible actions to take on unknown variable type."""

    def __init__(self, variables: Iterable[str]=None, on_unknown_type='ignore', **kwargs):
        """
        Create new LogVariables hook.

        :param variables: variable names to be logged; log all the variables by default
        :param on_unknown_type: an action to be taken if the variable type is not supported (e.g. a list)
        """
        assert on_unknown_type in LogVariables.UNKNOWN_TYPE_ACTIONS

        self._variables = variables
        self._on_unknown_type = on_unknown_type
        super().__init__(**kwargs)

    def _log_variables(self, epoch_data: EpochData):
        """
        Log variables from the epoch data.

        .. warning::
            At the moment, only scalars and dicts of scalars are properly formatted and logged.
            Other value types are ignored by default.

        One may set ``on_unknown_type`` to ``str`` in order to log all the variables anyways.

        :param epoch_data: epoch data to be logged
        :raise KeyError: if the specified variable is not found in the stream
        :raise TypeError: if the variable value is of unsupported type and ``self._on_unknown_type`` is set to ``error``
        """
        for stream_name in epoch_data.keys():
            stream_data = epoch_data[stream_name]
            variables = self._variables if self._variables is not None else stream_data.keys()
            for variable in variables:
                if variable not in stream_data:
                    raise KeyError('Variable `{}` to be logged was not found in the batch data for stream `{}`. '
                                   'Available variables are `{}`.'.format(variable, stream_name, stream_data.keys()))
                value = stream_data[variable]
                if np.isscalar(value):
                    logging.info('\t%s %s: %f', stream_name, variable, value)
                elif isinstance(value, dict):
                    keys = list(value.keys())
                    if len(keys) == 1:
                        logging.info('\t%s %s %s: %f', stream_name, variable, keys[0], value[keys[0]])
                    else:
                        logging.info('\t%s %s:', stream_name, variable)
                        for key, val in value.items():
                            logging.info('\t\t%s: %f', key, val)
                else:
                    if self._on_unknown_type == 'error':
                        raise TypeError('Variable type `{}` can not be logged. Variable name: `{}`.'
                                        .format(type(value).__name__, variable))
                    elif self._on_unknown_type == 'warn':
                        logging.warning('Variable type `%s` can not be logged. Variable name: `%s`.',
                                        type(value).__name__, variable)
                    elif self._on_unknown_type == 'str':
                        logging.info('\t%s %s: %s', stream_name, variable, value)

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        """
        Log the epoch data via :py:mod:`logging` API.
        Additionally, a blank line is printed directly to stderr to delimit the outputs from other epochs.

        :param epoch_id: number of processed epoch
        :param epoch_data: epoch data to be logged
        """
        self._log_variables(epoch_data)
