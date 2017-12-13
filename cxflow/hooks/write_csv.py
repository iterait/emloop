"""
Module with csv logging hook.
"""
import logging
import os
from os import path
from typing import Iterable, Optional

import numpy as np

from . import AbstractHook
from ..types import EpochData


class WriteCSV(AbstractHook):
    """
    Log ``epoch_data`` variables to a CSV file after each epoch.

    .. code-block:: yaml
        :caption: Log all the variables

        hooks:
          - WriteCSV

    .. code-block:: yaml
        :caption: Log only certain variables

        hooks:
          - WriteCSV:
              variables: [loss, fscore]

    .. code-block:: yaml
        :caption: Warn about unsupported variables

        hooks:
          - WriteCSV:
              variables: [loss, fscore, xxx]
              on_unknown_type: warn

    """

    UNKNOWN_TYPE_ACTIONS = ['error', 'warn', 'default']
    """Action executed on unknown type detection."""

    MISSING_VARIABLE_ACTIONS = ['error', 'warn', 'default']
    """Action executed on missing variable."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 output_dir: str, output_file: str="training.csv", delimiter: str=',',
                 default_value: str='', variables: Optional[Iterable[str]]=None, on_unknown_type: str='default',
                 on_missing_variable: str='error', **kwargs):
        """
        :param output_dir: directory to save the output CSV
        :param output_file: name of the output CSV file
        :param delimiter: CSV delimiter
        :param default_value: default value to supplement missing variables
        :param variables: subset of variable names to be written (all the variables are written by default)
        :param on_unknown_type: an action to be taken if the variable value type is not supported (e.g. a list)
        :param on_missing_variable: an action to be taken if the variable is specified but not provided
        """
        super().__init__(**kwargs)

        assert on_unknown_type in WriteCSV.UNKNOWN_TYPE_ACTIONS
        assert on_missing_variable in WriteCSV.MISSING_VARIABLE_ACTIONS

        self._variables = variables
        self._streams = None
        self._on_unknown_type = on_unknown_type
        self._on_missing_variable = on_missing_variable
        self._delimiter = delimiter
        self._default_value = default_value
        self._header_written = False

        self._file_path = path.join(output_dir, output_file)
        logging.debug('CSV output file "%s"', self._file_path)

    def _write_header(self, epoch_data: EpochData) -> None:
        """
        Write CSV header row with column names.

        Column names are inferred from the ``epoch_data`` and ``self.variables`` (if specified).
        Variables and streams expected later on are stored in ``self._variables`` and ``self._streams`` respectively.

        :param epoch_data: epoch data to be logged
        """
        self._variables = self._variables or list(epoch_data['train'].keys())
        self._streams = epoch_data.keys()

        header = ['"epoch_id"']
        for stream_name in self._streams:
            header += [stream_name + '_' + var for var in self._variables]
        with open(self._file_path, 'a') as file:
            file.write(self._delimiter.join(header) + '\n')
        self._header_written = True

    def _write_row(self, epoch_id: int, epoch_data: EpochData) -> None:
        """
        Write a single epoch result row to the CSV file.

        :param epoch_id: epoch number (will be written at the first column)
        :param epoch_data: epoch data
        :raise KeyError: if the variable is missing and ``self._on_missing_variable`` is set to ``error``
        :raise TypeError: if the variable has wrong type and  ``self._on_unknown_type`` is set to ``error``
        """

        # list of values to be written
        values = [epoch_id]

        for stream_name in self._streams:
            for variable_name in self._variables:
                column_name = stream_name+'_'+variable_name
                try:
                    value = epoch_data[stream_name][variable_name]
                except KeyError as ex:
                    err_message = '`{}` not found in epoch data.'.format(column_name)
                    if self._on_missing_variable == 'error':
                        raise KeyError(err_message) from ex
                    elif self._on_missing_variable == 'warn':
                        logging.warning(err_message)
                    values.append(self._default_value)
                    continue

                if isinstance(value, dict) and 'mean' in value:
                    value = value['mean']
                elif isinstance(value, dict) and 'nanmean' in value:
                    value = value['nanmean']

                if np.isscalar(value):
                    values.append(value)
                else:
                    err_message = 'Variable `{}` value is not scalar.'.format(variable_name)
                    if self._on_unknown_type == 'error':
                        raise TypeError(err_message)
                    elif self._on_unknown_type == 'warn':
                        logging.warning(err_message)
                    values.append(self._default_value)

        # write the row
        with open(self._file_path, 'a') as file:
            row = self._delimiter.join([str(value) for value in values])
            file.write(row + '\n')

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        """
        Write a new row to the CSV file with the given epoch data.

        In the case of first invocation, create the CSV header.

        :param epoch_id: number of the epoch
        :param epoch_data: epoch data to be logged
        """
        logging.debug('Saving epoch %d data to "%s"', epoch_id, self._file_path)
        if not self._header_written:
            self._write_header(epoch_data=epoch_data)
        self._write_row(epoch_id=epoch_id, epoch_data=epoch_data)
