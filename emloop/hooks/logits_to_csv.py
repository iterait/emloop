"""
Hook for saving class probabilities to a csv file.
"""

import logging
import pandas as pd
from typing import Iterable, Optional
from collections import OrderedDict

from . import AbstractHook
from ..types import Batch


class LogitsToCsv(AbstractHook):
    """
    Save a stream of logits to a csv file.

    In the generated file, there are |class_names| + 1 columns for each
    example.  The one extra column is for the id of the example.  The class
    names are used as headers for the corresponding columns and the id column
    is named by the corresponding stream source.

    .. code-block:: yaml
        :caption: Save a csv with columns `red`, `green`, and `blue` to `/tmp/colors.csv`.
                  The stream variable `color` is expected to be a sequence of three numbers.

        hooks:
          - LogitsToCsv:
              variable: color
              class_names: [red, green, blue]
              id_variable: picture_id
              output_file: /tmp/colors.csv
    """

    def __init__(self, variable: str, class_names: Iterable[str], id_variable: str,
                 output_file: str, streams: Optional[Iterable[str]]=None, **kwargs):
        """
        :param variable: name of the source with a sequence for each example
        :param class_names: the names of the individual classes; should correspond
                            to the size of the `variable` source
        :param id_variable: name of the source which represents a unique example id
        :param output_file: the desired name of the output csv file
        :param streams: names of the streams to be considered; leave None to consider all streams
        """
        assert len(class_names) > 0, 'You have to specify at least one class name.'

        super().__init__(**kwargs)

        self._variable = variable
        self._class_names = class_names
        self._id_variable = id_variable
        self._output_file = output_file
        self._streams = streams
        self._accumulator = []

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        """Accumulate the given logits."""
        if self._streams is not None and stream_name not in self._streams:
            return

        # Assert variables in batch data.
        if self._id_variable not in batch_data:
            raise KeyError('Variable `{}` to be used as unique id was not found in the batch data for stream `{}`. '
                           'Available variables are `{}`.'.format(self._id_variable, stream_name, batch_data.keys()))
        if self._variable not in batch_data:
            raise KeyError('Variable `{}` to be saved to csv was not found in the batch data for stream `{}`. '
                           'Available variables are `{}`.'.format(self._variable, stream_name, batch_data.keys()))

        # Assert equal batch sizes.
        assert len(batch_data[self._id_variable]) == len(batch_data[self._variable]), 'Batch sizes of variable ' \
            'to be saved `{}` and variable_id `{}` are not equal.'.format(self._variable, self._id_variable)

        # Iterate through examples.
        for example_idx, example_id in enumerate(batch_data[self._id_variable]):
            assert len(batch_data[self._variable][example_idx]) == len(self._class_names), 'Size of variable to save ' \
                '`{}` does not correspond to number of class names `{}`.'.format(self._variable, self._class_names)
            # Build the csv record.
            record = OrderedDict()
            record[self._id_variable] = example_id
            for class_idx, class_name in enumerate(self._class_names):
                record[class_name] = batch_data[self._variable][example_idx][class_idx]
            self._accumulator.append(record)

    def after_epoch(self, epoch_id: int, **_) -> None:
        """Save all the accumulated data to csv."""
        if len(self._accumulator) == 0:
            return
        logging.info('Saving logits from %s to %s.', self._variable, self._output_file)
        pd.DataFrame.from_records(self._accumulator).to_csv(self._output_file, index=False)
