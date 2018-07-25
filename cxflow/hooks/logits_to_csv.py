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
        :caption: Save a csv with columns `red`, `green`, and `blue` to `/var/colors.csv`.
                  The stream variable `color` is expected to be a sequence of three numbers.

        hooks:
          - LogitsToCsv:
              variable: color
              class_names: ['red', 'green', 'blue']
              id_variable: picture_id
              output_file: /var/colors.csv
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
        super().__init__(**kwargs)

        self._variable = variable
        self._class_names = class_names
        self._id_variable = id_variable
        self._output_file = output_file
        self._streams = streams
        self._accumulator = []

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        """
        Accumulate the given logits.
        """
        if self._streams is not None and stream_name not in self._streams:
            return

        # Assert variables in batch data.
        assert self._id_variable in batch_data
        assert self._variable in batch_data

        # Assert equal batch sizes.
        assert len(batch_data[self._id_variable]) == len(batch_data[self._variable])

        # Iterate through examples.
        for example_idx, example_id in enumerate(batch_data[self._id_variable]):
            assert len(batch_data[self._variable][example_idx]) == len(self._class_names)
            # Build the csv record.
            record = OrderedDict()
            record[self._id_variable] = example_id
            for class_idx, class_name in enumerate(self._class_names):
                record[class_name] = batch_data[self._variable][example_idx][class_idx]
            self._accumulator.append(record)

    def after_epoch(self, epoch_id: int, **_) -> None:
        """
        Save all the accumulated data to csv.
        """
        if self._accumulator != []:
            logging.info('Saving logits from '.join(self._variable) + ' to %s.', self._output_file)
            pd.DataFrame.from_records(self._accumulator).to_csv(self._output_file, index=False)
