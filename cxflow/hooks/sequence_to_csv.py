"""
Hook for saving a stream of sequences to a csv file.
"""

import logging
import pandas as pd
from typing import Iterable, Optional
from collections import OrderedDict

from . import AbstractHook
from ..types import Batch


class SequenceToCsv(AbstractHook):
    """
    Save a stream of sequences to a csv file.

    In this file, there the following columns: <id_source>, `index` and
    <source_name...>, where <id_source> is the name of the id column and
    <source_name...> are the names of the stream columns to be dumped.

    .. code-block:: yaml
        :caption: Save a csv with columns `video_id`, `index`,
                  `area` and `color` to `/var/areas.csv`.

        hooks:
          - SequenceToCsv:
              variables: [area, color]
              id_variable: video_id
              output_file: /var/areas.csv
    """

    def __init__(self, variables: Iterable[str], id_variable: str, output_file: str,
                 pad_mask_variable: Optional[str]=None,
                 streams: Optional[Iterable[str]]=None, **kwargs):
        """
        :param variables: names of the sources with an equally long sequence for each example
        :param id_variable: name of the source which represents a unique example id
        :param output_file: the desired name of the output csv file
        :param pad_mask_variable: name of the source which represents the padding mask
        :param streams: names of the streams to be considered; leave None to consider all streams
        """
        super().__init__(**kwargs)

        self._variables = variables
        self._id_variable = id_variable
        self._output_file = output_file
        self._pad_mask_variable = pad_mask_variable
        self._streams = streams
        self._accumulator = []

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        """
        Accumulate the given sequences.
        """
        if self._streams is not None and stream_name not in self._streams:
            return

        assert len(self._variables) > 0
        # Assert variables in batch data.
        assert self._id_variable in batch_data
        assert self._pad_mask_variable is None or self._pad_mask_variable in batch_data
        for variable in self._variables:
            assert variable in batch_data
        # Assert equal batch sizes.
        batch_size = -1
        for variable in self._variables:
            assert batch_size == -1 or batch_size == len(batch_data[variable])
            batch_size = len(batch_data[variable])
        # Assert equal sequence sizes for each example.
        seq_lens = [-1] * batch_size
        for variable in self._variables:
            for example_idx, example_seq in enumerate(batch_data[variable]):
                assert seq_lens[example_idx] == -1 or seq_lens[example_idx] == len(example_seq)
                seq_lens[example_idx] = len(example_seq)
        if self._pad_mask_variable is not None:
            assert batch_size == len(batch_data[self._pad_mask_variable])
            for var, mask in zip(batch_data[self._variables[0]], batch_data[self._pad_mask_variable]):
                assert len(var) == len(mask)

        # Iterate through examples.
        for example_idx, example_id in enumerate(batch_data[self._id_variable]):
            # Get padding mask if available.
            mask = None
            if self._pad_mask_variable is not None:
                mask = batch_data[self._pad_mask_variable][example_idx]
            # Iterate through the sequences.
            for seq_idx in range(seq_lens[example_idx]):
                # Only consider non-masked items.
                if mask is not None and mask[seq_idx] == False:
                    continue
                # Build the csv record.
                record = OrderedDict()
                record[self._id_variable] = example_id
                record['index'] = seq_idx
                for variable in self._variables:
                    record[variable] = batch_data[variable][example_idx][seq_idx]
                self._accumulator.append(record)

    def after_epoch(self, epoch_id: int, **_) -> None:
        """
        Save all the accumulated data to csv.
        """
        if self._accumulator != []:
            logging.info('Saving ' + ', '.join(self._variables) + ' to %s.', self._output_file)
            pd.DataFrame.from_records(self._accumulator).to_csv(self._output_file, index=False)
