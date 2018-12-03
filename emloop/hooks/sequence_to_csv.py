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
                  `area` and `color` to `/tmp/areas.csv`.

        hooks:
          - SequenceToCsv:
              variables: [area, color]
              id_variable: video_id
              output_file: /tmp/areas.csv
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
        assert len(variables) > 0, 'You have to specify at least one variable.'

        super().__init__(**kwargs)

        self._variables = variables
        self._id_variable = id_variable
        self._output_file = output_file
        self._pad_mask_variable = pad_mask_variable
        self._streams = streams
        self._accumulator = []

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        """Accumulate the given sequences."""
        if self._streams is not None and stream_name not in self._streams:
            return

        # Assert variables in batch data.
        if self._id_variable not in batch_data:
            raise KeyError('Variable `{}` to be used as unique id was not found in the batch data for stream `{}`. '
                           'Available variables are `{}`.'.format(self._id_variable, stream_name, batch_data.keys()))
        if self._pad_mask_variable is not None and self._pad_mask_variable not in batch_data:
            raise KeyError('Variable `{}` to be used as padding mask was not found in the batch data for stream `{}`. '
                           'Available variables are `{}`.'.format(self._pad_mask_variable, stream_name,
                                                                  batch_data.keys()))
        for variable in self._variables:
            if variable not in batch_data:
                raise KeyError('Variable `{}` to be saved to csv was not found in the batch data for stream `{}`. '
                               'Available variables are `{}`.'.format(variable, stream_name, batch_data.keys()))

        # Assert equal batch sizes.
        batch_size = -1
        for variable in self._variables:
            assert batch_size == -1 or batch_size == len(batch_data[variable]), 'Batch sizes of variables to save ' \
                'are not equal.'
            batch_size = len(batch_data[variable])
        # Assert equal sequence sizes for each example.
        seq_lens = [-1] * batch_size
        for variable in self._variables:
            for example_idx, example_seq in enumerate(batch_data[variable]):
                assert seq_lens[example_idx] == -1 or seq_lens[example_idx] == len(example_seq), 'Sequence sizes ' \
                    'of variables to save are not equal.'
                seq_lens[example_idx] = len(example_seq)
        if self._pad_mask_variable is not None:
            assert batch_size == len(batch_data[self._pad_mask_variable]), 'Batch sizes of variables to save ' \
                '`{}` and padding mask variable `{}` are not equal.'.format(self._variables, self._pad_mask_variable)
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
        """Save all the accumulated data to csv."""
        if len(self._accumulator) == 0:
            return
        logging.info('Saving ' + ', '.join(self._variables) + ' to %s.', self._output_file)
        pd.DataFrame.from_records(self._accumulator).to_csv(self._output_file, index=False)
