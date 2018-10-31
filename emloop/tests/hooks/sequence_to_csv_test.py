"""
Test module for saving a stream of sequences to a csv file hook (emloop.hooks.sequence_to_csv_hook).
"""

import numpy as np
import pandas as pd
import pytest
import os

from emloop.hooks.sequence_to_csv import SequenceToCsv


_ITERS = 2
_STREAM_NAME = 'train'


def get_batch():
    batch = {'color': np.arange(15).reshape(5, 3),
             'area': np.arange(30).reshape(5, 3, 2),
             'mask': ((0, 1, 1), (0, 1, 0), (0, 1, 1), (0, 1, 0), (0, 1, 1)),
             'pic_id': np.arange(5).reshape(5),
             'incorrect_0': np.arange(20).reshape(5, 4),
             'incorrect_1': np.arange(18).reshape(6, 3)}
    return batch


def test_saving_sequence_to_csv_without_optional_arguments(tmpdir):
    """Test sequences are correctly saved to file."""

    selected_vars = ['color', 'area']
    id_var = 'pic_id'
    filename = os.path.join(tmpdir, 'colors.csv')
    sequence_to_csv = SequenceToCsv(variables=selected_vars, id_variable=id_var, output_file=filename)

    for i in range(_ITERS):
        batch = get_batch()
        sequence_to_csv.after_batch(_STREAM_NAME, batch)

    sequence_to_csv.after_epoch(0)

    df = pd.DataFrame.from_records(sequence_to_csv._accumulator)
    expected_columns = [id_var] + ['index'] + selected_vars
    expected_color = list(range(15))
    assert list(df) == expected_columns
    assert df['color'].values.tolist() == expected_color + expected_color
    assert os.path.exists(filename)


def test_saving_sequence_to_csv_with_optional_arguments(tmpdir):
    """Test sequences are correctly saved to file with specified optional arguments."""

    selected_vars = ['color', 'area']
    id_var = 'pic_id'
    filename = os.path.join(tmpdir, 'colors.csv')
    sequence_to_csv = SequenceToCsv(variables=selected_vars, id_variable=id_var, output_file=filename,
                                    pad_mask_variable='mask', streams=['train', 'test'])

    for i in range(_ITERS):
        batch = get_batch()
        sequence_to_csv.after_batch(_STREAM_NAME, batch)

    sequence_to_csv.after_epoch(0)

    df = pd.DataFrame.from_records(sequence_to_csv._accumulator)
    expected_columns = [id_var] + ['index'] + selected_vars
    expected_color = [1, 2, 4, 7, 8, 10, 13, 14]
    assert list(df) == expected_columns
    assert df['color'].values.tolist() == expected_color + expected_color
    assert os.path.exists(filename)


_VARS = [['not-there'], ['color'], ['area'], ['area', 'incorrect_0'], ['area', 'incorrect_1'], [], ['color'], ['color']]
_ID_VAR = ['pic_id', 'not-there', 'pic_id', 'pic_id', 'pic_id', 'pic_id', 'pic_id', 'pic_id']
_MASK = ['mask', 'mask', 'not-there', 'mask', 'mask', 'mask', 'incorrect_0', 'incorrect_1']
_ERRORS = [KeyError, KeyError, KeyError, AssertionError, AssertionError, AssertionError, AssertionError, AssertionError]


@pytest.mark.parametrize('var, id_var, mask, error', zip(_VARS, _ID_VAR, _MASK, _ERRORS))
def test_saving_sequence_to_csv_raises_error(tmpdir, var, id_var, mask, error):
    """Test raising an assertion error if variable is not present in a batch/variable lengths are not same."""

    filename = os.path.join(tmpdir, 'colors.csv')
    sequence_to_csv = SequenceToCsv(variables=var, id_variable=id_var, output_file=filename, pad_mask_variable=mask)

    batch = get_batch()

    with pytest.raises(error):
        sequence_to_csv.after_batch(_STREAM_NAME, batch)


def test_saving_sequence_to_csv_stream_not_in_specified(tmpdir):
    """Test sequences are not saved to file when stream is not in available streams."""

    selected_vars = ['color', 'area']
    id_var = 'pic_id'
    filename = os.path.join(tmpdir, 'colors.csv')
    sequence_to_csv = SequenceToCsv(variables=selected_vars, id_variable=id_var, output_file=filename, streams=['test'])

    for i in range(_ITERS):
        batch = get_batch()
        sequence_to_csv.after_batch(_STREAM_NAME, batch)

    sequence_to_csv.after_epoch(0)

    assert len(sequence_to_csv._accumulator) == 0
    assert not os.path.exists(filename)
