"""
Test module for saving class probabilities to a csv file hook (cxflow.hooks.logits_to_csv_hook).
"""

import numpy as np
import pandas as pd
import pytest
import os

from cxflow.hooks.logits_to_csv import LogitsToCsv


_ITERS = 2
_CLASSES = ['red', 'green', 'blue']
_STREAM_NAME = 'train'


def get_batch():
    batch = {'color': np.arange(15).reshape(5, 3),
             'pic_id': np.arange(5).reshape(5),
             'incorrect_0': np.arange(20).reshape(5, 4),
             'incorrect_1': np.arange(18).reshape(6, 3)}
    return batch


def test_saving_logits_to_csv(tmpdir):
    """Test logits are correctly saved to file."""

    selected_var = 'color'
    id_var = 'pic_id'
    filename = os.path.join(tmpdir, 'colors.csv')
    logits_to_csv = LogitsToCsv(variable=selected_var, class_names=_CLASSES, id_variable=id_var, output_file=filename)

    for i in range(_ITERS):
        batch = get_batch()
        logits_to_csv.after_batch(_STREAM_NAME, batch)

    logits_to_csv.after_epoch(0)

    df = pd.DataFrame.from_records(logits_to_csv._accumulator)
    expected_columns = [id_var] + _CLASSES
    expected_ids = list(range(5))
    expected_red = list(range(0, 15, 3))
    assert list(df) == expected_columns
    assert df[id_var].values.tolist() == expected_ids + expected_ids
    assert df['red'].values.tolist() == expected_red + expected_red
    assert os.path.exists(filename)


_VARS = ['not-there', 'color', 'incorrect_0', 'incorrect_1']
_ID_VARS = ['pic_id', 'not-there', 'pic_id', 'pic_id']
_ERRORS = [KeyError, KeyError, AssertionError, AssertionError]


@pytest.mark.parametrize('var, id_var, error', zip(_VARS, _ID_VARS, _ERRORS))
def test_saving_logits_to_csv_raises_error(tmpdir, var, id_var, error):
    """Test raising an assertion error if variable is not present in a batch/variable lengths are not same."""

    filename = os.path.join(tmpdir, 'colors.csv')
    logits_to_csv = LogitsToCsv(variable=var, class_names=_CLASSES, id_variable=id_var, output_file=filename)

    batch = get_batch()

    with pytest.raises(error):
        logits_to_csv.after_batch(_STREAM_NAME, batch)


def test_saving_logits_to_csv_stream_not_in_specified(tmpdir):
    """Test logits are not saved to file when stream is not in available streams."""

    selected_var = 'color'
    id_var = 'pic_id'
    filename = os.path.join(tmpdir, 'colors.csv')
    logits_to_csv = LogitsToCsv(variable=selected_var, class_names=_CLASSES, id_variable=id_var, output_file=filename,
                                streams=['test'])

    for i in range(_ITERS):
        batch = get_batch()
        logits_to_csv.after_batch(_STREAM_NAME, batch)

    logits_to_csv.after_epoch(0)

    assert logits_to_csv._accumulator == []
    assert not os.path.exists(filename)
