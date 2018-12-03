"""
Test module for csv hook (:py:class:`emloop.hooks.WriteCSV).
"""

import numpy as np
import collections
import os
import tempfile
import pytest

from emloop.hooks.write_csv import WriteCSV


_EXAMPLES = 5
_VARIABLES = ['accuracy', 'precision', 'loss']


def _get_epoch_data():
    epoch_data = collections.OrderedDict([
        ('train', collections.OrderedDict([
            ('accuracy', 1),
            ('precision', np.ones(_EXAMPLES)),
            ('loss', collections.OrderedDict([('mean', 1)])),
            ('omitted', 0)])),
        ('test', collections.OrderedDict([
            ('accuracy', 2),
            ('precision', 2 * np.ones(_EXAMPLES)),
            ('loss', collections.OrderedDict([('nanmean', 2)])),
            ('omitted', 0)])),
        ('valid', collections.OrderedDict([
            ('accuracy', 3),
            ('precision', 3 * np.ones(_EXAMPLES)),
            ('loss', collections.OrderedDict([('mean', 3)])),
            ('omitted', 0),
            ('specific', 9)]))
    ])
    return epoch_data


@pytest.fixture
def tmpfile(tmpdir):
    return os.path.join(tmpdir, 'filename')


def test_init_hook(tmpfile):
    """Test correct hook initialization."""
    hook = WriteCSV(output_dir="", output_file=tmpfile,
                    variables=_VARIABLES)

    assert hook._variables == _VARIABLES

    with pytest.raises(AssertionError):
        WriteCSV(output_dir="", output_file=tmpfile,
                 on_unknown_type='raise')

    with pytest.raises(AssertionError):
        WriteCSV(output_dir="", output_file=tmpfile,
                 on_missing_variable='raise')


def test_write_header(tmpfile):
    """Test writing a correct header to csv file."""
    delimiter = ';'
    hook = WriteCSV(output_dir="", output_file=tmpfile,
                    variables=_VARIABLES, delimiter=delimiter)
    epoch_data = _get_epoch_data()
    hook._write_header(epoch_data)

    with open(tmpfile, 'r') as file:
        header = file.read()

    # header must ends with a newline symbol
    assert header[-1] == "\n"
    header = header[:-1]

    tested_header_columns = header.split(delimiter)

    # epoch_id column must be first
    assert tested_header_columns[0] == '"epoch_id"'
    tested_header_columns = tested_header_columns[1:]

    valid_header_columns = ['train_accuracy', 'train_precision',
                            'train_loss',
                            'test_accuracy', 'test_precision',
                            'test_loss',
                            'valid_accuracy', 'valid_precision',
                            'valid_loss']

    assert valid_header_columns == \
                     tested_header_columns


def test_write_row(tmpfile):
    """Test writing one row to csv file."""
    delimiter = ';'
    default_value = '?'
    hook = WriteCSV(output_dir="", output_file=tmpfile,
                    variables=_VARIABLES, delimiter=delimiter,
                    default_value=default_value)
    epoch_data = _get_epoch_data()
    hook._write_header(epoch_data)

    epoch_id = 6
    hook._write_row(epoch_id, epoch_data)

    with open(tmpfile, 'r') as file:
        content = file.readlines()
    row = content[1]

    # each line must ends with newline symbol
    assert row[-1] == "\n"
    row = row[:-1]

    valid_row = ['6', '1', '?', '1', '2', '?', '2', '3', '?', '3']
    assert valid_row == row.split(delimiter)


def test_raise_missing_variable(tmpfile):
    """
    Test raising error when selected variable is missing and
    on_missing_variable option is set to 'error'.
    """

    variables = _VARIABLES + ['missing']
    hook = WriteCSV(output_dir="", output_file=tmpfile,
                    variables=variables, on_missing_variable='error')
    epoch_data = _get_epoch_data()
    hook._write_header(epoch_data)
    epoch_id = 6
    with pytest.raises(KeyError):
        hook._write_row(epoch_id, epoch_data)


def test_raise_unknown_type(tmpfile):
    """
    Test raising error when on_unknown_type option is set to 'error' and
    value of selected variable is not scalar.
    """
    hook = WriteCSV(output_dir="", output_file=tmpfile,
                    on_unknown_type='error')
    epoch_data = _get_epoch_data()
    hook._write_header(epoch_data)
    epoch_id = 6
    with pytest.raises(TypeError):
        hook._write_row(epoch_id, epoch_data)


def test_after_epoch_one_header(tmpfile):
    """
    Tests result of after_epoch method and whether it writes
    only one header at the beginning of csv file.
    """
    delimiter = '|'
    hook = WriteCSV(output_dir="", output_file=tmpfile,
                    delimiter=delimiter, variables=_VARIABLES)
    epoch_data = _get_epoch_data()

    hook.after_epoch(6, epoch_data)
    hook.after_epoch(7, epoch_data)

    with open(tmpfile) as file:
        content = file.readlines()

    assert len(content) == 3
    header = content[0]

    for i, epoch_id in enumerate(['6', '7']):
        row = content[i + 1]
        assert header != row
        assert row[-1] == "\n"
        row = row[:-1]
        valid_row = [epoch_id, '1', '', '1', '2', '', '2', '3', '', '3']
        assert valid_row == row.split(delimiter)


def test_variable_deduction(tmpfile):
    """
    Test that the variable names are automatically deduced from the
    first available stream.
    """
    hook_train = WriteCSV(output_dir="", output_file=tmpfile)
    epoch_data = _get_epoch_data()
    hook_train._write_header(collections.OrderedDict([('train', epoch_data['train']),
                                                      ('valid', epoch_data['valid'])]))
    assert hook_train._variables == ['accuracy', 'precision', 'loss', 'omitted']

    hook_valid = WriteCSV(output_dir="", output_file=tmpfile)
    epoch_data = _get_epoch_data()
    hook_valid._write_header({'valid': epoch_data['valid']})
    assert hook_valid._variables == ['accuracy', 'precision', 'loss', 'omitted', 'specific']
