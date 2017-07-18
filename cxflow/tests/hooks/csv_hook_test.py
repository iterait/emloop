"""
Test module for csv hook (cxflow.hooks.csv_hook).
"""

import numpy as np
import collections
import os
import tempfile

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.csv_hook import CSVHook


_EXAMPLES = 5
_VARIABLES = ['accuracy', 'precision', 'loss']


class CSVHookTest(CXTestCase):
    """Test case for CSVHook."""

    def get_epoch_data(self):
        epoch_data = collections.OrderedDict([
            ('train', collections.OrderedDict([
                ('accuracy', 1),
                ('precision', np.ones(_EXAMPLES)),
                ('loss', collections.OrderedDict([('mean', 1)])),
                ('omitted', 0)])
            ),

            ('test', collections.OrderedDict([
                ('accuracy', 2),
                ('precision', 2 * np.ones(_EXAMPLES)),
                ('loss', collections.OrderedDict([('mean', 2)])),
                ('omitted', 0)])
            ),

            ('valid', collections.OrderedDict([
                ('accuracy', 3),
                ('precision', 3 * np.ones(_EXAMPLES)),
                ('loss', collections.OrderedDict([('mean', 3)])),
                ('omitted', 0)])
            )
        ])

        return epoch_data

    def test_init_hook(self):
        """Test correct hook initialization."""

        output_file = tempfile.NamedTemporaryFile().name

        hook = CSVHook(output_dir="", output_file=output_file, variables=_VARIABLES)

        self.assertEqual(hook._variables, _VARIABLES)
        self.assertTrue(os.path.isfile(output_file))

        with self.assertRaises(AssertionError):
            hook = CSVHook(output_dir="", output_file=output_file,
                           on_unknown_type='raise')

        with self.assertRaises(AssertionError):
            hook = CSVHook(output_dir="", output_file=output_file,
                           on_missing_variable='raise')



    def test_write_header(self):
        """Test writing a correct header to csv file."""

        output_file = tempfile.NamedTemporaryFile().name
        delimiter = ";"
        hook = CSVHook(output_dir="", output_file=output_file,
                       variables=_VARIABLES, delimiter=delimiter)
        epoch_data = self.get_epoch_data()
        hook._write_header(epoch_data)

        with open(output_file, 'r') as f:
            header = f.read()

        # header must ends with a newline symbol
        self.assertEqual(header[-1], "\n")
        header = header[:-1]

        tested_header_columns = header.split(delimiter)

        # epoch_id column must be first
        self.assertEqual(tested_header_columns[0], '"epoch_id"')
        tested_header_columns = tested_header_columns[1:]

        valid_header_columns = ['train_accuracy', 'train_precision', 'train_loss',
                                'test_accuracy', 'test_precision', 'test_loss',
                                'valid_accuracy', 'valid_precision', 'valid_loss']

        self.assertEqual(valid_header_columns,
                         tested_header_columns)


    def test_write_row(self):
        """Test writing one row to csv file."""

        output_file = tempfile.NamedTemporaryFile().name
        delimiter = ";"
        default_value = '?'
        hook = CSVHook(output_dir="", output_file=output_file,
                       variables=_VARIABLES, delimiter=delimiter,
                       default_value=default_value)
        epoch_data = self.get_epoch_data()
        hook._write_header(epoch_data)

        epoch_id = 6
        hook._write_row(epoch_id, epoch_data)

        with open(output_file, 'r') as f:
            content = f.readlines()
        row = content[1]

        # each line must ends with newline symbol
        self.assertEqual(row[-1], "\n")
        row = row[:-1]

        valid_row = ['6', '1', '?', '1', '2', '?', '2', '3', '?', '3']
        self.assertEqual(valid_row, row.split(delimiter))

    def test_raise_missing_variable(self):
        """
        Test raising error when selected variable is missing and
        on_missing_variable option is set to 'error'.
        """

        variables = _VARIABLES + ['missing']
        output_file = tempfile.NamedTemporaryFile().name
        hook = CSVHook(output_dir="", output_file=output_file, variables=variables, on_missing_variable='error')
        epoch_data = self.get_epoch_data()
        hook._write_header(epoch_data)
        epoch_id = 6
        with self.assertRaises(KeyError):
            hook._write_row(epoch_id, epoch_data)


    def test_raise_unknown_type(self):
        """
        Test raising error when on_unknown_type option is set to 'error' and
        value of selected variable is not scalar.
        """

        output_file = tempfile.NamedTemporaryFile().name
        hook = CSVHook(output_dir="", output_file=output_file, on_unknown_type='error')
        epoch_data = self.get_epoch_data()
        hook._write_header(epoch_data)
        epoch_id = 6
        with self.assertRaises(ValueError):
            hook._write_row(epoch_id, epoch_data)

    def test_after_epoch_one_header(self):
        """
        Tests result of after_epoch method and whether it writes
        only one header at the beginning of csv file.
        """

        output_file = tempfile.NamedTemporaryFile().name
        hook = CSVHook(output_dir="", output_file=output_file)
        epoch_data = self.get_epoch_data()

        hook.after_epoch(6, epoch_data)
        hook.after_epoch(7, epoch_data)

        with open(output_file) as f:
            content = f.readlines()

        header = content[0]
        row1 = content[1]
        row2 = content[2]

        self.assertEqual(len(content), 3)
        self.assertNotEqual(header, row1)
        self.assertNotEqual(header, row2)
