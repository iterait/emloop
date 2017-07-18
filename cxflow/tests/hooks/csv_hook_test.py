"""
Test module for csv hook (cxflow.hooks.csv_hook).
"""

import numpy as np
import unittest
import collections
import os
import tempfile

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.csv_hook import CSVHook


_EXAMPLES = 5


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
        "Test correct hook initialization."

        output_file = tempfile.NamedTemporaryFile().name
        variables = ['accuracy', 'precision', 'loss']

        hook = CSVHook(output_dir="", output_file=output_file, variables=variables)

        self.assertEqual(hook._variables, variables)
        self.assertTrue(os.path.isfile(output_file))

        on_unknown_type = 'raise'
        with self.assertRaises(AssertionError):
            hook = CSVHook(output_dir="", output_file=output_file, variables=variables,
                           on_unknown_type=on_unknown_type)

        on_missing_variable = 'raise'
        with self.assertRaises(AssertionError):
            hook = CSVHook(output_dir="", output_file=output_file, variables=variables,
                           on_missing_variable=on_missing_variable)



    def test_write_header(self):
        "Test writing a correct header to csv file."

        variables = ['accuracy', 'precision', 'loss']
        output_file = tempfile.NamedTemporaryFile().name
        delimiter = ";"
        hook = CSVHook(output_dir="", output_file=output_file,
                       variables=variables, delimiter=delimiter)

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



if __name__ == '__main__':
        unittest.main()
