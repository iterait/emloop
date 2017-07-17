"""
Test module for csv hook (cxflow.hooks.csv_hook).
"""

import numpy as np

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.csv_hook import CSVHook


_EXAMPLES = 5


class CSVHookTest(CXTestCase):
    """Test case for CSVHook."""

    def get_epoch_data(self):

        epoch_data = {
            'train': {
                'accuracy': 1,
                'precision': np.ones(_EXAMPLES)
                'loss': {'mean': 1},
                'omitted': 0
            },
            'test': {
                'accuracy': 2,
                'precision': 2 * np.ones(_EXAMPLES)
                'loss': {'mean': 2}
                'omitted': 0
            },
            'valid': {
                'accuracy': 3,
                'precision': 3 * np.ones(_EXAMPLES)
                'loss': {'mean': 3}
                'omitted': 0
            },
        }

        return epoch_data

    def test_init_hook(self):
        output_dir = ""
        output_file = "training.csv"
        delimiter = ","
        default_value = ""
        variables = ['accuracy', 'precision', 'loss']
        on_unknown_type: str='default'
        on_missing_variable: str='error'

        hook = CSVHook(output_dir=output_dir,
                       output_file=output_file,
                       delimiter=delimiter,
                       default_value=default_value,
                       variables=variables,
                       on_unknown_type=on_unknown_type,
                       on_missing_variable=on_missing_variable)
