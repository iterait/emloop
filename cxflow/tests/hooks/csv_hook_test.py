"""
Test module for csv hook (cxflow.hooks.csv_hook).
"""
import logging
import tempfile
import shutil
from os import path
from unittest import TestCase

from cxflow.hooks.csv_hook import CSVHook


class CSVHookTest(TestCase):
    """Test case for csv hook."""

    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_csv_log(self):
        """Test csv file being correctly dumped."""
        output_file = 'training.csv'
        temp_dir = tempfile.mkdtemp(prefix='csvhooktest', dir=tempfile.gettempdir())
        hook = CSVHook(net=None, output_file=output_file, output_dir=temp_dir, metrics_to_display=['accuracy', 'loss'],
                       config=None, dataset=None)

        hook.before_first_epoch(valid_results={'accuracy': 0.0, 'loss': 1.0},
                                test_results={'accuracy': 0.1, 'loss': 0.9})
        hook.after_epoch(epoch_id=1,
                         train_results={'accuracy': 0.2, 'loss': 0.7},
                         valid_results={'accuracy': 0.3, 'loss': 0.6},
                         test_results={'accuracy': 0.4, 'loss': 0.5})

        hook.after_epoch(epoch_id=2,
                         train_results={'accuracy': 0.5, 'loss': 0.4},
                         valid_results={'accuracy': 0.6, 'loss': 0.3},
                         test_results={'accuracy': 0.7, 'loss': 0.2})

        expected = """"epoch_id","train_accuracy","valid_accuracy","test_accuracy","train_loss","valid_loss","test_loss"
0,,0.0,0.1,,1.0,0.9
1,0.2,0.3,0.4,0.7,0.6,0.5
2,0.5,0.6,0.7,0.4,0.3,0.2
"""

        with open(path.join(temp_dir, output_file), 'r') as file:
            result = file.read()

        self.assertEqual(expected, result)

        shutil.rmtree(temp_dir)
