from cxflow.hooks.csv_hook import CSVHook

import pandas as pd

import logging
from os import path
from unittest import TestCase
import tempfile


class NetMocker:
    def __init__(self):
        self.log_dir = tempfile.mkdtemp(prefix='csvhooktest', dir=tempfile.gettempdir())


class CSVHookTest(TestCase):
    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_csv_log(self):
        net = NetMocker()
        output_file = 'training.csv'
        hook = CSVHook(net=net, output_file=output_file, metrics_to_display=['accuracy', 'loss'], config=None)

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

        with open(path.join(net.log_dir, output_file), 'r') as f:
            result = f.read()

        self.assertEqual(expected, result)
