"""
Test module for stats hook (cxflow.hooks.stats_hook).
"""

import numpy as np
import collections

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.stats_hook import StatsHook


class StatsHookTest(CXTestCase):
    """Test case for StatsHook."""

    def get_batch(self, batch_id):
        batch = {'accuracy': batch_id * (np.ones(5) + 1),
                 'loss': batch_id * (np.ones(5) + 2)}
        return batch

    def test_raise_on_init(self):
        """Tests raising error if specified aggregation is not supported."""

        variables = {'loss': ['mean', 'median', 'max'],
                     'accuracy': ['mean', 'median', 'not_supported']}
        with self.assertRaises(ValueError):
            hook = StatsHook(variables)

    def test_compute_save_stats(self):
        """Tests correctness of computed aggregations and their saving."""

        variables = {'loss': ['mean', 'std', 'min', 'max', 'median'],
                     'accuracy': ['mean', 'std', 'min', 'max', 'median']}

        hook = StatsHook(variables)

        epoch_data = {'train': {'accuracy': None, 'loss': None},
                      'test': {'accuracy': None, 'loss': None}}

        for batch_id in range(1, 3):
            for stream in epoch_data.keys():
                hook.after_batch(stream, self.get_batch(batch_id))

        hook.after_epoch(epoch_data)

        valid_aggrs = {'train':
                       {'loss': {'mean': 4.5, 'std': 1.5, 'min': 3, 'max': 6, 'median': 4.5},
                        'accuracy': {'mean': 3, 'std': 1, 'min': 2, 'max': 4, 'median': 3}},
                       'test':
                       {'loss': {'mean': 4.5, 'std': 1.5, 'min': 3, 'max': 6, 'median': 4.5},
                        'accuracy': {'mean': 3, 'std': 1, 'min': 2, 'max': 4, 'median': 3}}}

        self.assertEqual(epoch_data.keys(),
                         valid_aggrs.keys())
        for stream in epoch_data.keys():
            self.assertEqual(epoch_data[stream].keys(),
                             valid_aggrs[stream].keys())
            for variable in epoch_data[stream]:
                self.assertEqual(epoch_data[stream][variable].keys(),
                                 valid_aggrs[stream][variable].keys())
                for aggr in epoch_data[stream][variable]:
                    self.assertEqual(epoch_data[stream][variable][aggr],
                                     valid_aggrs[stream][variable][aggr])
