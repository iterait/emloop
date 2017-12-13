"""
Module with stop on NaN hook test case (see :py:class:`cxflow.hooks.StopOnNan`).
"""
import numpy as np
from testfixtures import LogCapture

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks import StopOnNaN, TrainingTerminated


class StopOnNaNTest(CXTestCase):
    """Test StopOnNaN hook."""

    @staticmethod
    def _get_data(val):
        return {
            'train': {'var': val, 'zero': 0},
            'valid': {'var': 0, 'zero': 0}
        }

    def test_variables(self):
        """Test stopping of NaNs of all, individual and missing variables. Check handling of infinities."""

        for nan in [np.nan, float('nan'), np.inf, float('inf')]:
            for val in [nan, [1, 2, nan], {1: {3: nan}, 2: 0}]:
                with self.assertRaises(TrainingTerminated):
                    StopOnNaN(variables=['var'], stop_on_inf=True).after_epoch(epoch_data=StopOnNaNTest._get_data(val))
                if not np.isnan(nan):
                    StopOnNaN(variables=['var']).after_epoch(epoch_data=StopOnNaNTest._get_data(val))
                    StopOnNaN().after_epoch(epoch_data=StopOnNaNTest._get_data(val))
                StopOnNaN(variables=['zero']).after_epoch(epoch_data=StopOnNaNTest._get_data(val))

        with self.assertRaises(KeyError):
            StopOnNaN(variables=['missing']).after_epoch(epoch_data=StopOnNaNTest._get_data(8))

    def test_unsupported(self):
        """Test handling of unsupported types."""

        with self.assertRaises(ValueError):
            StopOnNaN(on_unknown_type='error').after_epoch(epoch_data=StopOnNaNTest._get_data(lambda: 0))

        with self.assertRaises(AssertionError):
            StopOnNaN(on_unknown_type='bad value')

        with LogCapture() as log_capture:
            StopOnNaN(on_unknown_type='warn').after_epoch(epoch_data=StopOnNaNTest._get_data(lambda: 0))

        log_capture.check(
            ('root', 'WARNING', 'Variable `var` of type `<class \'function\'>` can not be checked for NaNs.'),
        )

        StopOnNaN().after_epoch(epoch_data=StopOnNaNTest._get_data(lambda: 0))

    def test_periodicity(self):
        """Test if checking after batch and after epoch according to configuration"""

        StopOnNaN(after_batch=True, after_epoch=False).after_epoch(epoch_data=StopOnNaNTest._get_data(np.nan))
        with self.assertRaises(TrainingTerminated):
            StopOnNaN(after_batch=True, after_epoch=False).after_batch(stream_name='train', batch_data=StopOnNaNTest._get_data(np.nan)['train'])
        with self.assertRaises(TrainingTerminated):
            StopOnNaN().after_epoch(epoch_data=StopOnNaNTest._get_data(np.nan))
        StopOnNaN().after_batch(stream_name='train', batch_data=StopOnNaNTest._get_data(np.nan)['train'])
