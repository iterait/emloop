"""
Test module for :py:class:`cxflow.hooks.Check`.
"""

import numpy as np
import collections

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.check import Check
from cxflow.hooks.abstract_hook import TrainingTerminated


_VAR = "accuracy"
_MIN_ACCURACY = 0.95
_MAX_EPOCH = 10
_CURRENT_EPOCH = 5


def _get_epoch_data():
    epoch_data = collections.OrderedDict([
        ('train', collections.OrderedDict([
            ('accuracy', 1),
        ])),
        ('test', collections.OrderedDict([
            ('accuracy', 0.5),
        ])),
        ('valid', collections.OrderedDict([
            ('accuracy', np.ones(10)),
        ]))
    ])
    return epoch_data


class CheckTest(CXTestCase):
    """Test case for :py:class:`cxflow.hooks.Check`."""

    def setUp(self):
        """Create new epoch data for each test separately."""
        self._epoch_data = _get_epoch_data()

    def test_stream_raise(self):
        """Test raising error, when stream not in epoch_data."""
        hook = Check(_VAR, _MIN_ACCURACY, _MAX_EPOCH, "unknown")
        with self.assertRaises(KeyError):
            hook.after_epoch(_CURRENT_EPOCH, self._epoch_data)

    def test_variable_raise(self):
        """Test raising error, when variable not in stream."""
        hook = Check("not_present", _MIN_ACCURACY, _MAX_EPOCH)
        with self.assertRaises(KeyError):
            hook.after_epoch(_CURRENT_EPOCH, self._epoch_data)

    def test_not_scalar_raise(self):
        """Test raising error, when variable is not scalar."""
        hook = Check(_VAR, _MIN_ACCURACY, _MAX_EPOCH)
        with self.assertRaises(TypeError):
            hook.after_epoch(_CURRENT_EPOCH, self._epoch_data)

    def test_training_terminated(self):
        """
        Test whether training terminates if the given
        stream variable exceeds the threshold.
        """
        hook = Check(_VAR, _MIN_ACCURACY, _MAX_EPOCH, "train")
        with self.assertRaises(TrainingTerminated):
            hook.after_epoch(_CURRENT_EPOCH, self._epoch_data)

    def test_epoch_raise(self):
        """
        Test raising error, when required value of
        stream variable wasn't reached in specified count of epochs.
        """
        hook = Check(_VAR, _MIN_ACCURACY, _MAX_EPOCH, "test")
        with self.assertRaises(ValueError):
            hook.after_epoch(11, self._epoch_data)
