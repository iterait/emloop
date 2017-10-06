"""
Test module for StopOnPlateau hook (cxflow.hooks.stop_on_plateau).
"""

import numpy as np

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.stop_on_plateau import StopOnPlateau
from cxflow.hooks.abstract_hook import TrainingTerminated


def get_epoch_data():
    """Return empty epoch data dict."""
    return {'train': {}, 'valid': {}, 'test': {}}


def get_batch_data(const=1):
    """
    Return batch data.

    :param const: constant used for multiplying ``loss`` values
    """
    return {'loss': const * np.ones(5),
            'accuracy': const * np.ones(5)}


def run_epoch(hook, const):
    """Call ``after_batch`` events."""
    for i in range(5):
        hook.after_batch('train', get_batch_data())
        hook.after_batch('valid', get_batch_data(const))
        hook.after_batch('test', get_batch_data(const))


class StopOnPlateauTest(CXTestCase):
    """Test case for :py:class:`cxflow.hooks.StopOnPlateau` hook."""

    def test_raise_short_term_smaller(self):
        """Test raise ``AssertionError`` if ``long_term`` is smaller than ``short_term``."""

        with self.assertRaises(AssertionError):
            StopOnPlateau(long_term=100, short_term=500)

    def test_stop_on_plateau(self):
        """Test raise ``TrainingTerminated`` on plateau."""
        # test minimizing test loss
        loss_hook = StopOnPlateau(short_term=3, long_term=6, stream='test')

        for value in [4, 3, 2, 2, 3]:
            run_epoch(loss_hook, value)
            loss_hook.after_epoch(0, get_epoch_data())
        run_epoch(loss_hook, 10)

        with self.assertRaises(TrainingTerminated):
            loss_hook.after_epoch(0, get_epoch_data())

        # test maximizing valid accuracy
        accuracy_hook = StopOnPlateau(short_term=3, long_term=6, variable='accuracy', objective='max')

        for value in [1, 5, 20, 2, 2]:
            run_epoch(accuracy_hook, value)
            accuracy_hook.after_epoch(0, get_epoch_data())
        run_epoch(accuracy_hook, 2)

        with self.assertRaises(TrainingTerminated):
            accuracy_hook.after_epoch(0, get_epoch_data())
