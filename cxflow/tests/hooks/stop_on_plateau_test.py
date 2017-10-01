"""
Test module for StopOnPlateau hook (cxflow.hooks.stop_on_plateau).
"""

import numpy as np

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.stop_on_plateau import StopOnPlateau
from cxflow.hooks.abstract_hook import TrainingTerminated

_EXAMPLES = 10
_STREAM = 'valid'
_STREAM_UNUSED = 'train'


def get_batch_data(const=1):
    """
    Return batch of data.

    :param const: constant used for multiplying ``loss`` values
    """
    return {'loss': const * np.ones(_EXAMPLES)}


class StopOnPlateauTest(CXTestCase):
    """Test case for :py:class:`cxflow.hooks.StopOnPlateau` hook."""

    def test_raise_short_term_smaller(self):
        """Test raise ``AssertionError`` if ``long_term`` is smaller than ``short_term``."""

        with self.assertRaises(AssertionError):
            StopOnPlateau(long_term=100, short_term=500)

    def test_stop_on_plateau(self):
        """Test raise ``TrainingTerminated`` if the ``loss`` stops improving."""

        hook = StopOnPlateau(short_term=15, long_term=25)
        hook.after_batch(_STREAM, get_batch_data(10))
        hook.after_batch(_STREAM, get_batch_data(5))
        hook.after_batch(_STREAM, get_batch_data(5))

        with self.assertRaises(TrainingTerminated):
            hook.after_batch(_STREAM, get_batch_data(20))

    def test_accumulator_reset(self):
        """Test whether the accumulator is reseted correctly."""

        long_term = 15
        hook = StopOnPlateau(short_term=5, long_term=long_term)
        hook.after_batch(_STREAM, get_batch_data())
        hook.after_batch(_STREAM, get_batch_data())
        hook.after_batch(_STREAM_UNUSED, get_batch_data())
        hook.after_batch(_STREAM_UNUSED, get_batch_data())

        self.assertTrue(np.array_equal(hook._accumulator[_STREAM]['loss'], np.ones(2 * _EXAMPLES)))
        self.assertTrue(np.array_equal(hook._accumulator[_STREAM_UNUSED]['loss'], np.ones(2 * _EXAMPLES)))

        hook.after_epoch(None, None)

        self.assertTrue(np.array_equal(hook._accumulator[_STREAM]['loss'], np.ones(long_term)))
        self.assertTrue(np.array_equal(hook._accumulator[_STREAM_UNUSED]['loss'], np.array([])))
