"""
Test module for StopAfter hook (cxflow.hooks.stop_after_hook).
"""
import time

from cxflow.main_loop import MainLoop
from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.stop_after import StopAfter
from cxflow.hooks.abstract_hook import TrainingTerminated

NOTRAIN_STREAM_NAME = 'valid'
assert MainLoop.TRAIN_STREAM is not NOTRAIN_STREAM_NAME

class EpochStopperHookTest(CXTestCase):
    """Test case for StopAfter hook."""

    def test_no_conditions_raise(self):
        """Test if ``__init__`` raises ValueError when no stopping condtion is specified."""
        self.assertRaises(ValueError, StopAfter)

    def test_stop_after_epochs(self):
        """Test epochs stopping condition."""
        # Test hook does not terminate training prematurely.
        hook = StopAfter(epochs=10)
        hook.after_epoch(epoch_id=5, epoch_data=None)
        hook.after_epoch(epoch_id=9, epoch_data=None)

        # Test hook does terminate the training correctly
        self.assertRaises(TrainingTerminated, hook.after_epoch, 10, None)
        self.assertRaises(TrainingTerminated, hook.after_epoch, 20, None)

    def test_stop_after_iters(self):
        """Test iterations stopping condition."""
        # Test hook does not terminate training prematurely.
        hook = StopAfter(iterations=10)
        for i in range(15):
            hook.after_batch(stream_name=NOTRAIN_STREAM_NAME, batch_data=None)

        for i in range(9):
            hook.after_batch(stream_name=MainLoop.TRAIN_STREAM, batch_data=None)

        hook.after_batch(stream_name=NOTRAIN_STREAM_NAME, batch_data=None)

        # Test hook does terminate the training correctly
        self.assertRaises(TrainingTerminated, hook.after_batch, stream_name=MainLoop.TRAIN_STREAM, batch_data=None)

    def test_stop_after_minutes(self):
        """Test iterations stopping condition."""
        # Test hook does not terminate training prematurely.
        hook = StopAfter(minutes=1./60)
        hook.before_training()
        hook.after_batch(stream_name=NOTRAIN_STREAM_NAME, batch_data=None)
        hook.after_epoch(epoch_id=1, epoch_data=None)
        time.sleep(1)

        # Test hook does terminate the training correctly
        self.assertRaises(TrainingTerminated, hook.after_batch, stream_name=NOTRAIN_STREAM_NAME, batch_data=None)
        self.assertRaises(TrainingTerminated, hook.after_epoch, epoch_id=1, epoch_data=None)
