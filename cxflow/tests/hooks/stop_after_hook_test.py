"""
Test module for epoch stopper hook (cxflow.hooks.stop_after_hook).
"""
from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.stop_after_hook import StopAfter
from cxflow.hooks.abstract_hook import TrainingTerminated


class EpochStopperHookTest(CXTestCase):
    """Test case for StopAfter hook."""

    def test_not_raise(self):
        """Test hook does not terminate training prematurely."""
        try:
            hook = StopAfter(epochs=10, model=None, config=None, dataset=None, output_dir=None)
            hook.after_epoch(epoch_id=5, epoch_data=None)
        except TrainingTerminated:
            self.fail('EpochStopperHook(10) raised at epoch 5')

        try:
            hook = StopAfter(epochs=10, model=None, config=None, dataset=None, output_dir=None)
            hook.after_epoch(epoch_id=9, epoch_data=None)
        except TrainingTerminated:
            self.fail('EpochStopperHook(10) raised at epoch 9')

    def test_raise(self):
        """ Test hook does terminate the training correctly."""
        hook = StopAfter(epochs=10, model=None, config=None, dataset=None, output_dir=None)
        self.assertRaises(TrainingTerminated, hook.after_epoch, 10)
        self.assertRaises(TrainingTerminated, hook.after_epoch, 20)
