"""
Test module for SigintHook (cxflow.hooks.sigint_hook).
"""
import os
import signal

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.sigint_hook import SigintHook
from cxflow.hooks.abstract_hook import TrainingTerminated


class SigintHookTest(CXTestCase):

    def _sigint_handler(self, *_):
        self._sigint_unhandled = True

    def setUp(self):
        """Register the _sigint_handler."""
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._sigint_unhandled = False

    def test_before_training(self):
        """Test SigintHook does not handle the sigint before the training itself."""
        try:
            SigintHook()
            os.kill(os.getpid(), signal.SIGINT)
            self.assertTrue(self._sigint_unhandled, 'SigintHook handles SIGINT before training')
        except TrainingTerminated:
            self.fail('SigintHook raises before training')

    def test_inside_training_no_raise(self):
        """Test SigintHook does handle the sigint during the training."""
        try:
            hook = SigintHook()
            hook.before_training()
            os.kill(os.getpid(), signal.SIGINT)
            self.assertFalse(self._sigint_unhandled, 'SigintHook does not handle SIGINT while training')
            # double SIGINT cannot be easily tested (it quits(1))
        except TrainingTerminated:
            self.fail('SigintHook raised outside of after_batch()')

    def test_inside_training_raise(self):
        """Test SigintHook does rise TrainingTerminated exception."""
        raised = False
        try:
            hook = SigintHook()
            hook.before_training()
            os.kill(os.getpid(), signal.SIGINT)
            hook.after_batch()
        except TrainingTerminated:
            raised = True
        self.assertTrue(raised, 'SigintHook does not raise')

    def test_after_training(self):
        """Test SigintHook does not handle the sigint after the training."""
        try:
            hook = SigintHook()
            hook.before_training()
            hook.after_batch()
            hook.after_epoch(epoch_id=1, epoch_data=None)
            hook.after_training()
            os.kill(os.getpid(), signal.SIGINT)
            self.assertTrue(self._sigint_unhandled, 'SigintHook handles SIGINT after training')
        except TrainingTerminated:
            self.fail('SigintHook raises after training')
