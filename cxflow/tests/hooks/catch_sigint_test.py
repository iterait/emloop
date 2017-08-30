"""
Test module for :py:class:`cxflow.hooks.catch_sigint_hook`.
"""
import os
import signal

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.catch_sigint import CatchSigint
from cxflow.hooks.abstract_hook import TrainingTerminated


class CatchSigintTest(CXTestCase):
    """Test case for ``CatchSigint`` hook."""

    def _sigint_handler(self, *_):
        self._sigint_unhandled = True

    def setUp(self):
        """Register the _sigint_handler."""
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._sigint_unhandled = False

    def test_before_training(self):
        """Test SigintHook does not handle the sigint before the training itself."""
        CatchSigint()
        os.kill(os.getpid(), signal.SIGINT)
        self.assertTrue(self._sigint_unhandled, 'SigintHook handles SIGINT before training')

    def test_inside_training_no_raise(self):
        """Test ``SigintHook`` does handle the sigint during the training."""
        hook = CatchSigint()
        hook.before_training()
        os.kill(os.getpid(), signal.SIGINT)
        self.assertFalse(self._sigint_unhandled, 'SigintHook does not handle SIGINT while training')
        # double SIGINT cannot be easily tested (it quits(1))

    def test_inside_training_raise(self):
        """Test ``SigintHook`` does rise TrainingTerminated exception."""
        with self.assertRaises(TrainingTerminated):
            hook = CatchSigint()
            hook.before_training()
            os.kill(os.getpid(), signal.SIGINT)
            hook.after_batch()

    def test_after_training(self):
        """Test ``SigintHook`` does not handle the sigint after the training."""
        hook = CatchSigint()
        hook.before_training()
        hook.after_batch()
        hook.after_epoch(epoch_id=1, epoch_data=None)
        hook.after_training()
        os.kill(os.getpid(), signal.SIGINT)
        self.assertTrue(self._sigint_unhandled, 'SigintHook handles SIGINT after training')
