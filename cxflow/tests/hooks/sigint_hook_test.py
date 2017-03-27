from cxflow.hooks.sigint_hook import SigintHook
from cxflow.hooks.abstract_hook import TrainingTerminated

import logging
import os
import signal
from unittest import TestCase


class SigintHookTest(TestCase):

    def _sigint_handler(self, signum, frame):
        self._sigint_unhandled = True

    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def setUp(self):
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._sigint_unhandled = False

    def test_before_training(self):
        try:
            hook = SigintHook(net=None, config=None)
            os.kill(os.getpid(), signal.SIGINT)
            self.assertTrue(self._sigint_unhandled,
                            'SigintHook handles SIGINT before training')
        except TrainingTerminated:
            self.fail('SigintHook raises before training')

    def test_inside_training_no_raise(self):
        try:
            hook = SigintHook(net=None, config=None)
            hook.before_training()
            os.kill(os.getpid(), signal.SIGINT)
            self.assertFalse(self._sigint_unhandled,
                             'SigintHook does not handle SIGINT while training')
            # double SIGINT cannot be easily tested (it quits(1))
        except TrainingTerminated:
            self.fail('SigintHook raised outside of after_batch()')

    def test_inside_training_raise(self):
        raised = False;
        try:
            hook = SigintHook(net=None, config=None)
            hook.before_training()
            os.kill(os.getpid(), signal.SIGINT)
            hook.after_batch()
        except TrainingTerminated:
            raised = True;
        self.assertTrue(raised, 'SigintHook does not raise')

    def test_after_training(self):
        try:
            hook = SigintHook(net=None, config=None)
            hook.before_training()
            hook.after_batch()
            hook.after_epoch(epoch_id=1, train_results=None, valid_results=None)
            hook.after_training()
            os.kill(os.getpid(), signal.SIGINT)
            self.assertTrue(self._sigint_unhandled,
                            'SigintHook handles SIGINT after training')
        except TrainingTerminated:
            self.assertTrue(raised, 'SigintHook raises after training')
