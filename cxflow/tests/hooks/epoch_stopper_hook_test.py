from cxflow.hooks.epoch_stopper_hook import EpochStopperHook
from cxflow.hooks.abstract_hook import TrainingTerminated

import logging
from unittest import TestCase


class EpochStopperHookTest(TestCase):
    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_not_raise(self):
        try:
            hook = EpochStopperHook(epoch_limit=10, net=None, config=None)
            hook.after_epoch(epoch_id=5)
        except Exception:
            self.fail('EpochStopperHook(10) raised at epoch 5')

        try:
            hook = EpochStopperHook(epoch_limit=10, net=None, config=None)
            hook.after_epoch(epoch_id=9)
        except Exception:
            self.fail('EpochStopperHook(10) raised at epoch 9')

    def test_raise(self):

        hook = EpochStopperHook(epoch_limit=10, net=None, config=None)
        self.assertRaises(TrainingTerminated, hook.after_epoch, 10)
        self.assertRaises(TrainingTerminated, hook.after_epoch, 20)
