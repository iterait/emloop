from .abstract_hook import AbstractHook, TrainingTerminated

import logging


class EpochStopperHook(AbstractHook):
    def __init__(self, epoch_limit: int, **kwargs):
        super().__init__(**kwargs)
        self.epoch_limit = epoch_limit

    def after_epoch(self, epoch_id: int, **kwargs) -> None:
        if epoch_id >= self.epoch_limit:
            logging.info('EpochStopperHook triggered')
            raise TrainingTerminated('Training terminated after epoch {}'.format(epoch_id))
