import logging

from .abstract_hook import AbstractHook, TrainingTerminated


class EpochStopperHook(AbstractHook):
    """Stop the training after given number of epochs."""

    def __init__(self, epoch_limit: int, **kwargs):
        """
        :param epoch_limit: maximum number of training epochs
        """
        super().__init__(**kwargs)
        self._epoch_limit = epoch_limit

    def after_epoch(self, epoch_id: int, **kwargs) -> None:
        if epoch_id >= self._epoch_limit:
            logging.info('EpochStopperHook triggered')
            raise TrainingTerminated('Training terminated after epoch {}'.format(epoch_id))
