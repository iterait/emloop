"""
Module with a hook which stops the training after the specified number of epochs.
"""
import logging

from .abstract_hook import AbstractHook, TrainingTerminated


class EpochStopperHook(AbstractHook):
    """
    Stop the training after the specified number of epochs.

    -------------------------------------------------------
    Example usage in config
    -------------------------------------------------------
    # stop the training after 500 epochs
    hooks:
      - class: EpochStopperHook
        epochs_limit: 500
    -------------------------------------------------------
    """

    def __init__(self, epoch_limit: int, **kwargs):
        """
        Create new epoch stopper hook.
        :param epoch_limit: maximum number of training epochs
        """
        super().__init__(**kwargs)
        self._epoch_limit = epoch_limit

    def after_epoch(self, epoch_id: int, **_) -> None:
        """Stop the training if `epoch_id > self._epoch_limit`."""
        if epoch_id >= self._epoch_limit:
            logging.info('EpochStopperHook triggered')
            raise TrainingTerminated('Training terminated after epoch {}'.format(epoch_id))
