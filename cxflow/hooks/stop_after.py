"""
Module with a hook which stops the training after the specified number of epochs.
"""
import logging

from .abstract_hook import AbstractHook, TrainingTerminated


class StopAfter(AbstractHook):
    """
    Stop the training after the specified number of epochs.

    .. code-block:: yaml
        :caption: stop the training after 500 epochs

        hooks:
          - StopAfter:
              epochs: 500
    """

    def __init__(self, epochs: int, **kwargs):
        """
        Create new epoch stopper hook.

        :param epochs: maximum number of training epochs
        """
        super().__init__(**kwargs)
        self._epoch_limit = epochs

    def after_epoch(self, epoch_id: int, **_) -> None:
        """Stop the training if `epoch_id > self._epoch_limit`."""
        if epoch_id >= self._epoch_limit:
            logging.info('EpochStopperHook triggered')
            raise TrainingTerminated('Training terminated after epoch {}'.format(epoch_id))
