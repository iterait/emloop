"""
Module with EveryNEpoch abstract hook which call ``_after_n_epoch`` method every n epoch.
"""

from . import AbstractHook
from abc import abstractmethod, ABCMeta


class EveryNEpoch(AbstractHook, metaclass=ABCMeta):
    """
    This hook should be used as base hook in the case when some action need to be processed every n epoch.
    Call ``_after_n_epoch`` method every ``n_epochs`` epoch.
    """

    def __init__(self, n_epochs: int=1, **kwargs):
        """
        Create EveryNEpoch hook.

        :param n_epochs: how often ``_after_n_epoch`` method is called
        """
        super().__init__(**kwargs)
        self._n_epochs = n_epochs

    @abstractmethod
    def _after_n_epoch(**kwargs):
        """
        Abstract method which is called every `n_epochs` epoch.
        This method must be overridden.

        :raise TypeError: if this method is not overridden
        """

    def after_epoch(self, epoch_id: int, **kwargs) -> None:
        """
        Call ``_after_n_epoch`` method every ``n_epochs`` epoch.

        :param epoch_id: number of the processed epoch
        """
        if epoch_id % self._n_epochs == 0:
            self._after_n_epoch(epoch_id=epoch_id, **kwargs)
