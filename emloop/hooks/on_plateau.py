"""
Module with OnPlateau abstract hook which call ``_on_plateau_action`` when the observed variable reaches its plateau.
"""

import numpy as np
from abc import abstractmethod, ABCMeta

from . import AbstractHook, ComputeStats
from ..types import EpochData



class OnPlateau(ComputeStats, metaclass=ABCMeta):
    """
    Base hook for hooks taking actions when certain variable reaches its plateau.
    The variable is observed on epoch level and plateau is reached when
    its ``long_term`` mean is lower/greater than the ``short_term`` mean.

    Call :py:meth:`_on_plateau_action` method when the observed variable reaches its plateau.
    """

    _AGGREGATION = 'mean'
    """Epoch aggregation method of the observed variable."""

    OBJECTIVES = {'min', 'max'}
    """Possible objectives for the observed variable."""

    def __init__(self,
                 long_term: int=50,
                 short_term: int=10,
                 stream: str='valid',
                 variable: str='loss',
                 objective: str='min',
                 **kwargs):
        """
        Create new OnPlateau hook.

        :param long_term: count of last epochs representing long training period
        :param short_term: count of last epochs representing short training period
        :param stream: name of the processed stream
        :param variable: name of the observed variable
        :param objective: observed variable objective; one of :py:attr:`OnPlateau.OBJECTIVES`
        :param kwargs: ignored

        :raise AssertionError: if ``long_term`` < ``short_term``
        """

        super().__init__(variables={variable: [OnPlateau._AGGREGATION]}, **kwargs)

        assert long_term >= short_term, '``long_term`` can not be smaller than ``short_term``'
        self._long_term = long_term
        self._short_term = short_term
        self._stream = stream
        self._variable = variable
        self._objective = objective
        self._saved_loss = []

    @abstractmethod
    def _on_plateau_action(self, **kwargs) -> None:
        """
        Abstract method which is called when the observed variable reaches its plateau.
        """

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        """
        Call :py:meth:`_on_plateau_action` if the ``long_term``
        variable mean is lower/greater than the ``short_term`` mean.
        """

        super().after_epoch(epoch_id=epoch_id, epoch_data=epoch_data)

        self._saved_loss.append(epoch_data[self._stream][self._variable][OnPlateau._AGGREGATION])

        long_mean = np.mean(self._saved_loss[-self._long_term:])
        short_mean = np.mean(self._saved_loss[-self._short_term:])
        if self._objective == 'min' and long_mean < short_mean:
            self._on_plateau_action(epoch_id=epoch_id, epoch_data=epoch_data)
        elif self._objective == 'max' and long_mean > short_mean:
            self._on_plateau_action(epoch_id=epoch_id, epoch_data=epoch_data)
