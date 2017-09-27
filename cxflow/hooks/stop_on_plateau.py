"""
Module with StopOnPlateau hook which terminates the training when the model stops improving.
"""
import numpy as np
from typing import List

from . import AbstractHook, TrainingTerminated, AccumulateVariables
from ..datasets import AbstractDataset


class StopOnPlateau(AccumulateVariables):
    """
    Terminate the training when the model stops improving.

    .. code-block:: yaml

        :caption: stop the training when the mean of last 8500 ``loss`` values is
                  smaller than the mean of last 250 ``loss`` values.
        hooks:
          - StopOnPlateau:
              long_term: 8500
              short_term: 250

    """

    def __init__(self, long_term: int=50000, short_term: int=5000, stream: str='valid', loss_var: str='loss', **kwargs):
        """
        Create new StopOnPlateau hook.

        :param long_term: count of last examples representing long training period
        :param short_term: count of last examples representing short training period
        :param stream: name of the processed stream
        :param loss_var: name of the processed loss variable
        :raise AssertionError: if ``long_term`` < ``short_term``
        """
        super().__init__(variables=[loss_var], **kwargs)

        assert long_term >= short_term, '``long_term`` can not be smaller than ``short_term``'
        self._long_term = long_term
        self._short_term = short_term
        self._stream = stream
        self._loss_var = loss_var

    def _loss_vals(self, term: int) -> List[float]:
        """
        Return the list with last ``term`` loss values.

        :param term: count of last ``loss_var`` values
        """
        return self._accumulator[self._stream][self._loss_var][-term:]

    def after_batch(self, stream_name: str, batch_data: AbstractDataset.Batch) -> None:
        """
        Stop the training if the mean of last ``long_term`` loss values is
        smaller than the mean of last ``short_term`` loss values.
        """
        super().after_batch(stream_name, batch_data)
        if stream_name == self._stream:

            loss_vals_long = self._loss_vals(self._long_term)
            loss_vals_short = self._loss_vals(self._short_term)

            if np.mean(loss_vals_long) < np.mean(loss_vals_short):
                raise TrainingTerminated('StopOnPlateau - the model '
                                         'stops improving on `{}` stream'.format(self._stream))

    def after_epoch(self, epoch_id: int, epoch_data: AbstractHook.EpochData) -> None:
        """
        Clear the ``_accumulator``.
        The maximum of last saved values of ``loss_var`` in ``stream`` is ``long_term``.
        Other values are removed from the ``_accumulator``.
        """
        loss_vals_long = self._loss_vals(self._long_term)
        super().after_epoch(epoch_id=epoch_id, epoch_data=epoch_data)
        self._accumulator[self._stream][self._loss_var] = loss_vals_long
