"""
Module with hooks saving the trained model under certain criteria.
"""
import logging

import numpy as np

from . import AbstractHook, EveryNEpoch
from ..models import AbstractModel
from ..types import EpochData


class SaveEvery(EveryNEpoch):
    """
    Save the model every ``n_epochs`` epoch.

    .. code-block:: yaml
        :caption: save every 10th epoch

        hooks:
          - SaveEvery:
              n_epochs: 10

    .. code-block:: yaml
        :caption: save every epoch and only warn on failure

        hooks:
          - SaveEvery:
              on_failure: warn

    """

    SAVE_FAILURE_ACTIONS = ['error', 'warn', 'ignore']
    """Action to be executed when model save fails."""

    def __init__(self, model: AbstractModel, on_failure: str='error', **kwargs):
        """
        :param model: trained model
        :param on_failure: action to be taken when model fails to save itself; one of :py:attr:`SAVE_FAILURE_ACTIONS`
        """
        super().__init__(model=model, **kwargs)
        assert on_failure in SaveEvery.SAVE_FAILURE_ACTIONS

        self._model = model
        self._on_save_failure = on_failure

    def _after_n_epoch(self, epoch_id: int, **_) -> None:
        """
        Save the model every ``n_epochs`` epoch.

        :param epoch_id: number of the processed epoch
        """
        SaveEvery.save_model(model=self._model, name_suffix=str(epoch_id), on_failure=self._on_save_failure)

    @staticmethod
    def save_model(model: AbstractModel, name_suffix: str, on_failure: str) -> None:
        """
        Save the given model with the given name_suffix. On failure, take the specified action.

        :param model: the model to be saved
        :param name_suffix: name to be used for saving
        :param on_failure: action to be taken on failure; one of :py:attr:`SAVE_FAILURE_ACTIONS`
        :raise IOError: on save failure with ``on_failure`` set to ``error``
        """
        try:
            logging.debug('Saving the model')
            save_path = model.save(name_suffix)
            logging.info('Model saved to: %s', save_path)
        except Exception as ex:  # pylint: disable=broad-except
            if on_failure == 'error':
                raise IOError('Failed to save the model.') from ex
            elif on_failure == 'warn':
                logging.warning('Failed to save the model.')


class SaveBest(AbstractHook):
    """
    Maintain the best performing model given the specified criteria.

    .. code-block:: yaml
        :caption: save model with minimal valid loss

        hooks:
          - BestSaverHook

    .. code-block:: yaml
        :caption: save model with max accuracy

        hooks:
          - SaveBest:
              variable: accuracy
              condition: max

    """

    _OUTPUT_NAME = 'best'

    OBJECTIVES = {'min', 'max'}
    """Possible objectives for the monitor variable."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 model: AbstractModel, variable: str='loss', condition: str='min', stream: str='valid',
                 aggregation: str='mean', on_save_failure: str='error', **kwargs):
        """
        Example: metric=loss, condition=min -> saved the model when the loss is best so far (on `stream`).

        :param model: trained model
        :param variable: variable name to be monitored
        :param condition: performance objective; one of :py:attr:`OBJECTIVES`
        :param stream: stream name to be monitored
        :param aggregation: variable aggregation to be used (``mean`` by default)
        :param on_save_failure: action to be taken when model fails to save itself, one of
            :py:attr:`SaveEvery.SAVE_FAILURE_ACTIONS`
        """

        assert on_save_failure in SaveEvery.SAVE_FAILURE_ACTIONS
        assert condition in SaveBest.OBJECTIVES

        super().__init__(**kwargs)
        self._model = model
        self._variable = variable
        self._condition = condition
        self._stream_name = stream
        self._aggregation = aggregation
        self._on_save_failure = on_save_failure

        self._best_value = None

    def _get_value(self, epoch_data: EpochData) -> float:
        """
        Retrieve the value of the monitored variable from the given epoch data.

        :param epoch_data: epoch data which determine whether the model will be saved or not
        :raise KeyError: if any of the specified stream, variable or aggregation is not present in the ``epoch_data``
        :raise TypeError: if the variable value is not a dict when aggregation is specified
        :raise ValueError: if the variable value is not a scalar
        """
        if self._stream_name not in epoch_data:
            raise KeyError('Stream `{}` was not found in the epoch data.\nAvailable streams are `{}`.'
                           .format(self._stream_name, epoch_data.keys()))

        stream_data = epoch_data[self._stream_name]
        if self._variable not in stream_data:
            raise KeyError('Variable `{}` for stream `{}` was not found in the epoch data. '
                           'Available variables for stream `{}` are `{}`.'
                           .format(self._variable, self._stream_name, self._stream_name, stream_data.keys()))

        value = stream_data[self._variable]
        if self._aggregation:
            if not isinstance(value, dict):
                raise TypeError('Variable `{}` is expected to be a dict when aggregation is specified. '
                                'Got `{}` instead.'.format(self._variable, type(value).__name__))
            if self._aggregation not in value:
                raise KeyError('Specified aggregation `{}` was not found in the variable `{}`. '
                               'Available aggregations: `{}`.'.format(self._aggregation, self._variable, value.keys()))
            value = value[self._aggregation]
        if not np.isscalar(value):
            raise ValueError('Variable `{}` value is not a scalar.'.format(value))

        return value

    def _is_value_better(self, new_value: float) -> bool:
        """
        Test if the new value is better than the best so far.

        :param new_value: current value of the objective function
        """
        if self._best_value is None:
            return True
        if self._condition == 'min':
            return new_value < self._best_value
        if self._condition == 'max':
            return new_value > self._best_value

    def after_epoch(self, epoch_data: EpochData, **_) -> None:
        """
        Save the model if the new value of the monitored variable is better than the best value so far.

        :param epoch_data: epoch data to be processed
        """
        new_value = self._get_value(epoch_data)

        if self._is_value_better(new_value):
            self._best_value = new_value
            SaveEvery.save_model(model=self._model, name_suffix=self._OUTPUT_NAME, on_failure=self._on_save_failure)


class SaveLatest(AbstractHook):
    """
    Save the latest model.

    .. code-block:: yaml
        :caption: save the latest model

        hooks:
          - SaveLatest
    """

    _OUTPUT_NAME = 'latest'

    def __init__(self, model: AbstractModel, on_save_failure: str='error', **kwargs):
        """
        Create new SaveLatest hook.

        :param model: trained model
        :param on_save_failure: action to be taken when model fails to save itself, one of
            :py:attr:`SaveEvery.SAVE_FAILURE_ACTIONS`
        """

        assert on_save_failure in SaveEvery.SAVE_FAILURE_ACTIONS

        super().__init__(**kwargs)
        self._model = model
        self._on_save_failure = on_save_failure

    def after_epoch(self, **_) -> None:
        """Save/override the latest model after every epoch."""
        SaveEvery.save_model(model=self._model, name_suffix=self._OUTPUT_NAME, on_failure=self._on_save_failure)
