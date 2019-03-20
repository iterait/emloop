"""
This module defines AbstractHook from which all the custom hooks shall be derived.

Furthermore, TrainingTerminated exception is defined.
"""
import logging
import inspect
from typing import Iterable, List
from ..types import EpochData, Batch, TimeProfile, TrainingTerminated


class AbstractHook:
    """
    **emloop** hook interface.

    Hook lifecycle (event -> method invocation):

    1. **emloop** constructs the hooks -> :py:meth:`__init__`
    2. **emloop** enters the main loop -> :py:meth:`before_training`
        a. **emloop** starts an epoch
        b. **emloop** computes a batch -> :py:meth:`after_batch`
        c. **emloop** finishes the epoch -> :py:meth:`after_epoch` and :py:meth:`after_epoch_profile`
    3. **emloop** terminates the main loop -> :py:meth:`after_training`

    .. caution::

        Hook naming conventions:

        - hook names should describe hook actions with verb stems. E.g.: ``LogProfile`` or ``SaveBest``
        - hook names should not include ``Hook`` suffix
    """

    EL_HOOK_INIT_ARGS = {'model', 'dataset', 'output_dir'}
    """Arguments which **emloop** pass, in addition to the config args, to ``__init__``
    methods of every hook being created."""

    def __init__(self, **kwargs):
        """
        Check and warn if there is any argument created by the user yet not recognized in the child hook ``__init__``
        method.

        :param kwargs: ``**kwargs`` not recognized in the child hook
        """
        for key in kwargs:
            if key not in AbstractHook.EL_HOOK_INIT_ARGS:
                logging.warning('Argument `%s` was not recognized by `%s`. Recognized arguments are `%s`.',
                                key, type(self).__name__, list(inspect.signature(type(self)).parameters.keys()))

        self._main_loop = None

    def before_training(self) -> None:
        """
        Before training event.

        No data were processed at this moment.

        .. note::
            This method is called exactly once during the training.
        """
        pass

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        """
        After batch event.

        This event is triggered after every processed batch regardless of stream type.
        Batch results are available in results argument.

        :param stream_name: name of the stream (usually ``train``, ``valid`` or``test``)
        :param batch_data: batch inputs and model outputs
        """
        pass

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        """
        After epoch event.

        This event is triggered after every epoch wherein all the streams were iterated. The ``epoch_data`` object is
        initially empty and shared among all the hooks.

        :param epoch_id: finished epoch id
        :param epoch_data: epoch data flowing through all hooks
        """
        pass

    def after_epoch_profile(self, epoch_id: int, profile: TimeProfile, streams: List[str]) -> None:
        """
        After epoch profile event.

        This event provides opportunity to process time profile of the finished epoch.

        :param epoch_id: finished epoch id
        :param streams: streams which are in profile
        """
        pass

    def after_training(self, success: bool) -> None:
        """
        After training event.

        This event is called after the training finished either naturally or thanks to an interrupt.

        .. note::
            This method is called exactly once during the training.

        :param success: whether the training ended with successfully or with exception
        """
        pass

    def register_mainloop(self, main_loop: 'emloop.MainLoop') -> None:
        """
        Pass :py:class:`emloop.MainLoop` to hook. Raise :py:class:`ValueError` if MainLoop was already passed before.

        :param main_loop: **emloop** main loop for training
        :raise ValueError: if MainLoop was already passed before
        """
        if self._main_loop is not None:
            raise ValueError('A MainLoop was already registered with this hook.')

        self._main_loop = main_loop
