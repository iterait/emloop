"""
This module defines AbstractHook from which all the custom hooks shall be derived.

Furthermore, TrainingTerminated exception is defined.
"""
import logging
import inspect
from typing import Iterable, NewType, Mapping

from ..datasets import AbstractDataset
from ..utils.profile import Timer


# Arguments which cxflow pass, in addition to the config args, to init methods of every hook being created.
CXF_HOOK_INIT_ARGS = {'net', 'dataset', 'output_dir'}


class TrainingTerminated(Exception):
    """Exception that is raised when a hook terminates the training."""
    pass


class AbstractHook:
    """
    Hook interface.

    The hook lifecycle of hook is as follows:
    1) The hook is constructed `__init__`.
    2) `before_training` is triggered.
    3) After each batch, regardless the stream type, `after_batch` is triggered.
    4) After an epoch is over, the summary statistics are passed to `after_epoch` together with `epoch_id`.
       The summary statistics (epoch_data) are mutable and each hook might add new information.
    5) When the whole training is over, `after_training` is triggered.
    """

    EpochData = NewType('EpochData', Mapping[str, AbstractDataset.Batch])

    def __init__(self, **kwargs):
        """
        Check for unrecognized arguments and warn about them.
        :param kwargs: kwargs not recognized in the child hook
        """
        for key in kwargs:
            if key not in CXF_HOOK_INIT_ARGS:
                logging.warning('Argument `%s` was not recognized by `%s`. Recognized arguments are `%s`.',
                                key, type(self).__name__, list(inspect.signature(type(self)).parameters.keys()))

    def before_training(self) -> None:
        """
        Before training event.

        No data were processed at this moment.

        This is called exactly once during the training.
        """
        pass

    def after_batch(self, stream_name: str, batch_data: AbstractDataset.Batch) -> None:
        """
        After batch event.

        This event is triggered after every processed batch regardless of stream type.
        Batch results are available in results argument.

        :param stream_name: type of the stream (usually train/valid/test or any other)
        :param batch_data: batch inputs and net outputs
        """
        pass

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        """
        After epoch event.

        This event is triggered after every epoch wherein all the streams were iterated and their results are available
        in aggregated (averaged) form. For any other aggregation method, one must manually handle `after_batch` events.

        :param epoch_id: finished epoch id
        :param epoch_data: epoch data flowing through all hooks
        """
        pass

    def after_epoch_profile(self, epoch_id: int, profile: Timer.TimeProfile, extra_streams: Iterable[str]) -> None:
        """
        After epoch profile event.

        This event provides opportunity to process time profile of the finished epoch.

        Note: time of processing this event is not included in the profiled

        This is called multiple times.

        :param epoch_id: finished epoch id
        :param profile: dictionary of lists of event timings that were measured during the epoch.
        :param extra_streams: enumeration of additional stream names
        """
        pass

    def after_training(self) -> None:
        """
        After training event.

        This event is called after the training finished either naturally or thanks to an interrupt.

        This is called exactly once during the training.
        """
        pass
