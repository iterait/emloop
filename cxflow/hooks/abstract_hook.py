"""
This module defines AbstractHook from which all the custom hooks shall be derived.

Furthermore, TrainingTerminated exception is defined.
"""
from ..datasets.abstract_dataset import AbstractDataset
from ..nets.abstract_net import AbstractNet
from ..utils.profile import Timer


class TrainingTerminated(Exception):
    """Exception that is raised when a hook terminates the training."""
    pass


class AbstractHook:
    """
    Hook interface.

    The hook lifecycle of hook is as follows:
    1) The hook is constructed `__init__`.
    2) `before_training` is triggered.
    3) The valid/test dataset is iterated and their results are pushed to `before_first_epoch`.
       This "epoch" is referred as zero.
    4) After each batch, regardless the stream type, `after_batch` is triggered.
    5) After an epoch is over, the summary statistics are passed to `after_epoch` together with epoch_id.
    6) When the whole training is over, `after_training` is triggered.
    """

    def __init__(self, net: AbstractNet, config: dict, dataset: AbstractDataset, **kwargs):
        """
        Create hook.
        :param net: net to be trained
        :param config: hook config dict
        :param dataset: dataset to be trained with
        :param kwargs: additional kwargs
        """
        pass

    def before_training(self, **kwargs) -> None:
        """
        Before training event.

        No data were processed at this moment.

        This is called exactly once during the training.

        :param kwargs: additional event kwargs
        """
        pass

    def before_first_epoch(self, valid_results: AbstractDataset.Batch, test_results: AbstractDataset.Batch=None,
                           **kwargs) -> None:
        """
        Before first epoch event.

        Test and Valid streams were already evaluated and their results are aggregated in valid and test results.

        This is called exactly once during the training.

        :param valid_results: averaged results from the valid stream
        :param test_results: averaged results from the test stream
        :param kwargs: additional event kwargs
        """
        pass

    def after_batch(self, stream_type: str, results: AbstractDataset.Batch, **kwargs) -> None:
        """
        After batch event.

        This event is triggered after every processed batch regardless of stream type.
        Batch results are available in results argument.

        :param stream_type: one of {'train', 'valid', 'test'}
        :param results: batch results
        :param kwargs: additional event kwargs
        """
        pass

    def after_epoch(self, epoch_id: int, train_results: AbstractDataset.Batch, valid_results: AbstractDataset.Batch,
                    test_results: AbstractDataset.Batch=None, **kwargs) -> None:
        """
        After epoch event.

        This event is triggered after every epoch wherein all the streams were iterated and their results are available
        in aggregated (averaged) form. For any other aggregation method, one must manually handle `after_batch` events.

        :param epoch_id: finished epoch id
        :param train_results: averaged results from train stream
        :param valid_results: averaged results from valid stream
        :param test_results: averaged results from test stream
        :param kwargs: additional event kwargs
        """
        pass

    def after_epoch_profile(self, epoch_id: int, profile: Timer.TimeProfile, **kwargs) -> None:
        """
        After epoch profile event.

        This event provides opportunity to process time profile of the finished epoch.

        Note: time of processing this event is not included in the profiled

        This is called multiple times.

        :param epoch_id: finished epoch id
        :param profile: dictionary of lists of event timings that were measured during the epoch.
        :param kwargs: additional event arguments
        """
        pass

    def after_training(self, **kwargs) -> None:
        """
        After training event.

        This event is called after the training finished either naturally or thanks to an interrupt.

        This is called exactly once during the training.

        :param kwargs: additional event kwargs
        """
        pass
