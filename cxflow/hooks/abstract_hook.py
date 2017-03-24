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

    def __init__(self, net: AbstractNet, config: dict, **kwargs):
        pass

    def before_training(self, **kwargs) -> None:
        pass

    def before_first_epoch(self, valid_results: AbstractDataset.Batch, test_results: AbstractDataset.Batch=None,
                           **kwargs) -> None:
        """Valid and test results are dictionaries of summary statistics of the zeroth epoch."""
        pass

    def after_batch(self, stream_type: str, results: AbstractDataset.Batch, **kwargs) -> None:
        """
        Stream type represents "train"/"valid"/"test" string. The results are dictionary of summary
        statistics of the previous batch
        """
        pass

    def after_epoch(self, epoch_id: int, train_results: AbstractDataset.Batch, valid_results: AbstractDataset.Batch,
                    test_results: AbstractDataset.Batch=None, **kwargs) -> None:
        """Train, valid and test results are dictionaries of summary statistics of the `epoch_id`-th epoch."""
        pass

    def after_epoch_profile(self, epoch_id: int, profile: Timer.TimeProfile, **kwargs) -> None:
        """
        This event provides opportunity to process time profile of the finished epoch.

        Note: time of processing this event is not included in the profiled

        :param epoch_id: finished epoch id
        :param profile: dictionary of lists of event timings that were measured during the epoch.
        :param kwargs: additional arguments
        :return: None
        """
        pass

    def after_training(self, **kwargs) -> None:
        pass
