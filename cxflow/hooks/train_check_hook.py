from cxflow.hooks.abstract_hook import AbstractHook, TrainingTerminated
from cxflow.datasets.abstract_dataset import AbstractDataset


class TrainCheckHook(AbstractHook):
    """
    Terminate training normally if the model is trained to given valid accuracy in given number of epochs;
    raise `ValueError` otherwise.
    """

    def __init__(self, required_valid_accuracy: float, max_epoch: int, **kwargs):
        super().__init__(required_valid_accuracy=required_valid_accuracy, max_epoch=max_epoch, **kwargs)
        self._required_valid_accuracy = required_valid_accuracy
        self._max_epoch = max_epoch

    def after_epoch(self, epoch_id: int, valid_results: AbstractDataset.Batch=None, **kwargs):
        if valid_results['accuracy'] > self._required_valid_accuracy:
            raise TrainingTerminated('Valid accuracy level matched.')
        elif epoch_id >= self._max_epoch:
            raise ValueError('Valid accuracy was only {} in epoch {}, but {} was required. Training failed.'.format(
                valid_results['accuracy'], epoch_id, self._required_valid_accuracy
            ))
