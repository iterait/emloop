from .abstract_hook import AbstractHook

from sklearn.metrics import precision_recall_fscore_support

import typing


class ClassificationInfoHook(AbstractHook):
    """Provides basic classification statistics such as fscore, recall and precision."""

    def __init__(self, predicted_variable: str, gold_variable: str, f1_average: str, **kwargs):
        """
        Note that `predicted_variable` and `gold_variable` must be included in evaluated variables.
        :param predicted_variable: name of the predicted variable.
        :param gold_variable: name of the gold variable
        :param f1_average: averaging type {micro, macro, weighted, samples} defined by
                           `sklearn.metrics.precision_recall_fscore_support`
        """
        super().__init__(**kwargs)

        self._predicted_variable = predicted_variable
        self._gold_variable = gold_variable
        self._f1_average = f1_average

        self._reset()

    def _reset(self):
        """Reset all predicted and ground truth buffers."""
        self._train_predicted = []
        self._train_gold = []
        self._valid_predicted = []
        self._valid_gold = []
        self._test_predicted = []
        self._test_gold = []

    def after_batch(self, stream_type: str, results: dict, **kwargs) -> None:
        """Save predicted and gold classes of this batch."""
        if stream_type == 'train':
            self._train_predicted += list(results[self._predicted_variable])
            self._train_gold += list(results[self._gold_variable])
        elif stream_type == 'valid':
            self._valid_predicted += list(results[self._predicted_variable])
            self._valid_gold += list(results[self._gold_variable])
        elif stream_type == 'test':
            self._test_predicted += list(results[self._predicted_variable])
            self._test_gold += list(results[self._gold_variable])
        else:
            raise ValueError('stream_type must be either train, valid or test')

    def before_first_epoch(self, valid_results: dict, test_results: dict=None, **kwargs) -> None:
        """Add precision, recall and fscore to the valid and test results."""

        valid_precision, valid_recall, valid_fscore, _ = self._get_results(self._valid_gold, self._valid_predicted)
        valid_results['precision'] = valid_precision
        valid_results['recall'] = valid_recall
        valid_results['fscore'] = valid_fscore

        test_precision, test_recall, test_fscore, _ = self._get_results(self._test_gold, self._test_predicted)
        test_results['precision'] = test_precision
        test_results['recall'] = test_recall
        test_results['fscore'] = test_fscore

        self._reset()

    def after_epoch(self, epoch_id: int, train_results: dict, valid_results: dict, test_results: dict=None,
                    **kwargs) -> None:
        """Add precision, recall and fscore to the train, valid and test results."""

        train_precision, train_recall, train_fscore, _ = self._get_results(self._train_gold, self._train_predicted)
        train_results['precision'] = train_precision
        train_results['recall'] = train_recall
        train_results['fscore'] = train_fscore

        valid_precision, valid_recall, valid_fscore, _ = self._get_results(self._valid_gold, self._valid_predicted)
        valid_results['precision'] = valid_precision
        valid_results['recall'] = valid_recall
        valid_results['fscore'] = valid_fscore

        test_precision, test_recall, test_fscore, _ = self._get_results(self._test_gold, self._test_predicted)
        test_results['precision'] = test_precision
        test_results['recall'] = test_recall
        test_results['fscore'] = test_fscore

        self._reset()

    def _get_results(self, gold: list, predicted: list) -> typing.Tuple[float, float, float, float]:
        """Compute precision, recall, fscore and support."""
        return precision_recall_fscore_support(gold, predicted, average=self._f1_average)
