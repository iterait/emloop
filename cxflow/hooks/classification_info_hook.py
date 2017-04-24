"""
Module with a hook computing epoch statistics for classification tasks.
"""
from typing import Iterable, Tuple

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from .abstract_hook import AbstractHook
from .accumulating_hook import AccumulatingHook


class ClassificationInfoHook(AccumulatingHook):
    """
    Accumulate the specified prediction and gold variables
    and compute their classification statistics after each epoch.

    In particular, accuracy, precisions, recalls and fscores are computed and saved to epoch data.

    -------------------------------------------------------
    Example usage in config
    -------------------------------------------------------
    # compute and save classification statistics between net output `prediction` and stream source `labels`
    hooks:
      - class: ClassificationInfoHook
        predicted_variable: prediction
        gold_variable: labels

    # prefix the output variables
    hooks:
      - class: ClassificationInfoHook
        predicted_variable: my_var
        gold_variable: my_gold
        var_prefix: 'prefix_'
    -------------------------------------------------------
    """

    def __init__(self, predicted_variable: str, gold_variable: str, f1_average: str=None,
                 var_prefix: str='', **kwargs):
        """
        :param predicted_variable: name of the predicted variable.
        :param gold_variable: name of the gold variable
        :param f1_average: averaging type {micro, macro, weighted, samples} defined by
                           `sklearn.metrics.precision_recall_fscore_support`
        :param var_prefix: prefix for the output variables
        """
        super().__init__(variables=[predicted_variable, gold_variable], **kwargs)

        self._predicted_variable = predicted_variable
        self._gold_variable = gold_variable
        self._f1_average = f1_average
        self._var_prefix = var_prefix

    def _get_metrics(self, gold: list, predicted: list) \
            -> Tuple[float, Iterable[float], Iterable[float], Iterable[float]]:
        """Compute accuracy, precision, recall and fscore."""

        precision, recall, fscore, _ = precision_recall_fscore_support(gold, predicted, average=self._f1_average)
        accuracy = accuracy_score(gold, predicted, normalize=True)
        return accuracy, precision, recall, fscore

    def _save_metrics(self, epoch_data: AbstractHook.EpochData) -> None:
        """
        Compute the classification statistics from the accumulator and save the results to the given epoch data.

        Set up 'accuracy', 'precision', 'recall' and 'fscore' epoch data variables prefixed with self._var_prefix.

        :param epoch_data: epoch data to save the results to

        Raises:
            ValueError: if the output variables are already set
        """

        for stream_name in epoch_data.keys():
            # variables are already checked in the AccumulatingHook; hence, we do not check them here
            metrics = self._get_metrics(self._accumulator[stream_name][self._gold_variable],
                                        self._accumulator[stream_name][self._predicted_variable])

            stream_data = epoch_data[stream_name]
            var_names = [self._var_prefix+var_name for var_name in ['accuracy', 'precision', 'recall', 'fscore']]
            for i, var_name in enumerate(var_names):
                if var_name in stream_data:
                    raise ValueError('Variable `{}` is set more than once for stream `{}` in epoch data. '
                                     'Use `var_prefix` parameter to avoid name conflicts.'
                                     .format(var_name, stream_name))
                else:
                    stream_data[var_name] = metrics[i]

    def after_epoch(self, epoch_data: AbstractHook.EpochData, **kwargs) -> None:
        """Compute and save the classification statistics and reset the accumulator."""
        self._save_metrics(epoch_data)
        super().after_epoch(**kwargs)

