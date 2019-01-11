"""
Hook computing epoch statistics for classification tasks.
"""

from typing import Mapping, List, Union, Optional
import logging

try:
    import sklearn.metrics as sk
except ImportError:
    logging.info('This hook requires SciKit.')

from . import AccumulateVariables
from ..types import EpochData


class ClassificationMetrics(AccumulateVariables):
    """
    Accumulate the specified prediction and gt variables and compute their classification statistics after each epoch.
    In particular, accuracy, precisions, recalls, f1s and sometimes specificity (if f1_average is set to 'binary') are
    computed and saved to epoch data.

    .. warning::
        Specificity will be computed only if `f1_average` is set to `binary`.

    .. code-block:: yaml
        :caption: Compute and save classification statistics between model output
                  `prediction` and stream source `labels`.

        hooks:
          - ClassificationMetrics:
              predicted_variable: prediction
              gt_variable: labels
    """

    def __init__(self, predicted_variable: str, gt_variable: str, f1_average: Optional[str]=None,
                 var_prefix: str='', **kwargs):
        """
        :param predicted_variable: name of the predicted variable.
        :param gt_variable: name of the ground truth variable
        :param f1_average: averaging type {binary, micro, macro, weighted, samples} defined by
                           `sklearn.metrics.precision_recall_fscore_support
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html>`_
        :param var_prefix: prefix for the output variables to avoid name conflicts; e.g. `classification_`
        """
        super().__init__(variables=[predicted_variable, gt_variable], **kwargs)

        self._predicted_variable = predicted_variable
        self._gt_variable = gt_variable
        self._f1_average = f1_average
        self._var_prefix = var_prefix

    def _get_metrics(self, gt: List[float], predicted: List[float]) -> Mapping[str, Union[float, List[float]]]:
        """Compute accuracy, precision, recall, f1 and sometimes specificity (if f1_average is set to 'binary')."""
        metrics = {}
        metrics[self._var_prefix+'precision'], metrics[self._var_prefix+'recall'], metrics[self._var_prefix+'f1'], _ = \
            sk.precision_recall_fscore_support(gt, predicted, average=self._f1_average)
        metrics[self._var_prefix+'accuracy'] = sk.accuracy_score(gt, predicted, normalize=True)
        if self._f1_average == 'binary':
            tn, fp, fn, tp = sk.confusion_matrix(gt, predicted).ravel()
            metrics[self._var_prefix+'specificity'] = tn / (tn + fp)
        return metrics

    def _save_metrics(self, epoch_data: EpochData) -> None:
        """
        Compute the classification statistics from the accumulator and save the results to the given epoch data.
        Set up 'accuracy', 'precision', 'recall', 'f1' and sometimes 'specificity' (if f1_average is set to 'binary')
        epoch data variables prefixed with self._var_prefix.

        :param epoch_data: epoch data to save the results to
        :raise ValueError: if the output variables are already set
        """
        for stream_name, stream_data in epoch_data.items():
            # variables are already checked in the AccumulatingHook; hence, we do not check them here
            metrics = self._get_metrics(self._accumulator[stream_name][self._gt_variable],
                                        self._accumulator[stream_name][self._predicted_variable])

            for var_name, var_data in metrics.items():
                if var_name in stream_data:
                    raise ValueError('Variable `{}` is set more than once for stream `{}` in epoch data. '
                                     'Use `var_prefix` parameter to avoid name conflicts.'
                                     .format(var_name, stream_name))

                stream_data[var_name] = var_data

    def after_epoch(self, epoch_data: EpochData, **kwargs) -> None:
        """Compute and save the classification statistics and reset the accumulator."""
        self._save_metrics(epoch_data)
        super().after_epoch(**kwargs)
