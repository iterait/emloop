import numpy as np
import os.path as path
from typing import Optional, Sequence, Tuple

import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import AccumulateVariables
from ..types import EpochData
from ..datasets import BaseDataset
from ..utils import confusion_matrix


class SaveConfusionMatrix(AccumulateVariables):
    """
    After each epoch, compute and save/store confusion matrix figure for the predicted and expected labels.

    .. code-block:: yaml
        :caption: Store confusion matrix figure to epoch data with green colorbar

        hooks:
          - SaveConfusionMatrix:
              figure_action: store
              cmap: Greens

    .. code-block:: yaml
        :caption: Defined classes' names and save confusion matrix figure to training logdir with absolute values

        hooks:
          - SaveConfusionMatrix:
              classes_names: [class_with_index_zero, class_with_index_one, class_with_index_three]
              normalize: False

    """

    FIGURE_ACTIONS = ['save', 'store']
    """
    Possible actions to be taken with the plotted figure. 
    It can be either saved to a file or stored in the epoch data.
    """

    def __init__(self,
                 output_dir: str,
                 dataset: BaseDataset,
                 labels_name: str='labels',
                 predictions_name: str='predictions',
                 classes_names: Optional[Sequence[str]]=None,
                 figsize: Optional[Tuple[int, int]]=None,
                 figure_action: str='save',
                 num_classes_method_name: str='num_classes',
                 classes_names_method_name: str='classes_names',
                 mask_name: Optional[str]=None,
                 normalize: bool=True,
                 cmap: str='Blues', **kwargs):
        """
        Create new :py:class:`SaveConfusionMatrix` hook.

        :param output_dir: output directory
        :param dataset: dataset (needed to translate predictions to strings)
        :param labels_name: annotation variable name
        :param predictions_name: prediction variable name
        :param classes_names: List of classes' names
        :param figsize: the size of the matplotlib figure
        :param figure_action: action to be taken with the plotted figure, one of :py:attr:`FIGURE_ACTIONS`
        :param normalize: False for plotting absolute values in confusion matrix, True for relative
        :param num_classes_method_name: ``self._dataset`` method name to get number of classes
        :param classes_names_method_name: ``self._dataset`` method name to get classes' names
                                            Parameter is ignored when ``classes_names`` is provided
        :param mask_name: the variable masking valid records (1 = valid, 0 = invalid)
        :param cmap: type of colorbar  # http://matplotlib.org/examples/color/colormaps_reference.html
        :raise ValueError: if the ``figure_action`` is not in ``FIGURE_ACTIONS``
        """
        if figure_action not in SaveConfusionMatrix.FIGURE_ACTIONS:
            raise ValueError('Unrecognized figure action `{}`. It must be one of `{}`'.
                             format(figure_action, SaveConfusionMatrix.FIGURE_ACTIONS))

        self._dataset = dataset
        self._output_dir = output_dir
        self._labels_name = labels_name
        self._predictions_name = predictions_name
        self._classes_names = classes_names
        self._figsize = figsize
        self._figure_action = figure_action
        self._num_classes_method_name = num_classes_method_name
        self._classes_names_method_name = classes_names_method_name
        self._mask_name = mask_name
        self._normalize = normalize
        self._cmap = cmap

        accum_variables = [labels_name, predictions_name]
        if self._mask_name is not None:
            accum_variables.append(self._mask_name)
        super().__init__(variables=accum_variables, **kwargs)

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        for stream_name, variables in self._accumulator.items():
            predicted = np.array(variables[self._predictions_name])
            expected = np.array(variables[self._labels_name])

            # Only use the masked data if requested
            if self._mask_name is not None:
                mask = np.asarray(variables[self._mask_name]).astype(np.bool)
                predicted = predicted[mask]
                expected = expected[mask]

            # Try to get names of classes from possible sources
            classes_names = False
            if self._classes_names is not None:
                classes_names = self._classes_names
                assert len(classes_names) > np.max([predicted, expected])
            elif hasattr(self._dataset, self._classes_names_method_name):
                classes_names = getattr(self._dataset, self._classes_names_method_name)()

            # Get number of classes from possible sources
            if classes_names:
                num_classes = len(classes_names)
            elif hasattr(self._dataset, self._num_classes_method_name):
                num_classes = getattr(self._dataset, self._num_classes_method_name)()
                assert num_classes > np.max([predicted, expected])
            else:
                num_classes = np.max([predicted, expected]) + 1

            # Calculate confusion matrix (cm) with absolute values
            cm_abs = confusion_matrix(expected=expected, predicted=predicted, num_classes=num_classes)
            # Calculate cm with relative values
            cm_norm = cm_abs.astype(np.float) / np.sum(cm_abs, axis=1)[:, np.newaxis]
            cm_norm[np.isnan(cm_norm)] = 0  # Is `np.nan`s appeared, replace them by zero
            # Choose cm type
            cm = cm_norm if self._normalize else cm_abs

            # Save the heatmap of confusion matrix
            plt.figure(figsize=self._figsize)
            plt.imshow(cm, interpolation='nearest', cmap=self._cmap)
            plt.title('Predicted', y=1.1)
            plt.ylabel('Expected')
            plt.tick_params(labeltop=True, labelbottom=False, top=True, bottom=False)
            plt.colorbar()

            # Change ticks if `classes_names` found
            if classes_names:
                plt.xticks(np.arange(num_classes), classes_names, rotation=90)
                plt.yticks(np.arange(num_classes), classes_names)

            # Add both normalized and absolute values to graph
            thresh = np.nanmax(cm) / 2.  # To avoid printing dark (bright) text to dark (bright) background
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, '{:.2f} / {}'.format(cm_norm[i, j], cm_abs[i, j]), fontdict={'size': 8},
                         horizontalalignment="center", color='white' if cm[i, j] > thresh else 'black')

            plt.tight_layout()

            # Save / store the figure
            if self._figure_action == 'store':
                fig = plt.gcf()
                # Draw the figure first
                fig.canvas.draw()
                # Now we can save it to a numpy array
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                epoch_data[stream_name]['confusion_heatmap'] = data
            else:
                fig_path = path.join(self._output_dir, 'confusion_matrix_epoch_{}_{}.png'.format(epoch_id, stream_name))
                plt.savefig(fig_path)
            plt.gcf().clear()

        super().after_epoch()
