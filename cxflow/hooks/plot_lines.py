"""
Hook for drawing line plots.
"""

import os
import collections
from typing import Iterable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import AbstractHook
from ..types import Batch


class PlotLines(AbstractHook):
    """
    Plot sequences of numbers using matplotlib.
    

    .. code-block:: yaml
        :caption: Plot `xs` variable for each example in test and valid streams.

        hooks:
          - PlotLines:
              variables: [xs]
              streams: [test, valid]

    .. code-block:: yaml
        :caption: Plot `xs` and `ys` variables only for the first two examples from the first ten
                  batches (from the train stream).

        hooks:
          - PlotLines:
              variables: [xs, ys]
              example_count: 2
              batch_count: 10
    """

    _ROOT_DIR = 'visual'

    def __init__(self, output_dir: str, variables: Iterable[str], streams: Optional[Iterable[str]]=None,
                 id_variable: str='ids', pad_mask_variable: Optional[str]=None, out_format: str='png',
                 ymin: float=None, ymax: float=None, example_count: int=None, batch_count: int=None, **kwargs):
        """
        Hook constructor.

        :param output_dir: output directory where plots will be saved
        :param variables: names of the variables to be plotted
        :param streams: list of stream names to be dumped; can be None to dump all streams
        :param id_variable: name of the source which represents a unique example id
        :param pad_mask_variable: name of the source which represents the padding mask
        :param out_format: extension of the saved image
        :param ymin: minimum on the Y axis
        :param ymin: maximum on the Y axis
        :param example_count: count of examples which will be plotted from each batch
                              (first ``example_count`` examples will be plotted)
        :param batch_count: count of batches from which the plot will be saved
                            (first ``batch_count`` will be processed)
        """
        super().__init__(**kwargs)

        self._output_dir = output_dir
        self._variables = variables
        self._streams = streams
        self._id_variable = id_variable
        self._pad_mask_variable = pad_mask_variable
        self._out_format = out_format
        self._ymin = ymin
        self._ymax = ymax
        self._example_count = example_count
        self._batch_count = batch_count

        self._current_epoch_id = '_'
        self._reset()

    @property
    def figure_suffix(self) -> str:
        """The suffix of the saved figure, used to distinguish between images from different hooks."""
        return '-vs-'.join(self._variables)

    def plot_figure(self, idx: int, batch_data: Batch) -> plt.Figure:
        """Plot the selected variables to a new figure."""
        fig, ax = plt.subplots()
        # extract pad mask if available
        mask = None
        if self._pad_mask_variable is not None:
            mask = np.asarray(batch_data[self._pad_mask_variable][idx], dtype=np.bool)
        # plot all variables
        for var in self._variables:
            data = batch_data[var][idx]
            if mask is not None:
                data = data[mask]
            ax.plot(data, label=var)
        # set Y axis limits
        if self._ymin is not None:
            ax.set_ylim(ymin=self._ymin)
        if self._ymax is not None:
            ax.set_ylim(ymax=self._ymax)
        ax.legend()
        fig.tight_layout()
        return fig

    def after_batch(self, stream_name: str, batch_data: Batch):
        """
        Save images in provided streams from selected variable. The amount of batches and images to be processed is
        possible to control by ``batch_count`` and ``example_count`` parameters.
        """
        if self._streams is None or stream_name in self._streams:
            # assert variables in batch data
            assert self._id_variable in batch_data
            assert self._pad_mask_variable is None or self._pad_mask_variable in batch_data
            for variable in self._variables:
                assert variable in batch_data

            # only plot the requested number of batches
            self._batch_done[stream_name] += 1
            if self._batch_count and self._batch_done[stream_name] > self._batch_count:
                return

            # create the output directory
            stream_out_dir = os.path.join(self._output_dir, self._ROOT_DIR,
                                          'epoch_{}'.format(self._current_epoch_id), stream_name)
            os.makedirs(stream_out_dir, exist_ok=True)

            # iterate through the examples and generate plots
            for i, ex_id in enumerate(batch_data[self._id_variable]):
                # only plot the requested number of examples
                if self._example_count and i + 1 > self._example_count:
                    break
                ex_id = ex_id.replace(os.sep, '___')
                filename = '{}_batch_{}_plot-{}.{}'.format(ex_id, self._batch_done[stream_name],
                                                           self.figure_suffix, self._out_format)
                fig = self.plot_figure(i, batch_data)
                fig.savefig(os.path.join(stream_out_dir, filename))
                plt.close(fig)

    def after_epoch(self, epoch_id: int, **_):
        """
        Set ``_current_epoch_id`` which is used for distinguish between epoch directories.
        Call the ``_reset`` function.
        """
        self._current_epoch_id = epoch_id + 1
        self._reset()

    def _reset(self) -> None:
        """Reset ``_batch_count`` to initial value."""
        self._batch_done = collections.defaultdict(lambda: 0)
