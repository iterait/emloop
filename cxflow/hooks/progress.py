"""
Module with the hook which displays progress of the current epoch.
"""

import shutil

from . import AbstractHook
from ..datasets import AbstractDataset

_ERASE_LINE = '\x1b[2K'
"""Code for erasing a line."""


class Progress(AbstractHook):
    """
    Display progress of a processed stream in the current epoch.

    .. code-block:: yaml
        :caption: display progress of the current epoch

        hooks:
          - Progress
    """

    def __init__(self, **kwargs):
        """
        Create new Progress hook.
        """
        super().__init__(**kwargs)
        self._batch_count = {}
        self._batch_count_saved = None
        self._first_batch_in_epoch = True

    def _reset(self) -> None:
        """
        Reset `_batch_count` and `_first_batch_in_epoch` to initial values.
        """
        self._batch_count = {}
        self._first_batch_in_epoch = True

    def _print_progress_bar(self, iteration: int, total: int, prefix: str='',
                            suffix: str='', length: int=50, fill: str='█') -> None:
        """
        Display current state of the progress bar.

        param iteration: current step in computation
        param total: total count of steps in computation
        param prefix: info text displayed before the progress bar
        param suffix: info text displayed after the progress bar
        param length: length of progress bar
        param fill: char to be displayed as a step in the progress bar
        """

        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_len = int(length * iteration // total)
        bar = fill * filled_len + '-' * (length - filled_len)

        print('%s |%s| %s/%s=%s%% %s' % (prefix, bar, iteration, total, percent, suffix), end='\r')

    def after_batch(self, stream_name: str, batch_data: AbstractDataset.Batch):
        """
        For the first epoch just the count of processed batches is displayed, because the size of the
        current stream is unknown. For the following epochs the progress bar is displayed.

        param stream_name: name of current stream
        """

        if stream_name not in self._batch_count:
            self._batch_count[stream_name] = 1
        else:
            self._batch_count[stream_name] += 1

        if not self._first_batch_in_epoch:
            print(_ERASE_LINE, end='\r')
        else:
            self._first_batch_in_epoch = False

        current_batch = self._batch_count[stream_name]
        prefix = 'Progress of {} stream:'.format(stream_name)
        suffix = 'complete'

        if self._batch_count_saved:
            total_batches = self._batch_count_saved[stream_name]
            extra_chars_count = 5
            bar_len = shutil.get_terminal_size().columns - (extra_chars_count +
                                                            len(prefix) + len(suffix) +
                                                            len("{0}/{0}=100.0%".format(total_batches)))

            self._print_progress_bar(current_batch, total_batches, prefix=prefix, suffix=suffix, length=bar_len)

        else:
            print("{}: {} {}".format(prefix, current_batch, suffix), end='\r')

    def after_epoch(self, **_):
        """
        After the first epoch the count of batches is saved, because of displaying the progress bar.
        After every epoch the current batch count is reseted.
        """
        if not self._batch_count_saved:
            self._batch_count_saved = self._batch_count.copy()
        self._reset()
