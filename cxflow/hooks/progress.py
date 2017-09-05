"""
Module with the hook which shows progress of the current epoch.
"""

import shutil
import collections

from . import AbstractHook
from ..datasets import AbstractDataset


class ShowProgress(AbstractHook):
    """
    Show progress of a processed stream in the current epoch.

    .. code-block:: yaml
        :caption: show progress of the current epoch

        hooks:
          - ShowProgress
    """

    def __init__(self, **kwargs):
        """Create new ShowProgress hook."""
        super().__init__(**kwargs)
        self._batch_count_saved = {}
        self._reset()

    def _reset(self) -> None:
        """Reset `_batch_count` and `_first_batch_in_epoch` to initial values."""
        self._batch_count = collections.defaultdict(lambda: 0)
        self._first_batch_in_epoch = True

    @staticmethod
    def _print_progress_bar(iteration: int, total: int, prefix: str='',
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

        percent = "{0:.1f}".format(100 * (iteration / float(total)))
        filled_len = int(length * iteration // total)
        bar = fill * filled_len + '-' * (length - filled_len)

        print('\r%s |%s| %s/%s=%s%% %s' % (prefix, bar, iteration, total, percent, suffix), end='\r')

    @staticmethod
    def _erase_line():
        """Erase the current line."""
        print('\x1b[2K', end='\r')

    def after_batch(self, stream_name: str, batch_data: AbstractDataset.Batch):
        """
        For the first epoch just the count of processed batches is displayed, because the size of the
        current stream is unknown. For the following epochs the progress bar is displayed.

        param stream_name: name of current stream
        """

        self._batch_count[stream_name] += 1

        if not self._first_batch_in_epoch:
            self._erase_line()
        else:
            self._first_batch_in_epoch = False

        current_batch = self._batch_count[stream_name]
        prefix = 'Progress of {} stream:'.format(stream_name)
        suffix = 'complete'

        terminal_width = shutil.get_terminal_size().columns

        # not first epochs
        if self._batch_count_saved:
            total_batches = self._batch_count_saved[stream_name]
            extra_chars_count = 5
            bar_len = terminal_width - (extra_chars_count + len(prefix) +
                                        len(suffix) + len("{0}/{0}=100.0%".format(total_batches)))

            if bar_len > 0:
                self._print_progress_bar(current_batch, total_batches, prefix=prefix, suffix=suffix, length=bar_len)
            else:
                # print progress as short as possible
                print("{}: {}/{}".format(stream_name, current_batch, total_batches), end='\r')

            # erase progress bar after last batch
            if total_batches == current_batch:
                self._erase_line()

        # first epoch
        else:
            progress_msg = "{} {} {}".format(prefix, current_batch, suffix)
            if len(progress_msg) <= terminal_width:
                print(progress_msg, end='\r')
            else:
                # print progress as short as possible
                print("{}: {}".format(stream_name, current_batch), end='\r')

    def after_epoch(self, **_):
        """
        After the first epoch the count of batches is saved, because of displaying the progress bar.
        After every epoch the current batch count is reseted.
        """
        # after first epoch
        if not self._batch_count_saved:
            self._batch_count_saved = self._batch_count.copy()
        self._reset()
