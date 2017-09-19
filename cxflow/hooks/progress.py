"""
Module with ShowProgress hook which shows progress of the current epoch.
"""

import shutil
import collections
import time

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
        self._batch_count = collections.defaultdict(lambda: 0)
        self._first_batch_in_epoch = True
        self._current_stream_start = None
        self._first_stream_in_epoch = True

    def _reset(self) -> None:
        """
        Set ``_batch_count``, ``_first_batch_in_epoch``, ``_current_stream_start``
        and ``_first_stream_in_epoch`` to initial values.
        """
        self._batch_count.clear()
        self._first_batch_in_epoch = True
        self._current_stream_start = None
        self._first_stream_in_epoch = True

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

        percent = '{0:.1f}'.format(100 * (iteration / float(total)))
        filled_len = int(length * iteration // total)
        bar = fill * filled_len + '-' * (length - filled_len)

        print('\r%s |%s| %s/%s=%s%% %s' % (prefix, bar, iteration, total, percent, suffix), end='\r')

    @staticmethod
    def _erase_line() -> None:
        """Erase the current line."""
        print('\x1b[2K', end='\r')

    @staticmethod
    def get_formatted_time(seconds: float) -> str:
        """
        Convert seconds to the time format ``H:M:S.UU``.

        :param seconds: time in seconds
        :return: formatted human-readable time
        """
        seconds = round(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return '{:d}:{:02d}:{:02d}'.format(h, m, s)

    def after_batch(self, stream_name: str, batch_data: AbstractDataset.Batch) -> None:
        """
        For the first epoch just the count of processed batches is displayed, because the size of the
        current stream is unknown. For the following epochs the progress bar is displayed.
        """

        self._batch_count[stream_name] += 1

        if not self._first_batch_in_epoch:
            self._erase_line()
        else:
            self._first_batch_in_epoch = False

        current_batch = self._batch_count[stream_name]
        prefix = 'Progress of {} stream:'.format(stream_name)

        terminal_width = shutil.get_terminal_size().columns

        # not first epochs
        if self._batch_count_saved:

            total_batches = self._batch_count_saved[stream_name]

            eta = ''
            if self._current_stream_start:
                measured_batches = current_batch - (1 if self._first_stream_in_epoch else 0)
                avg_batch_time = (time.time() - self._current_stream_start) / measured_batches
                eta_sec = avg_batch_time * (total_batches - current_batch)
                eta = self.get_formatted_time(eta_sec)
            else:
                self._current_stream_start = time.time()

            extra_chars_count = 5
            bar_len = terminal_width - (extra_chars_count + len(prefix) +
                                        len(eta) + len('{0}/{0}=100.0%'.format(total_batches)))

            if bar_len > 0:
                self._print_progress_bar(current_batch, total_batches, prefix=prefix, suffix=eta, length=bar_len)
            else:
                # print progress as short as possible
                print('{}: {}/{}'.format(stream_name, current_batch, total_batches), end='\r')

            # erase progress bar after last batch
            if total_batches == current_batch:
                self._erase_line()
                self._current_stream_start = time.time()
                self._first_stream_in_epoch = False

        # first epoch
        else:
            progress_msg = '{} {}'.format(prefix, current_batch)
            if len(progress_msg) <= terminal_width:
                print(progress_msg, end='\r')
            else:
                # print progress as short as possible
                print('{}: {}'.format(stream_name, current_batch), end='\r')

    def after_epoch(self, **_) -> None:
        """
        After the first epoch the count of batches is saved, because of displaying the progress bar.
        After every epoch the current batch count is reseted.
        """
        if not self._batch_count_saved:  # save batch counts from the 1st epoch
            self._batch_count_saved = self._batch_count.copy()
        self._reset()
