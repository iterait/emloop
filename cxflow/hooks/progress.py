"""
Module with ShowProgress hook which shows progress of the current epoch.
"""
import logging
import shutil
import collections
import time

from . import AbstractHook
from ..datasets import AbstractDataset


class ShowProgress(AbstractHook):
    """
    Show stream progresses and ETA in the current epoch.

    .. tip::
        If the dataset provides ``num_batches`` property, the hook will be able to display the progress and ETA for the
        1st epoch as well. The property should return a mapping of ``<stream name>`` -> ``<batch count>``.

    .. code-block:: yaml
        :caption: show progress of the current epoch

        hooks:
          - ShowProgress
    """

    def __init__(self, dataset: AbstractDataset, **kwargs):
        """Create new ShowProgress hook."""
        super().__init__(**kwargs)
        self._total_batch_count_saved = False
        if hasattr(dataset, 'num_batches'):
            logging.debug('Capturing batch counts from dataset')
            self._total_batch_count = dataset.num_batches
        else:
            self._total_batch_count = {}
        self._current_batch_count = collections.defaultdict(lambda: 0)
        self._first_batch_in_epoch = True
        self._current_stream_start = None
        self._first_stream_in_epoch = True

    def _reset(self) -> None:
        """
        Set ``_batch_count``, ``_first_batch_in_epoch``, ``_current_stream_start``
        and ``_first_stream_in_epoch`` to initial values.
        """
        self._current_batch_count.clear()
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
        Display the progress and ETA for the current stream in the epoch.
        If the stream size (total batch count) is unknown (1st epoch), print only the number of processed batches.
        """

        self._current_batch_count[stream_name] += 1

        if not self._first_batch_in_epoch:
            self._erase_line()
        else:
            self._first_batch_in_epoch = False

        current_batch = self._current_batch_count[stream_name]
        prefix = 'Progress of {} stream:'.format(stream_name)

        terminal_width = shutil.get_terminal_size().columns

        # total batch count is available
        if stream_name in self._total_batch_count:

            total_batches = self._total_batch_count[stream_name]

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

        # total batch count is not available (1st epoch)
        else:
            progress_msg = '{} {}'.format(prefix, current_batch)
            if len(progress_msg) <= terminal_width:
                print(progress_msg, end='\r')
            else:
                # print progress as short as possible
                print('{}: {}'.format(stream_name, current_batch), end='\r')

    def after_epoch(self, **_) -> None:
        """
        Reset progress counters. Save ``total_batch_count`` after the 1st epoch.
        """
        if not self._total_batch_count_saved:
            self._total_batch_count = self._current_batch_count.copy()
            self._total_batch_count_saved = True
        self._reset()
