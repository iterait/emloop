"""
Module with ShowProgress hook which shows progress of the current epoch.
"""
import logging
import shutil
import collections
import time

from . import AbstractHook
from ..datasets import AbstractDataset
from ..types import Batch


def print_progress_bar(done: int, total: int, prefix: str = '', suffix: str = '') -> None:
    """
    Print a progressbar with the given prefix and suffix, without newline at the end.

    param done: current step in computation
    param total: total count of steps in computation
    param prefix: info text displayed before the progress bar
    param suffix: info text displayed after the progress bar
    """

    percent = '{0:.1f}'.format(100 * (done / float(total)))
    base_len = shutil.get_terminal_size().columns - 7 - len(prefix) - len(suffix)
    base_len = min([base_len, 50])
    min_length = base_len - 1 - len('{}/{}={}'.format(total, total, '100.0'))
    length = base_len - len('{}/{}={}'.format(done, total, percent))
    if min_length > 0:
        filled_len = int(min_length * done // total)
        bar = '='*filled_len + '-'*(min_length - filled_len)
        spacing = ' '*(length - min_length)
        print('\r{}: |{}|{}{}/{}={}% {}'.format(prefix, bar, spacing, done, total, percent, suffix), end='\r')
    else:
        short_progress = '\r{}: {}/{}'.format(prefix, done, total)
        if len(short_progress) <= shutil.get_terminal_size().columns:
            print(short_progress, end='\r')
        else:
            print(['-', '\\', '|', '/'][done % 4], end='\r')


def erase_line() -> None:
    """Erase the current line."""
    print('\x1b[2K', end='\r')


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


class ShowProgress(AbstractHook):
    """
    Show stream progresses and ETA in the current epoch.

    .. tip::
        If the dataset provides ``num_batches`` property, the hook will be able to display the progress and ETA for the
        1st epoch as well. The property should return a mapping of ``<stream name>`` -> ``<batch count>``.

    .. caution::
        ``ShowProgress`` hook should be placed as the first in hooks config section, otherwise 
        the progress bar may not be displayed correctly.

    .. code-block:: yaml
        :caption: show progress of the current epoch

        hooks:
          - ShowProgress
    """

    def __init__(self, dataset: AbstractDataset, **kwargs):
        """
        Create new ShowProgress hook.

        Fetch the batch counts from ``dataset.num_batches`` property if available.

        :param dataset: training dataset
        """
        super().__init__(**kwargs)
        self._total_batch_count_saved = False
        if hasattr(dataset, 'num_batches'):
            logging.debug('Capturing batch counts from dataset')
            self._total_batch_count = dataset.num_batches
        else:
            self._total_batch_count = {}
        self._current_batch_count = collections.defaultdict(lambda: 0)
        self._current_stream_start = None
        self._current_stream_name = None

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        """
        Display the progress and ETA for the current stream in the epoch.
        If the stream size (total batch count) is unknown (1st epoch), print only the number of processed batches.
        """
        if self._current_stream_name is None or self._current_stream_name != stream_name:
            self._current_stream_name = stream_name
            self._current_stream_start = None
        erase_line()

        self._current_batch_count[stream_name] += 1
        current_batch = self._current_batch_count[stream_name]

        # total batch count is available
        if stream_name in self._total_batch_count:
            # compute ETA
            total_batches = self._total_batch_count[stream_name]
            if self._current_stream_start:
                measured_batches = current_batch - 1
                avg_batch_time = (time.time() - self._current_stream_start) / measured_batches
                eta_sec = avg_batch_time * (total_batches - current_batch)
                eta = get_formatted_time(eta_sec)
            else:
                self._current_stream_start = time.time()
                eta = ''

            print_progress_bar(current_batch, total_batches, prefix=stream_name, suffix=eta)

        # total batch count is not available (1st epoch)
        else:
            short_progress = '{}: {}'.format(stream_name, current_batch)
            if len(short_progress) <= shutil.get_terminal_size().columns:
                print(short_progress, end='\r')
            else:
                print(['-', '\\', '|', '/'][current_batch % 4],  end='\r')

    def after_epoch(self, **_) -> None:
        """
        Reset progress counters. Save ``total_batch_count`` after the 1st epoch.
        """
        if not self._total_batch_count_saved:
            self._total_batch_count = self._current_batch_count.copy()
            self._total_batch_count_saved = True
        self._current_batch_count.clear()
        self._current_stream_start = None
        self._current_stream_name = None
        erase_line()
