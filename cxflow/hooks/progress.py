from . import AbstractHook
from ..datasets import AbstractDataset

_ERASE_LINE = '\x1b[2K'


class Progress(AbstractHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._batch_count = {}
        self._first_batch_in_epoch = True
        self._first_epoch = True
        self._batch_count_saved = None

    def after_batch(self, stream_name: str, batch_data: AbstractDataset.Batch):

        if stream_name not in self._batch_count:
            self._batch_count[stream_name] = 0
        else:
            self._batch_count[stream_name] += 1

        if not self._first_batch_in_epoch:
            print(_ERASE_LINE, end='\r')
        else:
            self._first_batch_in_epoch = False

        current_batch = self._batch_count[stream_name] + 1

        if self._batch_count_saved:
            total_batches = self._batch_count_saved[stream_name] + 1
            self._print_progress_bar(
                current_batch,
                total_batches,
                prefix='Progress of {} stream:'.format(stream_name), suffix='complete')
        else:
            print("Progress of {} stream: {} batch complete".format(stream_name,
                                                                    current_batch), end='\r')

    def _reset(self):
        self._batch_count = {}
        self._first_batch = True

    def after_epoch(self, **_):
        if self._first_epoch:
            self._batch_count_saved = self._batch_count.copy()
            self._first_epoch = False
        self._reset()

    def _print_progress_bar(self, iteration, total, prefix='', suffix='', length=50, fill='█'):
        percent = ("{0:.2f}").format(100 * (iteration / float(total)))
        filled_len = int(length * iteration // total)
        bar = fill * filled_len + '-' * (length - filled_len)
        print('\r%s |%s| %s/%s %s%% %s' % (prefix, bar, iteration, total, percent, suffix), end='\r')
