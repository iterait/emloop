from .abstract_hook import AbstractHook
from ..nets.abstract_net import AbstractNet

from collections import defaultdict
import typing


class ResultHook(AbstractHook):
    """Save model outputs on each epoch"""

    def __init__(self, net: AbstractNet, metrics_to_log: typing.List[str], log_train_results: bool=False, **kwargs):
        """
        :param net: trained network
        :param log_train_results: should also train results be saved?
        :param metrics_to_log: list of names of metrics to be be logged
        """
        super().__init__(net=net, **kwargs)
        self._net = net
        self._log_train_results = log_train_results
        self._metrics_to_log = metrics_to_log

        self._reset()

    def _reset(self):
        """Reset all buffers."""

        self._train_buffer = defaultdict(list)
        self._valid_buffer = defaultdict(list)
        self._test_buffer = defaultdict(list)

    def _save_partial_results(self, buffer: dict, results: dict):
        """Append result to a buffer"""

        for metric in self._metrics_to_log:
            buffer[metric] += results[metric].tolist()

    def after_batch(self, stream_type: str, results: dict, **kwargs) -> None:
        """Save metrics of this batch."""

        if stream_type == 'train':
            self._save_partial_results(self._train_buffer, results)
        elif stream_type == 'valid':
            self._save_partial_results(self._valid_buffer, results)
        elif stream_type == 'test':
            self._save_partial_results(self._test_buffer, results)
        else:
            raise ValueError('stream_type must be either train, valid or test. Instead, `%s` was'
                             'provided'.format(stream_type))

    def before_first_epoch(self, valid_results: dict, test_results: dict=None, **kwargs) -> None:
        valid_results['results'] = dict(self._valid_buffer)
        test_results['results'] = dict(self._test_buffer)
        self._reset()

    def after_epoch(self, train_results: dict, valid_results: dict, test_results: dict=None, **kwargs) -> None:
        if self._log_train_results:
            train_results['results'] = dict(self._train_buffer)
        valid_results['results'] = dict(self._valid_buffer)
        test_results['results'] = dict(self._test_buffer)
        self._reset()
