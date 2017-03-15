from .abstract_hook import AbstractHook
from ..nets.abstract_net import AbstractNet

import logging
from os import path
import typing


class ResultHook(AbstractHook):
    """Save model outputs whenever it outperforms itself."""

    def __init__(self, net: AbstractNet, metric: str, condition: str, metrics_to_log: typing.List[str],
                 output_prefix: str='result', **kwargs):
        """
        Example: metric=loss, condition=min -> saved the model when the loss is best so far.
        :param net: trained network
        :param metric: metric to be evaluated (usually loss)
        :param condition: {min,max}
        :param metrics_to_log: list of names of metrics to be be logged
        :param output_prefix: prefix of the dumped file
        """
        super().__init__(net=net, **kwargs)
        self._net = net
        self._metric = metric
        self._condition = condition
        self._metrics_to_log = metrics_to_log

        self._valid_f = path.join(self._net.log_dir, output_prefix + '_valid.csv')
        self._test_f = path.join(self._net.log_dir, output_prefix + '_test.csv')

        self._best_metric = None

        logging.info('Results will be saved to "%s" and "%s"', self._valid_f, self._test_f)
        self._reset()

    def _reset(self):
        """Reset all buffers."""

        self._valid_buffer = []
        self._test_buffer = []

    def _save_partial_results(self, buffer: list, results: dict):
        """Append result to a buffer"""

        for i in range(len(results[self._metrics_to_log[0]])):
            buffer.append([results[metric][i] for metric in self._metrics_to_log])

    def _save_results(self, buffer: list, f_name: str):
        with open(f_name, 'w') as f:
            header = ','.join(['"{}"'.format(metric) for metric in self._metrics_to_log]) + '\n'
            f.write(header)

            for result_row in buffer:
                row = ','.join(map(str, result_row))
                f.write(row + '\n')

    def after_batch(self, stream_type: str, results: dict, **kwargs) -> None:
        """Save metrics of this batch."""

        if stream_type == 'train':
            pass
        elif stream_type == 'valid':
            self._save_partial_results(self._valid_buffer, results)
        elif stream_type == 'test':
            self._save_partial_results(self._test_buffer, results)
        else:
            raise ValueError('stream_type must be either train, valid or test. Instead, `%s` was'
                             'provided'.format(stream_type))

    def before_first_epoch(self, valid_results: dict, **kwargs) -> None:
        self._best_metric = valid_results[self._metric]
        self._save_results(self._valid_buffer, self._valid_f)
        self._save_results(self._test_buffer, self._test_f)

    def after_epoch(self, valid_results: dict, **kwargs) -> None:
        if self._condition == 'min':
            if self._best_metric is None or valid_results[self._metric] < self._best_metric:
                logging.info('Saving results')
                self._best_metric = valid_results[self._metric]
                self._save_results(self._valid_buffer, self._valid_f)
                self._save_results(self._test_buffer, self._test_f)
        elif self._condition == 'max':
            if self._best_metric is None or valid_results[self._metric] > self._best_metric:
                logging.info('Saving results')
                self._best_metric = valid_results[self._metric]
                self._save_results(self._valid_buffer, self._valid_f)
                self._save_results(self._test_buffer, self._test_f)
        else:
            logging.error('BestSaverHook support only {min,max} as a condition')
            raise ValueError('BestSaverHook support only {min,max} as a condition')
        self._reset()
