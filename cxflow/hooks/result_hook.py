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
        self.net = net
        self.metric = metric
        self.condition = condition
        self.metrics_to_log = metrics_to_log

        self.valid_f = path.join(self.net.log_dir, output_prefix+'_valid.csv')
        self.test_f = path.join(self.net.log_dir, output_prefix+'_test.csv')

        self.best_metric = None

        logging.info('Results will be saved to "%s" and "%s"', self.valid_f, self.test_f)
        self._reset()

    def _reset(self):
        """Reset all buffers."""

        self.valid_buffer = []
        self.test_buffer = []

    def _save_partial_results(self, buffer: list, results: dict):
        """Append result to a buffer"""

        for i in range(len(results[self.metrics_to_log[0]])):
            buffer.append([results[metric][i] for metric in self.metrics_to_log])

    def _save_results(self, buffer: list, f_name: str):
        with open(f_name, 'w') as f:
            header = ','.join(['"{}"'.format(metric) for metric in self.metrics_to_log]) + '\n'
            f.write(header)

            for result_row in buffer:
                row = ','.join(map(str, result_row))
                f.write(row + '\n')

    def after_batch(self, stream_type: str, results: dict, **kwargs) -> None:
        """Save metrics of this batch."""

        if stream_type == 'train':
            pass
        elif stream_type == 'valid':
            self._save_partial_results(self.valid_buffer, results)
        elif stream_type == 'test':
            self._save_partial_results(self.test_buffer, results)
        else:
            raise ValueError('stream_type must be either train, valid or test. Instead, `%s` was'
                             'provided'.format(stream_type))

    def before_first_epoch(self, valid_results: dict, **kwargs) -> None:
        self.best_metric = valid_results[self.metric]
        self._save_results(self.valid_buffer, self.valid_f)
        self._save_results(self.test_buffer, self.test_f)

    def after_epoch(self, valid_results: dict, **kwargs) -> None:
        if self.condition == 'min':
            if valid_results[self.metric] < self.best_metric:
                logging.info('Saving results')
                self.best_metric = valid_results[self.metric]
                self._save_results(self.valid_buffer, self.valid_f)
                self._save_results(self.test_buffer, self.test_f)
        elif self.condition == 'max':
            if valid_results[self.metric] > self.best_metric:
                logging.info('Saving results')
                self.best_metric = valid_results[self.metric]
                self._save_results(self.valid_buffer, self.valid_f)
                self._save_results(self.test_buffer, self.test_f)
        else:
            logging.error('BestSaverHook support only {min,max} as a condition')
            raise ValueError('BestSaverHook support only {min,max} as a condition')
