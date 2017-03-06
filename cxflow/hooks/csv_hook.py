from .abstract_hook import AbstractHook
from ..nets.abstract_net import AbstractNet

import logging
from os import path
import typing


class CSVHook(AbstractHook):
    def __init__(self, net: AbstractNet, metrics_to_display: typing.Iterable[str],
                 output_file: str="training.csv", **kwargs):
        super().__init__(net=net, **kwargs)
        self.metrics_to_display = metrics_to_display
        self.file = path.join(net.log_dir, output_file)

        logging.info('Saving training log to "%s"', self.file)

        with open(self.file, 'w') as f:
            header = '"epoch_id",' +\
                     ','.join(['"train_{0}","valid_{0}","test_{0}"'.format(metric) for metric in metrics_to_display]) +\
                     '\n'
            f.write(header)

    def before_first_epoch(self, valid_results: dict, test_results: dict = None, ** kwargs) -> None:
        logging.info('Before first epoch')

        result_row = [0]
        for key in self.metrics_to_display:
            result_row.append('')
            if key in valid_results:
                result_row.append(valid_results[key])
            else:
                logging.error('\tMissing valid variable %s', key)
                result_row.append('')

            if test_results:
                if key in test_results:
                    result_row.append(test_results[key])
                else:
                    logging.error('\tMissing test variable %s', key)
                    result_row.append('')
            else:
                result_row.append('')

        with open(self.file, 'a') as f:
            row = ','.join(map(str, result_row))
            f.write(row + '\n')

    def after_epoch(self, epoch_id: int, train_results: dict, valid_results: dict, test_results: dict=None,
                    **kwargs) -> None:

        logging.info('After epoch %d', epoch_id)
        result_row = [epoch_id]
        for key in self.metrics_to_display:
            if key in train_results:
                result_row.append(train_results[key])
            else:
                logging.error('\tMissing train variable %s', key)
                result_row.append('')

            if key in valid_results:
                result_row.append(valid_results[key])
            else:
                logging.error('\tMissing valid variable %s', key)
                result_row.append('')

            if test_results:
                if key in test_results:
                    result_row.append(test_results[key])
                else:
                    logging.error('\tMissing test variable %s', key)
                    result_row.append('')
            else:
                result_row.append('')

        with open(self.file, 'a') as f:
            row = ','.join(map(str, result_row))
            f.write(row + '\n')
