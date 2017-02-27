from .abstract_hook import AbstractHook

import logging
import typing


class LoggingHook(AbstractHook):
    def __init__(self, metrics_to_display: typing.Iterable[str], **kwargs):
        super().__init__(**kwargs)
        self.metrics_to_display = metrics_to_display

    def before_first_epoch(self, valid_results: dict, test_results: dict = None, ** kwargs) -> None:
        logging.info('Before first epoch')

        for key in self.metrics_to_display:
            if key in valid_results:
                logging.info('\tValid %s:\t%f', key, valid_results[key])
            else:
                logging.error('\tMissing valid variable %s', key)

            if test_results:
                if key in test_results:
                    logging.info('\tTest %s:\t%f', key, test_results[key])
                else:
                    logging.error('\tMissing test variable %s', key)

    def after_epoch(self, epoch_id: int, train_results: dict, valid_results: dict, test_results: dict=None,
                    **kwargs) -> None:
        logging.info('After epoch %d', epoch_id)
        for key in self.metrics_to_display:
            if key in train_results:
                logging.info('\tTrain %s:\t%f', key, train_results[key])
            else:
                logging.error('\tMissing train variable %s', key)

            if key in valid_results:
                logging.info('\tValid %s:\t%f', key, valid_results[key])
            else:
                logging.error('\tMissing valid variable %s', key)

            if test_results:
                if key in test_results:
                    logging.info('\tTest %s:\t%f', key, test_results[key])
                else:
                    logging.error('\tMissing test variable %s', key)
