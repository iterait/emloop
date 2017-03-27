from .abstract_hook import AbstractHook
from ..datasets.abstract_dataset import AbstractDataset

import logging
import typing
import sys


class LoggingHook(AbstractHook):
    """Log the training results to stderr via standard logging module."""

    def __init__(self, metrics_to_display: typing.Iterable[str], **kwargs):
        """
        :param net: trained network
        :param metrics_to_display: list of names of statistics that will be logged
        """

        super().__init__(**kwargs)
        self._metrics_to_display = metrics_to_display

    def before_first_epoch(self, valid_results: dict, test_results: dict = None, ** kwargs) -> None:

        print('\n\n', file=sys.stderr)
        logging.info(' Before first epoch')

        for key in self._metrics_to_display:
            if key in valid_results:
                logging.info('\tValid %s:\t%f', key, valid_results[key])
            else:
                logging.error('\tMissing valid variable %s', key)

            if test_results:
                if key in test_results:
                    logging.info('\tTest %s:\t%f', key, test_results[key])
                else:
                    logging.error('\tMissing test variable %s', key)

    def after_epoch(self, epoch_id: int, train_results: AbstractDataset.Batch, valid_results: AbstractDataset.Batch,
                    test_results: AbstractDataset.Batch=None, **kwargs) -> None:

        print('\n\n', file=sys.stderr)
        logging.info('After epoch {}'.format(epoch_id))

        for key in self._metrics_to_display:
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
