from .abstract_hook import AbstractHook
from ..nets.abstract_net import AbstractNet
from ..datasets.abstract_dataset import AbstractDataset

import tensorflow as tf

import logging
import typing


class TensorBoardHook(AbstractHook):
    """Log the training to TensorBoard"""

    def __init__(self, net: AbstractNet, metrics_to_log: typing.Iterable[str], **kwargs):
        """
        :param net: trained net
        :param metrics_to_log: list of names of the variables to be logged
        """
        super().__init__(net=net, **kwargs)
        self._net = net
        self._metrics_to_log = metrics_to_log

    def before_first_epoch(self, valid_results: AbstractDataset.Batch, test_results: AbstractDataset.Batch=None,
                           **kwargs) -> None:
        logging.debug('TensorBoard logging before first epoch')

        measures = [tf.Summary.Value(tag='valid_{}'.format(key), simple_value=valid_results[key])
                    for key in self._metrics_to_log]

        if test_results:
            measures += [tf.Summary.Value(tag='test_{}'.format(key), simple_value=test_results[key])
                         for key in self._metrics_to_log]

        self._net.summary_writer.add_summary(tf.Summary(value=measures), 0)

    def after_epoch(self, epoch_id: int, train_results: AbstractDataset.Batch, valid_results: AbstractDataset.Batch,
                    test_results: AbstractDataset.Batch=None, **kwargs) -> None:
        logging.debug('TensorBoard logging after epoch %d', epoch_id)

        measures = [tf.Summary.Value(tag='train_{}'.format(key), simple_value=train_results[key])
                    for key in self._metrics_to_log]

        measures += [tf.Summary.Value(tag='valid_{}'.format(key), simple_value=valid_results[key])
                     for key in self._metrics_to_log]

        if test_results:
            measures += [tf.Summary.Value(tag='test_{}'.format(key), simple_value=test_results[key])
                         for key in self._metrics_to_log]

        self._net.summary_writer.add_summary(tf.Summary(value=measures), epoch_id)
