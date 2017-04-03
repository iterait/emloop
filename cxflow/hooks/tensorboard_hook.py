import logging
import typing

import tensorflow as tf

from .abstract_hook import AbstractHook
from ..nets.tf_net import BaseTFNet
from ..datasets.abstract_dataset import AbstractDataset


class TensorBoardHook(AbstractHook):
    """
    Hook that logs summary for of every epoch to tensorboard.

    Note: it is suggested to use `split on underscore` option in tensorboard.
    """

    def __init__(self, net: BaseTFNet, metrics_to_log: typing.Iterable[str], output_dir: str, **kwargs):
        """
        Create new TensorBoard logging hook.
        :param net: a BaseTFNet being trained
        :param metrics_to_log: list of names of the variables to be logged
        :param output_dir: output dir to save the tensorboard logs
        """
        super().__init__(net=net, **kwargs)
        self._metrics_to_log = metrics_to_log

        for metric in metrics_to_log:
            if metric not in net.output_names:
                raise KeyError('Variable `{}` to be monitored in tensorboard is not listed as net output.'.format(metric))

        logging.debug('Creating TensorBoard writer')
        self._summary_writer = tf.summary.FileWriter(logdir=output_dir, graph=net.graph, flush_secs=10)

    def before_first_epoch(self, valid_results: AbstractDataset.Batch, test_results: AbstractDataset.Batch=None,
                           **kwargs) -> None:
        """Log valid (and test) measures from the 0th epoch."""
        logging.debug('TensorBoard logging before first epoch')

        measures = [tf.Summary.Value(tag='valid_{}'.format(key), simple_value=valid_results[key])
                    for key in self._metrics_to_log]

        if test_results:
            measures += [tf.Summary.Value(tag='test_{}'.format(key), simple_value=test_results[key])
                         for key in self._metrics_to_log]

        self._summary_writer.add_summary(tf.Summary(value=measures), 0)

    def after_epoch(self, epoch_id: int, train_results: AbstractDataset.Batch, valid_results: AbstractDataset.Batch,
                    test_results: AbstractDataset.Batch=None, **kwargs) -> None:
        """Train, valid (and test) measures for every epoch."""
        logging.debug('TensorBoard logging after epoch %d', epoch_id)

        measures = [tf.Summary.Value(tag='train_{}'.format(key), simple_value=train_results[key])
                    for key in self._metrics_to_log]

        measures += [tf.Summary.Value(tag='valid_{}'.format(key), simple_value=valid_results[key])
                     for key in self._metrics_to_log]

        if test_results:
            measures += [tf.Summary.Value(tag='test_{}'.format(key), simple_value=test_results[key])
                         for key in self._metrics_to_log]

        self._summary_writer.add_summary(tf.Summary(value=measures), epoch_id)
