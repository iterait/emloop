"""
Module with a tensorboard logging hook.
"""
import logging

import numpy as np
import tensorflow as tf

from .abstract_hook import AbstractHook
from ..nets.tf_net import BaseTFNet


class TensorBoardHook(AbstractHook):
    """
    Log epoch summaries to TensorBoard.

    -------------------------------------------------------
    Example usage in config
    -------------------------------------------------------
    hooks:
      - class: TensorBoardHook
    -------------------------------------------------------
    # unknown variable types are casted to strings
    hooks:
      - class: TensorBoardHook
        on_unknown_type: str
    -------------------------------------------------------
    """

    UNKNOWN_TYPE_ACTIONS = {'error', 'warn', 'ignore'}

    def __init__(self, net: BaseTFNet, output_dir: str, flush_secs: int=10, on_unknown_type: str='ignore', **kwargs):
        """
        Create new TensorBoard logging hook.

        :param net: a BaseTFNet being trained
        :param output_dir: output dir to save the tensorboard logs
        :param on_unknown_type: an action to be taken if the variable value type is not supported (e.g. a list)
        """
        assert isinstance(net, BaseTFNet)

        super().__init__(net=net, output_dir=output_dir, **kwargs)
        self._on_unknown_type = on_unknown_type

        logging.debug('Creating TensorBoard writer')
        self._summary_writer = tf.summary.FileWriter(logdir=output_dir, graph=net.graph, flush_secs=flush_secs)

    def _log_to_tensorboard(self, epoch_id: int, epoch_data: AbstractHook.EpochData):
        """
        Log the metrics from the result given to the tensorboard.

        :param epoch_id: epoch number
        :param epoch_data: the epoch data to be logged
        """

        measures = []

        for stream_name in epoch_data.keys():
            stream_data = epoch_data[stream_name]
            for variable in stream_data.keys():
                value = stream_data[variable]
                if np.isscalar(value):  # try logging the scalar values
                    result = value
                elif isinstance(value, dict) and 'mean' in value:  # or the mean
                    result = value['mean']
                else:
                    err_message = 'Variable `{}` in stream `{}` is not scalar and does not contain `mean` aggregation'\
                                  .format(variable, stream_name)
                    if self._on_unknown_type == 'warn':
                        logging.warning(err_message)
                        result = str(value)
                    elif self._on_unknown_type == 'error':
                        raise ValueError(err_message)
                    else:
                        continue

                measures.append(tf.Summary.Value(tag='{}_{}'.format(stream_name, variable), simple_value=result))

        self._summary_writer.add_summary(tf.Summary(value=measures), epoch_id)

    def after_epoch(self, epoch_id: int, epoch_data: AbstractHook.EpochData) -> None:
        """Log the specified epoch data variables to the tensorboar."""
        logging.debug('TensorBoard logging after epoch %d', epoch_id)
        self._log_to_tensorboard(epoch_id=epoch_id, epoch_data=epoch_data)
