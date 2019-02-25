"""Module with :py:class:`TrainingTrace` class."""
from .abstract_hook import AbstractHook
from ..utils.yaml import yaml_to_file
from ..constants import EL_TRACE_FILE
from ..types import EpochData

from collections import OrderedDict
from datetime import datetime


class TrainingTraceKeys:
    """Enumeration of training trace keys."""

    TRAIN_BEGIN = 'train_begin'
    """Training begin datetime."""

    TRAIN_END = 'train_end'
    """Training end datetime."""

    EPOCHS_DONE = 'epochs_done'
    """Number of finished training epochs."""

    EXIT_STATUS = 'exit_status'
    """Program exit status."""


class TrainingTrace(AbstractHook):
    """
    Takes care of the "trace.yaml" file in output_dir.
    """
    def __init__(self, output_dir: str, **kwargs):
        super().__init__(**kwargs)

        self._output_dir = output_dir
        self._trace = OrderedDict([(TrainingTraceKeys.TRAIN_BEGIN, None), (TrainingTraceKeys.TRAIN_END, None),
                                   (TrainingTraceKeys.EPOCHS_DONE, 0), (TrainingTraceKeys.EXIT_STATUS, None)])

    def before_training(self) -> None:
        self._trace[TrainingTraceKeys.TRAIN_BEGIN] = datetime.now()
        self.save()

    def after_training(self, success: bool) -> None:
        self._trace[TrainingTraceKeys.TRAIN_END] = datetime.now()
        self._trace[TrainingTraceKeys.EXIT_STATUS] = 1 - int(success)
        self.save()

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        self._trace[TrainingTraceKeys.EPOCHS_DONE] = epoch_id
        self.save()

    def save(self):
         yaml_to_file(self._trace, self._output_dir, EL_TRACE_FILE)
