"""Module with :py:class:`TrainingTrace` class."""
from enum import Enum
from .yaml import yaml_to_file
from typing import Optional, Any
from collections import OrderedDict

from .config import load_config
from ..constants import CXF_TRACE_FILE


class TrainingTraceKeys(Enum):
    """Enumeration of training trace keys."""

    TRAIN_BEGIN = 'train_begin'
    """Training begin datetime."""

    TRAIN_END = 'train_end'
    """Training end datetime."""

    EPOCHS_DONE = 'epochs_done'
    """Number of finished training epochs."""

    EXIT_STATUS = 'exit_status'
    """Program exit status."""


class TrainingTrace:
    """
    Lightweight wrapper object allowing to modify, save and load training trace variables
    defined in :py:class:`TrainingTraceKeys`.
    """

    _ENUM_VALUES = {}
    """
    Mapping of :py:class:`TrainingTraceKeys` to ``Enum``s used for (de)serialization and checks of ``Enum`` values.
    """

    def __init__(self, output_dir: Optional[str]=None, autosave: bool=True):
        """
        Create new ``TrainingTrace``.

        :param output_dir: training output directory to save the trace to
        :param autosave: auto save the trace after setting an item
        """
        self._output_dir = output_dir
        self._autosave = autosave
        self._trace = OrderedDict([(key.value, None) for key in TrainingTraceKeys])

    def __setitem__(self, key: TrainingTraceKeys, item: Any) -> None:
        """
        Set training trace item to the given value.
        Only :py:class:`TrainingTraceKeys` are allowed as keys.
        """
        assert isinstance(key, TrainingTraceKeys), 'TrainingTrace key must be instance of TrainingTraceKeys enum'
        if key in TrainingTrace._ENUM_VALUES:
            assert isinstance(item, TrainingTrace._ENUM_VALUES[key]), \
                '`{}` item must be instance of `{}`'.format(key, TrainingTrace._ENUM_VALUES[key])
            item = item.value  # map enum item to its value
        self._trace[key.value] = item
        if self._autosave:
            self.save()

    def __getitem__(self, key: TrainingTraceKeys) -> Optional[Any]:
        """
        Get an item for the given key.
        Only :py:class:`TrainingTraceKeys` are allowed as keys.
        """
        assert isinstance(key, TrainingTraceKeys), 'TrainingTrace key must be instance of TrainingTraceKeys enum'
        value = self._trace[key.value]
        if key in TrainingTrace._ENUM_VALUES:
            value = TrainingTrace._ENUM_VALUES[key](value)  # map enum item value back to enum item
        return value

    def save(self) -> None:
        """
        Save the training trace to :py:attr:`CXF_TRACE_FILE` file under the specified directory.

        :raise ValueError: if no output directory was specified
        """
        if self._output_dir is None:
            raise ValueError('Can not save TrainingTrace without output dir.')
        yaml_to_file(self._trace, self._output_dir, CXF_TRACE_FILE)

    @staticmethod
    def from_file(filepath: str):
        """
        Load training trace from the given ``filepath``.

        :param filepath: training trace file path
        :return: training trace
        """
        trace = TrainingTrace()
        trace._trace = load_config(filepath)
        return trace
