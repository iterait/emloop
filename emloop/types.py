from typing import Iterable, Any, Mapping, List, Sequence


Batch = Mapping[str, Sequence[Any]]
"""Batch type: :py:class:`typing.Mapping` of ``variable_name`` to an :py:class:`typing.Iterable` of examples."""

Stream = Iterable[Batch]
"""Stream type: :py:class:`typing.Iterable` of :py:attr:`Batch` es."""

EpochData = Mapping[str, object]
"""Epoch data type."""

TimeProfile = Mapping[str, List[float]]
"""Time profile type."""

class TrainingTerminated(Exception):
    """Exception that is raised when a hook terminates the training."""
    pass
