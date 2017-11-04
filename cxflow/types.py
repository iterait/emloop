from typing import Iterable, Any, Mapping, List


Batch = Mapping[str, Iterable[Any]]
"""Batch type: :py:class:`typing.Mapping` of ``variable_name`` to an :py:class:`typing.Iterable` of examples."""

Stream = Iterable[Batch]
"""Stream type: :py:class:`typing.Iterable` of :py:attr:`Batch` es."""

EpochData = Mapping[str, object]
"""Epoch data type."""

TimeProfile = Mapping[str, List[float]]
"""Time profile type."""
