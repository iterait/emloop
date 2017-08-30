"""
This module contains AbstractDataset concept.

At the moment it is for typing only.
"""

from typing import Mapping, Iterable, Any, NewType


class AbstractDataset:
    """
    This concept prescribes the API that is required from every **cxflow** dataset.

    Every **cxflow** dataset has to have a constructor which takes YAML string config.
    Additionally, one may implement any ``<stream_name>_stream`` method
    in order to make ``stream_name`` stream available in the **cxflow** :py:class:`cxflow.MainLoop`.

    All the defined stream methods should return a :py:attr:`Stream`.
    """

    Batch = Mapping[str, Iterable[Any]]
    """Batch type: :py:class:`typing.Mapping` of ``variable_name`` to an :py:class:`typing.Iterable` of examples."""

    Stream = Iterable[Batch]
    """Stream type: :py:class:`typing.Iterable` of :py:attr:`Batch` es."""

    def __init__(self, config_str: str):
        """
        Create new dataset configured with the given YAML string (obligatory).

        The configuration must contain ``dataset`` entry and may contain ``output_dir`` entry.

        :param config_str: YAML string config
        """
        pass
