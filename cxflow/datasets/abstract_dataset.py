"""
This module contains AbstractDataset concept.

At the moment it is for typing only.
"""

from typing import Mapping, Iterable, Any, NewType


class AbstractDataset:
    """
    This concept prescribes the API that is required from every cxflow dataset.
    Every cxflow dataset has to have a constructor which takes YAML string config.
    Additionally, one may implement any `<stream_name>_stream` method
    in order to make `stream_name` stream available in the cxflow `MainLoop`.
    """

    Batch = Mapping[str, Iterable[Any]]
    Stream = Iterable[Batch]

    def __init__(self, config_str: str):
        """
        Create new dataset configured with the given yaml string (obligatory).
        The configuration must contain 'dataset' entry and may contain 'output_dir' entry.
        :param config_str: YAML string config
        """
        pass
