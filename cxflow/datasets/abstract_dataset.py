"""
This module contains AbstractDataset concept.

At the moment it is for typing only.
"""
from typing import Mapping, Iterable, Any, NewType


class AbstractDataset:
    """
    This concept prescribes the API that is required from every cxflow dataset.
    Every cxflow dataset has to:
        - have a constructor which takes yaml string config
        - have a train_stream method which provides the iterator thought the train stream batches
    Additionally, one may implement any [stream_name]_stream method
    in order to make [stream_name] stream available in the cxflow main_loop.
    """


    Batch = NewType('Batch', Mapping[str, Iterable[Any]])
    Stream = NewType('Stream', Iterable[Batch])

    def __init__(self, config_str: str):
        """
        Create new dataset configured with the given yaml string (obligatory).
        The configuration must contain 'dataset' entry and may contain 'output_dir' entry.
        :param config_str: yaml string config
        """
        pass

    def train_stream(self) -> Stream:  # pylint: disable=undefined-variable
        """Return a train stream iterator (obligatory)."""
        pass

    def split(self, num_splits: int, train: float, valid: float, test: float):
        """Perform cross-validation split."""
        pass
