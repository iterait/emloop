"""
This module contains AbstractDataset concept.

At the moment it is for typing only.
"""
from typing import Mapping, Iterable, Any, NewType


class AbstractDataset:
    """A dataset.
    Dataset classes implement the interface to a particular dataset. The
    interface consists of a number of routines to manipulate so called
    "state" objects, e.g. open, reset and close them.
    Parameters
    ----------
    sources : tuple of strings, optional
        The data sources to load and return by :meth:`get_data`. By default
        all data sources are returned.
    axis_labels : dict, optional
        Maps source names to tuples of strings describing axis semantics,
        one per axis. Defaults to `None`, i.e. no information is available.
    Attributes
    ----------
    sources : tuple of strings
        The sources this dataset will provide when queried for data e.g.
        ``('features',)`` when querying only the data from MNIST.
    provides_sources : tuple of strings
        The sources this dataset *is able to* provide e.g. ``('features',
        'targets')`` for MNIST (regardless of which data the data stream
        actually requests). Any implementation of a dataset should set this
        attribute on the class (or at least before calling ``super``).
    example_iteration_scheme : :class:`.IterationScheme` or ``None``
        The iteration scheme the class uses in order to produce a stream of
        examples.
    default_transformers: It is expected to be a tuple with one element per
        transformer in the pipeline. Each element is a tuple with three
        elements:
            - the Transformer subclass to apply,
            - a list of arguments to pass to the subclass constructor, and
            - a dict of keyword arguments to pass to the subclass
              constructor.
    Notes
    -----
    Datasets should only implement the interface; they are not expected to
    perform the iteration over the actual data. As such, they are
    stateless, and can be shared by different parts of the library
    simultaneously.
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
