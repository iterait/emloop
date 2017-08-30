"""Module with **cxflow** dataset concept (:py:class:`AbstractDataset`) and python :py:class:`BaseDataset`."""

from .abstract_dataset import AbstractDataset
from .base_dataset import BaseDataset

AbstractDataset.__module__ = '.datasets'
BaseDataset.__module__ = '.datasets'

__all__ = ['AbstractDataset', 'BaseDataset']
