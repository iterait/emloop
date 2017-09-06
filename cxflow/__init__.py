"""**cxflow** core module."""
from .entry_point import train
from .datasets import AbstractDataset, BaseDataset
from .hooks import AbstractHook
from .main_loop import MainLoop
from .models.abstract_model import AbstractModel
from . import constants

Batch = AbstractDataset.Batch
Stream = AbstractDataset.Stream
EpochData = AbstractHook.EpochData

__all__ = ['MainLoop']

__version__ = '0.8.2'
