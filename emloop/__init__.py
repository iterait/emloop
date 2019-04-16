"""**emloop** core module."""
from .entry_point import train
from .datasets import AbstractDataset, BaseDataset, DownloadableDataset
from .hooks import AbstractHook
from .main_loop import MainLoop
from .models import AbstractModel
from .types import Batch, Stream, EpochData, TimeProfile

from .api import *
from . import cli
from . import constants
from . import datasets
from . import hooks
from . import models
from . import utils

__all__ = ['MainLoop', 'create_output_dir', 'create_dataset', 'create_model', 'create_hooks', 'create_main_loop',
           'load_yaml', 'AbstractDataset', 'BaseDataset', 'DownloadableDataset', 'AbstractHook', 'MainLoop',
           'AbstractModel', 'Batch', 'Stream', 'EpochData', 'TimeProfile']

__version__ = '0.2.1'
