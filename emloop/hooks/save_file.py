"""
Module with simple hook saving files to the output dir.
"""
import shutil
from typing import Iterable
import os

from . import AbstractHook


class SaveFile(AbstractHook):
    """
    Save files to the output dir before training.

    .. code-block:: yaml
        :caption: save files to output dir

        hooks:
          - SaveFile
              files: [path]
    """

    def __init__(self, files: Iterable[str], output_dir: str, **kwargs):
        """
        :param files: files to be saved
        :param output_dir: directory to save the files
        """
        super().__init__(**kwargs)

        self._files = files
        self._output_dir = output_dir

    def before_training(self):
        for path in self._files:
            assert os.path.exists(path), f'file `{path}` does not exist'
            shutil.copy(path, self._output_dir)
