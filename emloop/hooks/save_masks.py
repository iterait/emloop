"""
Hook for saving masks to the filesystem.
"""

import os
import logging

import numpy as np

from . import AbstractHook
from ..types import Batch

try:
    import cv2
except ImportError:
    logging.info('This hook requires OpenCV.')


class SaveMasks(AbstractHook):
    """
    Save a stream of masks.
    

    .. code-block:: yaml
        :caption: Save masks from variable `predictions` to paths from variable
                  `image_paths` suffixed by `_mask.png`.

        hooks:
          - SaveMasks:
              mask_variable: predictions
              path_variable: image_paths
    """

    def __init__(self, mask_variable: str, path_variable: str, factor: float=255.,
                 output_root: str='', suffix: str='_mask.png', **kwargs):
        """
        :param mask_variable: the variable with the mask image
        :param path_variable: the variable with the image path
        :param factor: the factor by which is the mask multiplied
        :param output_root: the data root where should the masks be saved;
                            this directory is basically a prefix for `path_variable`
        :param suffix: the suffix to be added to mask filename
        """
        super().__init__(**kwargs)

        self._mask_variable = mask_variable
        self._path_variable = path_variable
        self._factor = factor
        self._output_root = output_root
        self._suffix = suffix

    def save_mask(self, mask: np.ndarray, path: str) -> None:
        """
        Save the given mask to a file.
        """
        mask_path = path + self._suffix
        logging.info('Saving mask to %s.', mask_path)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, mask * self._factor)

    def after_batch(self, stream_name: str, batch_data: Batch):
        """
        Save the masks for each example.
        """
        assert self._mask_variable in batch_data
        assert self._path_variable in batch_data
        assert len(batch_data[self._mask_variable]) == len(batch_data[self._path_variable])
        for i, mask in enumerate(batch_data[self._mask_variable]):
            path = batch_data[self._path_variable][i]
            self.save_mask(mask, os.path.join(self._output_root, path))
