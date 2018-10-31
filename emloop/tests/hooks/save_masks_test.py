"""
Test module for saving masks to the filesystem hook (cxflow.hooks.save_masks_hook).
"""

import numpy as np
import pytest
import os

from cxflow.hooks.save_masks import SaveMasks


_ITERS = 2
_STREAM_NAME = 'train'


def get_batch():
    batch = {'mask': np.ones((5, 3)),
             'path': ['a', 'b', 'c', 'd', 'e'],
             'incorrect': np.arange(18).reshape(6, 3)}
    return batch


def test_saving_mask(tmpdir):
    """Test masks are correctly saved to file."""

    mask = 'mask'
    path = 'path'
    suffix = '_is_mask.png'
    save_masks = SaveMasks(mask_variable=mask, path_variable=path, output_root=str(tmpdir), suffix=suffix)

    for i in range(_ITERS):
        batch = get_batch()
        save_masks.after_batch(_STREAM_NAME, batch)
        for _, j in zip(batch[mask], batch[path]):
            filename = j + suffix
            filename_path = os.path.join(tmpdir, filename)
            assert os.path.exists(filename_path)


_MASK = ['not-there', 'mask', 'incorrect', 'mask']
_PATH = ['path', 'not-there', 'path', 'incorrect']


@pytest.mark.parametrize('mask, path', zip(_MASK, _PATH))
def test_saving_mask_raises_error(tmpdir, mask, path):
    """Test raising an assertion error if variable is not present in a batch/variable lengths are not same."""

    save_masks = SaveMasks(mask_variable=mask, path_variable=path, output_root=str(tmpdir))

    batch = get_batch()

    with pytest.raises(AssertionError):
        save_masks.after_batch(_STREAM_NAME, batch)
