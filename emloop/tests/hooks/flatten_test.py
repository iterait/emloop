"""
Test module for flattening variables hook (emloop.hooks.flatten_hook).
"""

import numpy as np
from collections import OrderedDict
import pytest

from emloop.hooks.flatten import Flatten


_ITERS = 5
_STREAM_NAME = 'train'


def get_batch():
    batch = OrderedDict([('1d', 1.11),
                         ('2d', np.arange(10).reshape(2, 5)),
                         ('3d', np.arange(20).reshape(2, 5, 2))])
    return batch


def test_flattening_variables():
    """Test flattening selected variables when hook applied to all available streams."""

    selected_vars = OrderedDict([('1d', '1d_flat'), ('3d', '3d_flat')])
    expected_flat_vars = [np.array([1.11]), np.array(range(20))]
    flatten_vars = Flatten(variables=selected_vars)

    for _ in range(_ITERS):
        batch = get_batch()
        flatten_vars.after_batch(_STREAM_NAME, batch)

    for var_flat, exp_flat in zip(selected_vars.values(), expected_flat_vars):
        assert np.array_equal(batch[var_flat], exp_flat)

    assert '2d_flat' not in batch


def test_flattening_variables_stream_not_in_specified():
    """Test flattening selected variables is not done when stream is not in available streams."""

    selected_vars = OrderedDict([('1d', '1d_flat'), ('2d', '2d_flat'), ('3d', '3d_flat')])
    flatten_vars = Flatten(variables=selected_vars, streams=['test'])

    for _ in range(_ITERS):
        batch = get_batch()
        flatten_vars.after_batch(_STREAM_NAME, batch)

    for var_flat in selected_vars.values():
        assert var_flat not in batch


def test_flattening_variables_raises_error():
    """Test raising an assertion error if variable is not present in a batch."""

    selected_vars = OrderedDict([('not_in_batch', 'not_in_batch_flat'), ('3d', '3d_flat')])
    flatten_vars = Flatten(variables=selected_vars)

    for _ in range(_ITERS):
        batch = get_batch()
        with pytest.raises(KeyError):
            flatten_vars.after_batch(_STREAM_NAME, batch)
