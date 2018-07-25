"""
Test module for drawing line plots hook (cxflow.hooks.plot_lines_hook).
"""

import numpy as np
import pytest
import os

from cxflow.hooks.plot_lines import PlotLines


_ITERS = 3
_EXAMPLES = 5
_FEATURES = 6
_ROOT_DIR = 'visual'
_EPOCH_ID = '_'
_STREAM_NAME = 'train'


def get_batch():
    batch = {'input': np.ones((_EXAMPLES, _FEATURES)),
             'target': np.zeros(_EXAMPLES),
             'accuracy': np.ones(_EXAMPLES),
             'cost': np.ones(_EXAMPLES),
             'ids': ['a', 'b', 'c', 'd', 'e'],
             'a_ids': ['a', 'b', 'c', 'd', 'e']}
    return batch


def test_plotting_lines_without_optional_arguments(tmpdir):
    """Test plotting variables with minimum specification creates correctly named file."""

    selected_vars = ['input', 'cost']
    suffix = '-vs-'.join(selected_vars)
    plot_lines = PlotLines(output_dir=tmpdir, variables=selected_vars)

    for i in range(_ITERS):
        batch = get_batch()
        plot_lines.after_batch(_STREAM_NAME, batch)

        for id_var in batch['ids']:
            filename = '{}_batch_{}_plot-{}.{}'.format(id_var, i+1, suffix, 'png')
            assert os.path.exists(os.path.join(tmpdir, _ROOT_DIR, 'epoch_{}'.format(_EPOCH_ID), _STREAM_NAME, filename))


def test_plotting_lines_with_optional_arguments(tmpdir):
    """
    Test plotting variables with specified optional arguments and changed default ones creates correctly named files
    only until example_count and batch_count is reached.
    """

    selected_vars = ['input', 'cost']
    id_variable = 'a_ids'
    out_format = 'jpg'
    examples = 2
    batches = 2
    suffix = '-vs-'.join(selected_vars)

    plot_lines = PlotLines(output_dir=tmpdir, variables=selected_vars, streams=['train'], id_variable=id_variable,
                           pad_mask_variable='accuracy', out_format=out_format, ymin=0, ymax=1, example_count=examples,
                           batch_count=batches)

    for _ in range(_ITERS):
        batch = get_batch()
        plot_lines.after_batch(_STREAM_NAME, batch)

    for i in range(batches):
        for e, id_var in zip(range(_EXAMPLES), batch[id_variable]):
            filename = '{}_batch_{}_plot-{}.{}'.format(id_var, i+1, suffix, out_format)
            if e < examples:
                assert os.path.exists(
                    os.path.join(tmpdir, _ROOT_DIR, 'epoch_{}'.format(_EPOCH_ID), _STREAM_NAME, filename))
            else:
                assert not os.path.exists(
                    os.path.join(tmpdir, _ROOT_DIR, 'epoch_{}'.format(_EPOCH_ID), _STREAM_NAME, filename))

    for i_not in range(batches, _ITERS):
        for id_not in batch['ids']:
            filename = '{}_batch_{}_plot-{}.{}'.format(id_not, i_not+1, suffix, out_format)
            assert not os.path.exists(
                os.path.join(tmpdir, _ROOT_DIR, 'epoch_{}'.format(_EPOCH_ID), _STREAM_NAME, filename))


_VARS = [['not-there', 'cost'],
         ['input', 'cost'],
         ['input', 'cost']]
_ID_VARS = ['a_ids', 'not-there', 'a_ids']
_MASK_VARS = ['accuracy', 'accuracy', 'not-there']


@pytest.mark.parametrize('vars, id_var, mask_var', zip(_VARS, _ID_VARS, _MASK_VARS))
def test_plotting_lines_raises_error(vars, id_var, mask_var, tmpdir):
    """Test raising an assertion error if variable is not present in a batch."""

    plot_lines = PlotLines(output_dir=tmpdir, variables=vars, id_variable=id_var, pad_mask_variable=mask_var)
    batch = get_batch()
    with pytest.raises(AssertionError):
        plot_lines.after_batch(_STREAM_NAME, batch)


def test_plotting_lines_stream_not_in_specified(tmpdir):
    """Test plotting variables not done when stream is not in available streams."""

    selected_vars = ['input', 'cost']
    suffix = '-vs-'.join(selected_vars)
    plot_lines = PlotLines(output_dir=tmpdir, variables=selected_vars, streams=['test'])

    for i in range(_ITERS):
        batch = get_batch()
        plot_lines.after_batch(_STREAM_NAME, batch)

        for id_var in batch['ids']:
            filename = '{}_batch_{}_plot-{}.{}'.format(id_var, i+1, suffix, 'png')
            assert not os.path.exists(
                os.path.join(tmpdir, _ROOT_DIR, 'epoch_{}'.format(_EPOCH_ID), _STREAM_NAME, filename))


def test_resetting_batch_count(tmpdir):
    """Test resetting batch_count after epoch."""

    selected_vars = ['input', 'cost']
    plot_lines = PlotLines(output_dir=tmpdir, variables=selected_vars)

    for i in range(_ITERS):
        batch = get_batch()
        plot_lines.after_batch(_STREAM_NAME, batch)

    plot_lines.after_epoch(0)
    assert not plot_lines._batch_done
    assert plot_lines._current_epoch_id == 1
