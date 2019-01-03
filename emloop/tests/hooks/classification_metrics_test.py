"""
Test module for computing epoch statistics for classification tasks hook (emloop.hooks.classification_metrics).
"""
import collections
import os

import numpy as np
import pytest

from emloop.hooks.classification_metrics import ClassificationMetrics


def get_epoch_data():
    epoch_data = collections.OrderedDict([
        ('train', collections.OrderedDict([
            ('gt', np.array([0, 0, 0, 0, 1])),
            ('prediction', np.array([0, 0, 0, 1, 1])),
            ('accuracy', 1),
        ])),
        ('test', collections.OrderedDict([
            ('gt', np.array([1, 1, 1, 1, 0])),
            ('prediction', np.array([1, 1, 1, 0, 0])),
        ]))
    ])
    return epoch_data


@pytest.mark.skipif(os.environ.get('EXTRA_PKGS', None) is None, reason='This test requires SciKit.')
def test_computing_metrics():
    """Test metrics is correctly computed."""

    prefix = 'metrics_'
    hook = ClassificationMetrics('prediction', 'gt', 'binary', prefix)
    epoch_data = get_epoch_data()

    hook.after_batch(stream_name='train', batch_data=epoch_data['train'])
    hook.after_batch(stream_name='test', batch_data=epoch_data['test'])
    hook.after_epoch(epoch_data)

    assert epoch_data['train'][prefix+'precision'] == 0.5
    assert epoch_data['train'][prefix+'recall'] == 1.0
    assert epoch_data['train'][prefix+'f1'] == 2 / (1/0.5 + 1/1)
    assert epoch_data['train'][prefix+'accuracy'] == 0.8
    assert epoch_data['train'][prefix+'specificity'] == 0.75

    assert epoch_data['test'][prefix+'precision'] == 1.0
    assert epoch_data['test'][prefix+'recall'] == 0.75
    assert epoch_data['test'][prefix+'f1'] == pytest.approx(2 / (1/1 + 1/0.75))
    assert epoch_data['test'][prefix+'accuracy'] == 0.8
    assert epoch_data['test'][prefix+'specificity'] == 1.0


@pytest.mark.skipif(os.environ.get('EXTRA_PKGS', None) is None, reason='This test requires SciKit.')
def test_computing_metrics_raises_error():
    """Test raises value error if variables should be repeated."""

    hook = ClassificationMetrics('prediction', 'gt')
    epoch_data = get_epoch_data()

    hook.after_batch(stream_name='train', batch_data=epoch_data['train'])
    with pytest.raises(ValueError):
        hook.after_epoch(epoch_data)
