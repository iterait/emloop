"""
Module with benchmarking hook test case (see :py:class:`emloop.hooks.Benchmark`).
"""
import logging
import numpy as np

from emloop.hooks import Benchmark


_PROFILE = {'eval_batch_train': [4.0001, 7, 0.005, 4.542],
            'eval_batch_valid': [0, 1, 6, 4],
            'eval_batch_test': [1.3, 3.332, 3.001, 0.5]}

train_mean = np.mean([2.00005, 3.5, 0.0025])
train_median = np.median([2.00005, 3.5, 0.0025])
valid_mean = np.mean([0, 0.5, 3])
valid_median = np.median([0, 0.5, 3])
test_mean = np.mean([0.65, 1.666, 1.5005])
test_median = np.median([0.65, 1.666, 1.5005])

hook = Benchmark(batch_size=2)


def test_single_hook_benchmarking(caplog):
    """Test single hook benchmarking."""
    caplog.set_level(logging.INFO)

    hook.after_epoch_profile(0, _PROFILE, ['train'])
    assert caplog.record_tuples == [
        ('root', logging.INFO, 'train - time per example: mean={:.5f}s, median={:.5f}s'.format(train_mean,
                                                                                               train_median))
    ]


def test_all_hooks_benchmarking(caplog):
    """Test all hooks benchmarking."""
    caplog.set_level(logging.INFO)

    hook.after_epoch_profile(0, _PROFILE, ['train', 'test', 'valid'])
    assert caplog.record_tuples == [
        ('root', logging.INFO, 'train - time per example: mean={:.5f}s, median={:.5f}s'.format(train_mean,
                                                                                               train_median)),
        ('root', logging.INFO, 'test - time per example: mean={:.5f}s, median={:.5f}s'.format(test_mean, test_median)),
        ('root', logging.INFO, 'valid - time per example: mean={:.5f}s, median={:.5f}s'.format(valid_mean,
                                                                                               valid_median))
    ]
