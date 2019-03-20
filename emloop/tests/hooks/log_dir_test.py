"""
Module with log output directory hook test case (see :py:class:`emloop.hooks.LogDir`).
"""
import os.path as path
import logging

from emloop.hooks import LogDir


def test_log_dir(caplog):
    """Test output dir logging in the respective events."""
    test_dir = path.join('some', 'path')
    hook = LogDir(output_dir=test_dir)
    caplog.set_level(logging.INFO)
    hook.before_training()
    hook.after_epoch(batch_data={}, stream_name='Dummy')
    hook.after_training(True)

    assert caplog.record_tuples == [
        ('root', logging.INFO, 'Output dir: {}'.format(test_dir)),
        ('root', logging.INFO, 'Output dir: {}'.format(test_dir)),
        ('root', logging.INFO, 'Output dir: {}'.format(test_dir))
    ]
