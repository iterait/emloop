"""
Module with log output directory hook test case (see :py:class:`cxflow.hooks.LogDir`).
"""
import os.path as path

from testfixtures import LogCapture

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks import LogDir


class LogProfileTest(CXTestCase):
    """
    Test case for :py:class:`cxflow.hooks.LogDir` hook.
    """

    def setUp(self):
        self._dir = path.join('some', 'path')
        self._hook = LogDir(output_dir=self._dir)

    def test_log_dir(self):
        """Test output dir logging in the respective events."""
        with LogCapture() as log_capture:
            self._hook.before_training()
            self._hook.after_epoch(batch_data={}, stream_name='Dummy')
            self._hook.after_training()

        log_capture.check(
            ('root', 'INFO', 'Output dir: {}'.format(self._dir)),
            ('root', 'INFO', 'Output dir: {}'.format(self._dir)),
            ('root', 'INFO', 'Output dir: {}'.format(self._dir))
        )
