"""
Test module for profile utils (cxflow.utils.profile).
"""
import time

from cxflow.tests.test_core import CXTestCase
from cxflow.utils.profile import Timer


class TimerTest(CXTestCase):
    """Test case for Timer profiling object."""

    def test_empty_timer(self):
        """Test near zero measured time for no-op."""
        log = {}
        with Timer('noop', log):
            pass

        self.assertIn('noop', log)
        self.assertEqual(len(log['noop']), 1)
        self.assertAlmostEqual(log['noop'][0], 0, places=4)

    def test_appending(self):
        """Test correct usage of the given profile."""
        log = {}

        n_iter = 100

        for _ in range(n_iter):
            with Timer('noop', log):
                pass

        self.assertEqual(len(log['noop']), n_iter)

    def test_nested(self):
        """Test nesting of Timer usage pattern."""
        log = {}

        with Timer('timerop', log):
            with Timer('noop', log):
                pass

        self.assertIn('noop', log)
        self.assertIn('timerop', log)

        self.assertGreater(log['timerop'][0], log['noop'][0])

    def test_time(self):
        """Test reasonable precision in measuring time."""
        log = {}

        sleep_time = 1

        with Timer('sleep', log):
            time.sleep(sleep_time)

        self.assertAlmostEqual(log['sleep'][0], sleep_time, places=1)

    def test_order(self):
        """Test correct order of measurements in the profile."""
        log = {}

        for i in range(3):

            with Timer('sleep', log):
                time.sleep(i)

            self.assertAlmostEqual(log['sleep'][i], i, places=1)
