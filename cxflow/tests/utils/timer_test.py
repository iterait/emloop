from cxflow.utils.profile import Timer

import logging
import time
from unittest import TestCase


class TimerTest(TestCase):
    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_empty_timer(self):
        log = {}
        with Timer('noop', log):
            pass

        self.assertIn('noop', log)
        self.assertEqual(len(log['noop']), 1)
        self.assertAlmostEqual(log['noop'][0], 0, places=4)

    def test_appending(self):
        log = {}

        n_iter = 100

        for i in range(n_iter):
            with Timer('noop', log):
                pass

        self.assertEqual(len(log['noop']), n_iter)

    def test_nested(self):
        log = {}

        with Timer('timerop', log):
            with Timer('noop', log):
                pass

        self.assertIn('noop', log)
        self.assertIn('timerop', log)

        self.assertGreater(log['timerop'][0], log['noop'][0])

    def test_time(self):
        log = {}

        sleep_time = 1

        with Timer('sleep', log):
            time.sleep(sleep_time)

        self.assertAlmostEqual(log['sleep'][0], sleep_time, places=1)

    def test_order(self):
        log = {}

        for i in range(3):

            with Timer('sleep', log):
                time.sleep(i)

            self.assertAlmostEqual(log['sleep'][i], i, places=1)
