"""
Hook for benchmarking models and logging average example times.
"""
import logging
import numpy as np
from typing import List

from . import AbstractHook
from ..types import TimeProfile


class Benchmark(AbstractHook):
    """
    Log mean and median example times via standard :py:mod:`logging`.

    .. code-block:: yaml
        :caption: log mean and median example times after each epoch

        hooks:
          - Benchmark

    """

    def __init__(self, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = batch_size

    def after_epoch_profile(self, epoch_id: int, profile: TimeProfile, streams: List[str]):
        """
        Log average example times after each epoch.

        The profile is expected to contain at least `eval_batch_{stream}` entry for each logged stream.

        :param epoch_id: number of the processed epoch
        :param profile: epoch timings profile
        :param streams: streams for which example times will be logged
        """
        for stream_name in streams:
            batch_times = profile.get('eval_batch_' + stream_name, [])
            # last batch may be smaller than the other ones, so we drop it to not skew the measurement
            example_times = list(map(lambda x: x / float(self._batch_size), batch_times[:-1]))
            logging.info('{} - time per example: mean={:.5f}s, median={:.5f}s'.format(stream_name,
                                                                                      np.mean(example_times),
                                                                                      np.median(example_times)))
