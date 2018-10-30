"""
Module with a hook which reports the time profile data in the stanard logging.
"""
import logging
from itertools import chain
from typing import Iterable

from . import AbstractHook
from ..types import TimeProfile


class LogProfile(AbstractHook):
    """
    Summarize and log epoch profile via standard :py:mod:`logging`.

    Epoch profile contains info about time spent training, reading data etc. For full reference, see
    :py:class:`emloop.MainLoop`.

    .. code-block:: yaml
        :caption: log the time profile after each epoch

        hooks:
          - LogProfile

    """

    def after_epoch_profile(self, epoch_id, profile: TimeProfile) -> None:
        """
        Summarize and log the given epoch profile.

        The profile is expected to contain at least:
            - ``read_data_train``, ``eval_batch_train`` and ``after_batch_hooks_train`` entries produced by the train
              stream (if train stream name is `train`)
            - ``after_epoch_hooks`` entry

        :param profile: epoch timings profile
        :param extra_streams: enumeration of additional stream names
        """

        streams = []
        for key_name in profile.keys():
            if key_name.startswith("eval_batch_"):
                streams.append(key_name[len("eval_batch_"):])

        read_data_total = 0
        eval_total = 0
        hooks_total = sum(profile.get('after_epoch_hooks', []))

        for stream_name in streams:
            read_data_total += sum(profile.get('read_batch_' + stream_name, []))
            hooks_total += sum(profile.get('after_batch_hooks_' + stream_name, []))

        for stream_name in streams:
            logging.info('\tT %s:\t%f', stream_name, sum(profile.get('eval_batch_{}'.format(stream_name), [])))

        logging.info('\tT read data:\t%f', read_data_total)
        logging.info('\tT hooks:\t%f', hooks_total)
