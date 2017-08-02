"""
Module with a hook which reports the time profile data in the stanard logging.
"""
import logging
from itertools import chain
from typing import Iterable

from .abstract_hook import AbstractHook
from ..utils.profile import Timer


class ProfileHook(AbstractHook):
    """
    Summarize and log epoch profile.

    -------------------------------------------------------
    Example usage in config
    -------------------------------------------------------
    # log all the variables
    hooks:
      - class: ProfileHook
    -------------------------------------------------------
    """

    def after_epoch_profile(self, _, profile: Timer.TimeProfile, extra_streams: Iterable[str]) -> None:
        """
        Summarize and log the given epoch profile.

        The profile is expected to contain at least:
            - `read_data_train`, `eval_batch_train` and `after_batch_hooks_train` entries produced by the train stream
            - `after_epoch_hooks` entry

        :param profile: epoch timings profile
        :param extra_streams: enumeration of additiona stream names
        """

        read_data_total = 0
        eval_total = 0
        train_total = sum(profile['eval_batch_train'])
        hooks_total = sum(profile['after_epoch_hooks'])

        for stream_name in chain(extra_streams, ['train']):
            read_data_total += sum(profile['read_data_' + stream_name])
            hooks_total += sum(profile['after_batch_hooks_' + stream_name])
            if stream_name != 'train':
                eval_total += sum(profile['eval_batch_' + stream_name])

        logging.info('\tT read data:\t%f', read_data_total)
        logging.info('\tT train:\t%f', train_total)
        logging.info('\tT eval:\t%f', eval_total)
        logging.info('\tT hooks:\t%f', hooks_total)
