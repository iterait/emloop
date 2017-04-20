"""
Module with a hook which reports the time profile data in the stanard logging.
"""
import logging

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

    def after_epoch_profile(self, epoch_id: int, profile: Timer.TimeProfile) -> None:
        """Summarize and log the given epoch profile."""

        # time spent reading data from streams (train + valid + test)
        read_data = sum(profile['read_batch_train']) + sum(profile['read_batch_valid'])
        if 'read_batch_test' in profile:
            read_data += sum(profile['read_batch_test'])

        # time spent processing hooks (after batch + after epoch for train + valid + test)
        hooks = (sum(profile['after_batch_hooks_train']) +
                 sum(profile['after_batch_hooks_valid']) +
                 sum(profile['after_epoch_hooks']))

        if 'after_batch_hooks_test' in profile:
            hooks += sum(profile['after_batch_hooks_test'])

        # time spent evaluating valid + test
        _eval = sum(profile['eval_batch_valid'])
        if 'eval_batch_valid' in profile:
            _eval += sum(profile['eval_batch_valid'])

        # time spent training
        train = sum(profile['eval_batch_train'])

        logging.info('\tT read data:\t%f', read_data)
        logging.info('\tT train:\t%f', train)
        logging.info('\tT valid+test:\t%f', _eval)
        logging.info('\tT hooks:\t%f', hooks)
