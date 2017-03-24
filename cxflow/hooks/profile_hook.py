from .abstract_hook import AbstractHook
from ..utils.profile import Timer
import logging


class ProfileHook(AbstractHook):
    """Summarize and log epoch profile."""

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def after_epoch_profile(self, epoch_id: int, profile: Timer.TimeProfile, **kwargs) -> None:

        # time spent reading data from streams (train + valid + test)
        read_data = sum(profile['read_batch_train']) + sum(profile['read_batch_valid'])
        if 'read_batch_test' in profile:
            read_data += sum(profile['read_batch_test'])

        # time spent processing hooks (after batch + after epoch for train + valid + test)
        hooks = sum(profile['after_batch_hooks_train']) + \
                sum(profile['after_batch_hooks_valid']) + \
                sum(profile['after_epoch_hooks'])
        if 'after_batch_hooks_test' in profile:
            hooks += sum(profile['after_batch_hooks_test'])

        # time spent evaluating valid + test
        eval = sum(profile['eval_batch_valid'])
        if 'eval_batch_valid' in profile:
            eval += sum(profile['eval_batch_valid'])

        # time spent training
        train = sum(profile['eval_batch_train'])

        logging.info('\tT read data:\t%f', read_data)
        logging.info('\tT train:\t%f', train)
        logging.info('\tT valid+test:\t%f', eval)
        logging.info('\tT hooks:\t%f', hooks)
