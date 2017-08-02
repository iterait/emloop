"""
Module with profile hook test case (see cxflow.hooks.profile_hook).
"""
from testfixtures import LogCapture

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks import ProfileHook

_TRAIN_ONLY_PROFILE = {'read_data_train': [1.12, 2, 3],
                       'after_batch_hooks_train': [4, 0, 0.4],
                       'eval_batch_train': [6, 7.9, 8],
                       'after_epoch_hooks': [5.4]}

_TRAIN_AND_VALID_PROFILE = {'read_data_train': [1.001, 2, 3],
                            'after_batch_hooks_train': [4, 0, 0],
                            'eval_batch_train': [6, 7, 8.54],
                            'read_data_valid': [3, 5, 6],
                            'after_batch_hooks_valid': [1, 1.11, 1],
                            'eval_batch_valid': [0, 1, 2],
                            'after_epoch_hooks': [5.3]}

_TRAIN_TEST_AND_VALID_PROFILE = {'read_data_train': [1.001, 2, 3],
                                 'after_batch_hooks_train': [4, 0, 0],
                                 'eval_batch_train': [6, 7, 8.54],
                                 'read_data_valid': [3, 5, 6],
                                 'after_batch_hooks_valid': [1, 1.11, 1],
                                 'eval_batch_valid': [0, 1, 2],
                                 'read_data_test': [1.1, 1.11, 1.111],
                                 'after_batch_hooks_test': [2.2, 2.22, 2.222],
                                 'eval_batch_test': [3.3, 3.33, 3.333],
                                 'after_epoch_hooks': [5.3]}


class ProfileHookTest(CXTestCase):
    """
    Test case for ProfileHook.

    Hereby, we test only proper handling (logging) of the profile.
    The profiling itself is tested in the main_loop test (see cxflow.tests.main_loop_test.py).
    """

    def setUp(self):
        self._hook = ProfileHook()

    def test_missing_train(self):
        """Test KeyError raised on missing profile entries."""
        self.assertRaises(KeyError, self._hook.after_epoch_profile, 0, {}, [])
        self.assertRaises(KeyError, self._hook.after_epoch_profile, 0, {'some_contents': 1}, [])

    def test_train_only(self):
        """Test profile handling with only train stream."""
        with LogCapture() as log_capture:
            self._hook.after_epoch_profile(1, _TRAIN_ONLY_PROFILE, [])

        log_capture.check(
            ('root', 'INFO', '\tT read data:\t6.120000'),
            ('root', 'INFO', '\tT train:\t21.900000'),
            ('root', 'INFO', '\tT eval:\t0.000000'),
            ('root', 'INFO', '\tT hooks:\t9.800000'),
        )

    def test_extra_streams(self):
        """Test extra streams handling."""
        self.assertRaises(KeyError, self._hook.after_epoch_profile, 0, _TRAIN_ONLY_PROFILE, ['valid'])

        # test additional entries being ignored if the extra stream was not specified
        with LogCapture() as log_capture:
            self._hook.after_epoch_profile(1, _TRAIN_AND_VALID_PROFILE, [])
        log_capture.check(
            ('root', 'INFO', '\tT read data:\t6.001000'),
            ('root', 'INFO', '\tT train:\t21.540000'),
            ('root', 'INFO', '\tT eval:\t0.000000'),
            ('root', 'INFO', '\tT hooks:\t9.300000'),
        )

        # test one additional stream
        with LogCapture() as log_capture2:
            self._hook.after_epoch_profile(1, _TRAIN_AND_VALID_PROFILE, ['valid'])
        log_capture2.check(
            ('root', 'INFO', '\tT read data:\t20.001000'),
            ('root', 'INFO', '\tT train:\t21.540000'),
            ('root', 'INFO', '\tT eval:\t3.000000'),
            ('root', 'INFO', '\tT hooks:\t12.410000'),
        )

        # test two additional streams
        with LogCapture() as log_capture2:
            self._hook.after_epoch_profile(1, _TRAIN_TEST_AND_VALID_PROFILE, ['valid', 'test'])
        log_capture2.check(
            ('root', 'INFO', '\tT read data:\t23.322000'),
            ('root', 'INFO', '\tT train:\t21.540000'),
            ('root', 'INFO', '\tT eval:\t12.963000'),
            ('root', 'INFO', '\tT hooks:\t19.052000'),
        )
