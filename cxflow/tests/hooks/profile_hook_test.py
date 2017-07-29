"""
Module with profile hook test case (see cxflow.hooks.profile_hook).
"""
from cxflow.tests.test_core import CXTestCase
from cxflow.hooks import ProfileHook


class ProfileHookTest(CXTestCase):
    """
    Test case for ProfileHook.

    Hereby, we test only proper handling (logging) of the profile.
    The profiling itself is tested in the main_loop test (see cxflow.tests.main_loop_test.py).
    """

    def test_missing_train(self):
        """Test KeyError raised on missing profile entries."""
        hook = ProfileHook()

        self.assertRaises(KeyError, hook.after_epoch_profile, 0, {}, [])
        self.assertRaises(KeyError, hook.after_epoch_profile, 0, {'some_contents': 1}, [])


    def test_train_only(self):
        """Test profile handling with only train stream."""
        hook = ProfileHook()

        hook.after_epoch_profile(1, {'read_batch_train': [1, 2, 3],
                                     'after_batch_hooks_train': [4, 0, 0],
                                     'after_epoch_hooks': [5],
                                     'eval_batch_train': [6, 7, 8]}, [])
