"""
Module with log variables hook test case (see :py:class:`cxflow.hooks.LogVariables`).
"""
import collections

from testfixtures import LogCapture
import numpy as np

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks import LogVariables

_EXAMPLES = 6
_EPOCH_ID = 666


def _get_epoch_data():
    """Return testing epoch data."""
    epoch_data = collections.OrderedDict([
        ('train', collections.OrderedDict([
            ('accuracy', 1),
            ('precision', np.ones(_EXAMPLES)),
            ('loss', collections.OrderedDict([('mean', 1)])),
            ('loss2', collections.OrderedDict([('mean', 1),
                                               ('median', 11)])),
        ])),
        ('test', collections.OrderedDict([
            ('accuracy', 2),
            ('precision', 2 * np.ones(_EXAMPLES)),
            ('loss', collections.OrderedDict([('mean', 2)])),
            ('loss2', collections.OrderedDict([('mean', 2),
                                               ('median', 22)])),
        ])),
    ])
    return epoch_data


class LogVariablesTest(CXTestCase):
    """
    Test case for :py:class:`cxflow.hooks.LogVariables` hook.
    """

    def test_raise_unknown_type_action(self):
        """
        Test raising error, if the `on_unknown_type` option is not set
        to the one of the following value: `error`, `warn`, `str` or `ignore`.
        """
        with self.assertRaises(AssertionError):
            LogVariables([], on_unknown_type="unknown")

    def test_raise_unknown_var(self):
        """
        Test raising `KeyError`, if some of the selected
        variable is not present in `epoch_data` streams.
        """
        with self.assertRaises(KeyError):
            LogVariables(['accuracy', 'precision', 'unknown']).after_epoch(_EPOCH_ID, _get_epoch_data())

    def test_log_variables_all(self):
        """
        Test logging all variables from `epoch_data` streams.
        """
        with LogCapture() as log_capture:
            LogVariables().after_epoch(_EPOCH_ID, _get_epoch_data())

        log_capture.check(
            ('root', 'INFO', '\ttrain accuracy: 1.000000'),
            ('root', 'INFO', '\ttrain loss mean: 1.000000'),
            ('root', 'INFO', '\ttrain loss2:'),
            ('root', 'INFO', '\t\tmean: 1.000000'),
            ('root', 'INFO', '\t\tmedian: 11.000000'),
            ('root', 'INFO', '\ttest accuracy: 2.000000'),
            ('root', 'INFO', '\ttest loss mean: 2.000000'),
            ('root', 'INFO', '\ttest loss2:'),
            ('root', 'INFO', '\t\tmean: 2.000000'),
            ('root', 'INFO', '\t\tmedian: 22.000000')
        )

    def test_log_variables_selected(self):
        """
        Test logging of selected variables from `epoch_data` streams.
        """
        with LogCapture() as log_capture:
            LogVariables(['accuracy', 'loss2']).after_epoch(_EPOCH_ID, _get_epoch_data())

        log_capture.check(
            ('root', 'INFO', '\ttrain accuracy: 1.000000'),
            ('root', 'INFO', '\ttrain loss2:'),
            ('root', 'INFO', '\t\tmean: 1.000000'),
            ('root', 'INFO', '\t\tmedian: 11.000000'),
            ('root', 'INFO', '\ttest accuracy: 2.000000'),
            ('root', 'INFO', '\ttest loss2:'),
            ('root', 'INFO', '\t\tmean: 2.000000'),
            ('root', 'INFO', '\t\tmedian: 22.000000')
        )

    def test_log_variables_raise_error(self):
        """
        Test raising error, if `on_unknown_type` is set to `error` action
        and the type of some variable in `epoch_data` streams is not `scalar` or `dict`.
        """
        with self.assertRaises(TypeError):
            LogVariables(on_unknown_type='error').after_epoch(_EPOCH_ID, _get_epoch_data())

    def test_log_variables_warn(self):
        """
        Test logging warning, if `on_unknown_type` is set to `warn` action
        and the type of some variable in `epoch_data` streams is not `scalar` or `dict`.
        """
        with LogCapture() as log_capture:
            LogVariables(['precision'], on_unknown_type='warn').after_epoch(_EPOCH_ID, _get_epoch_data())
        log_capture.check(
            ('root', 'WARNING', 'Variable type `ndarray` can not be logged. Variable name: `precision`.'),
            ('root', 'WARNING', 'Variable type `ndarray` can not be logged. Variable name: `precision`.')
        )

    def test_log_variables_str(self):
        """
        Test logging variables, if `on_unknown_type` is set to `str` action
        and the type of some variable in `epoch_data` streams is not `scalar` or `dict`.
        """
        with LogCapture() as log_capture:
            LogVariables(['precision'], on_unknown_type='str').after_epoch(_EPOCH_ID, _get_epoch_data())
        log_capture.check(
            ('root', 'INFO', '\ttrain precision: [1. 1. 1. 1. 1. 1.]'),
            ('root', 'INFO', '\ttest precision: [2. 2. 2. 2. 2. 2.]')
        )
