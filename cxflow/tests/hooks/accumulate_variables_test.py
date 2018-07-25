"""
Test module for accumulating hook (cxflow.hooks.accumulate_variables_hook).
"""

import numpy as np

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.accumulate_variables import AccumulateVariables


_ITERS = 9
_EXAMPLES = 5
_FEATURES = 6


class AccumulateVariablesTest(CXTestCase):
    """Test case for AccumulateVariables hook."""

    def get_batch(self):
        batch = {'input': np.ones((_EXAMPLES, _FEATURES)),
                 'target': np.zeros(_EXAMPLES),
                 'accuracy': np.ones(_EXAMPLES),
                 'cost': np.ones(_EXAMPLES),
                 'not_iter': 1}
        return batch

    def test_accumulating_present_variables(self):
        """Test accumulating selected variables which are present in a batch."""

        selected_vars = ["accuracy", "cost"]
        stream_name = "train"
        accum_hook = AccumulateVariables(variables=selected_vars)

        for _ in range(_ITERS):
            batch = self.get_batch()
            accum_hook.after_batch(stream_name, batch)

        for var in selected_vars:
            self.assertEqual(len(accum_hook._accumulator[stream_name][var]), _EXAMPLES * _ITERS)
            self.assertTrue(np.array_equal(accum_hook._accumulator[stream_name][var],
                                           np.ones(_EXAMPLES * _ITERS)))

    def test_raise_var_not_present(self):
        """Test raising an exception if selected variable is not present in a batch."""

        selected_vars = ["accuracy", "cost", "classes"]
        stream_name = "train"
        accum_hook = AccumulateVariables(variables=selected_vars)

        batch = self.get_batch()
        with self.assertRaises(KeyError):
            accum_hook.after_batch(stream_name, batch)


    def test_raise_var_not_iterable(self):
        """Test raising an exception if selected variable is not iterable."""

        selected_vars = ["accuracy", "cost", "not_iter"]
        stream_name = "train"
        accum_hook = AccumulateVariables(variables=selected_vars)
        batch = self.get_batch()
        with self.assertRaises(TypeError):
            accum_hook.after_batch(stream_name, batch)

    def test_init_accumulator(self):
        """Test reseting accumulator after epoch."""

        selected_vars = ["accuracy", "cost"]
        stream_name = "train"
        accum_hook = AccumulateVariables(variables=selected_vars)

        for _ in range(_ITERS):
            batch = self.get_batch()
            accum_hook.after_batch(stream_name, batch)

        accum_hook.after_epoch()
        self.assertFalse(accum_hook._accumulator)
