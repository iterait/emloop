"""Test module for :py:class:`TrainingTrace`."""
import os.path as path
from datetime import datetime

from cxflow.utils import TrainingTrace, TrainingTraceKeys
from cxflow.constants import CXF_TRACE_FILE

from ..test_core import CXTestCaseWithDir


class TrainingTraceTest(CXTestCaseWithDir):
    """Test case for :py:class:`TrainingTrace` class."""

    def test_bad_keys(self):
        """Test training trace key types being asserted."""
        trace = TrainingTrace()
        with self.assertRaises(AssertionError):
            print(trace['train_begin'])
        with self.assertRaises(AssertionError):
            trace['train_begin'] = 'value'

    def test_set_get(self):
        """Test training trace __getitem__ and __setitem__."""
        value = 42
        trace = TrainingTrace(autosave=False)
        trace[TrainingTraceKeys.EPOCHS_DONE] = value
        self.assertEqual(trace[TrainingTraceKeys.EPOCHS_DONE], value)

    def test_save_load(self):
        """Test training trace (auto)saving and loading."""
        trace = TrainingTrace(output_dir=self.tmpdir)
        now = datetime.now()
        trace[TrainingTraceKeys.TRAIN_BEGIN] = now  # should auto-save now
        loaded_trace = TrainingTrace.from_file(path.join(self.tmpdir, CXF_TRACE_FILE))
        self.assertEqual(loaded_trace[TrainingTraceKeys.TRAIN_BEGIN], now)

    def test_output_dir(self):
        """Test ValueError is raised in save with not output_dir."""
        with self.assertRaises(ValueError):
            TrainingTrace().save()
