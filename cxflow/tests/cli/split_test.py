"""
Test module for cxflow split command (cli/split.py)
"""

from cxflow.cli.split import split
from cxflow.tests.test_core import CXTestCaseWithDir
from cxflow.utils.config import config_to_file
from cxflow.constants import CXF_CONFIG_FILE


class SplitDataset:
    """Simple dataset which records its split method calls."""

    def __init__(self, _):
        _SPLIT_DATASET_INSTANCES.append(self)
        self.split_calls = []

    def split(self, num_splits: int, train: float, valid: float, test: float):
        """Record the call arguments."""
        self.split_calls.append({'n': num_splits, 'tr': train, 'v': valid, 'te': test})

_SPLIT_DATASET_INSTANCES = []


class CLISplitTest(CXTestCaseWithDir):
    """Split function test case."""

    def test_split(self):
        """Test if split creates the dataset and calls the split function properly."""
        config = {'dataset': {'module': 'cxflow.tests.cli.split_test', 'class': 'SplitDataset'}}
        config_file = config_to_file(config, self.tmpdir, CXF_CONFIG_FILE)
        split(config_file, 7, 5, 3, 1)

        self.assertEqual(len(_SPLIT_DATASET_INSTANCES), 1)
        dataset = _SPLIT_DATASET_INSTANCES[0]
        self.assertIsInstance(dataset, SplitDataset)
        self.assertEqual(len(dataset.split_calls), 1)
        self.assertDictEqual(dataset.split_calls[0], {'n': 7, 'tr': 5, 'v': 3, 'te': 1})
