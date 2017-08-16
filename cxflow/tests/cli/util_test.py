"""
Test module for cxflow cli utils (cli/util.py).
"""

from cxflow.cli.util import validate_config

from cxflow.tests.test_core import CXTestCaseWithDir
from cxflow.utils.config import config_to_file, load_config


class CLIUtilTest(CXTestCaseWithDir):
    """Entry point functions test case."""

    def test_train_load_config(self):
        """Test correct config loading."""

        # test a config call with both dataset and model
        good_config = {'dataset': None, 'model': None}
        config_path = config_to_file(good_config, self.tmpdir, 'config1.yaml')

        # test assertion when config is incomplete
        missing_model_config = {'dataset': None}
        config_path2 = config_to_file(missing_model_config, self.tmpdir, 'config2.yaml')
        loaded_config2 = load_config(config_path2, [])
        self.assertRaises(AssertionError, validate_config, loaded_config2)

        missing_dataset_config = {'dataset': None}
        config_path3 = config_to_file(missing_dataset_config, self.tmpdir, 'config3.yaml')
        loaded_config3 = load_config(config_path3, [])
        self.assertRaises(AssertionError, validate_config, loaded_config3)

        # test return value
        returned_config = load_config(config_path, [])
        validate_config(returned_config)
        self.assertDictEqual(returned_config, load_config(config_path, []))
        self.assertDictEqual(returned_config, good_config)
