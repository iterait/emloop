"""
Test module for yaml utils functions (cxflow.utils.yaml).
"""
from os import path
from collections import OrderedDict

import yaml

from cxflow.tests.test_core import CXTestCaseWithDir
from cxflow.utils.yaml import yaml_to_file, yaml_to_str, load_yaml
from cxflow.constants import CXF_CONFIG_FILE


_TEST_ANCHORLESS_YAML = """
e:
  f: f
  h:
    - j
    - k
"""


_TEST_ANCHORED_YAML = """
a: &anchor
  b: c
  d: 11

e:
  <<: *anchor
  f: f
  h:
    - j
    - k
"""


class YAMLTest(CXTestCaseWithDir):
    """Test case for YAML util functions."""

    def test_load_anchorless_yaml(self):
        """Test loading of a YAML without yaml anchors."""

        f_name = path.join(self.tmpdir, 'conf.yaml')

        with open(f_name, 'w') as file:
            file.write(_TEST_ANCHORLESS_YAML)

        self.assertDictEqual(load_yaml(f_name), {'e': {'f': 'f', 'h': ['j', 'k']}})

    def test_load_anchored_yaml(self):
        """Test loading of a YAML with yaml anchors."""
        f_name = path.join(self.tmpdir, 'conf.yaml')

        with open(f_name, 'w') as file:
            file.write(_TEST_ANCHORED_YAML)

        self.assertDictEqual(load_yaml(f_name)['a'], {'b': 'c', 'd': 11})
        self.assertEqual(OrderedDict(load_yaml(f_name)['e']), OrderedDict([('f', 'f'), ('h', ['j', 'k']),
                                                                           ('b', 'c'), ('d', 11)]))

    def test_dump_yaml(self):
        """Test yaml_to_file and yaml_to_str function."""

        config = {'e': {'f': 'f', 'h': ['j', 'k']}}

        # test if the return path is correct and re-loading does not change the config
        config_path = yaml_to_file(config, output_dir=self.tmpdir, name=CXF_CONFIG_FILE)
        self.assertTrue(path.exists(config_path))
        self.assertDictEqual(load_yaml(config_path), config)

        # test custom naming
        dump_name = 'my-conf.yaml'
        config_path = yaml_to_file(config, output_dir=self.tmpdir, name=dump_name)
        self.assertEqual(path.join(self.tmpdir, dump_name), config_path)
        self.assertTrue(path.exists(config_path))

        # test dump to string (effectively, test pyaml)
        yaml_str = yaml_to_str(config)
        self.assertDictEqual(yaml.load(yaml_str), config)
