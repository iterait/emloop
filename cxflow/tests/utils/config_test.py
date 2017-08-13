"""
Test module for config utils functions (cxflow.utils.config).
"""
from os import path

import yaml

from cxflow.tests.test_core import CXTestCase, CXTestCaseWithDir
from cxflow.utils.config import parse_arg, load_config, config_to_file, config_to_str
from cxflow.constants import CXF_CONFIG_FILE


class ConfigTestParseArg(CXTestCase):
    """Test case for parse_arg function."""

    def test_default_type(self):
        """Test default type."""
        for key, val in [('common.name', 'BatchSize1'), ('net.name', 'netie'), ('stream.train.seed', 'none')]:
            parsed_key, parsed_val = parse_arg(key+'='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), str)

    def test_str_type(self):
        """Test str type."""
        for key, val in [('common.name', 'BatchSize1'), ('net.name', 'netie'), ('stream.train.seed', 'none')]:
            parsed_key, parsed_val = parse_arg(key+':str='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), str)

    def test_int_type(self):
        """Test int type."""
        for key, val in [('common.batch_size', 12), ('net.dropout', 0), ('stream.train.seed', 123)]:
            parsed_key, parsed_val = parse_arg(key+':int='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), int)

        self.assertRaises(ValueError, parse_arg, 'common.batch_size:int=12.7')

    def test_float_type(self):
        """Test float type."""
        for key, val in [('common.some_int_number', 12), ('net.dropout', 0.5), ('stream.train.float_seed', 123.456)]:
            parsed_key, parsed_val = parse_arg(key+':float='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), float)

    def test_bool_type(self):
        """Test boolean type."""
        for key, val in [('common.quiet', 1), ('net.dropout', 0), ('stream.train.float_seed', 1)]:
            parsed_key, parsed_val = parse_arg(key+':bool='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), bool)

    def test_ast_type(self):
        """Test ast type."""
        for key, val in [('common.arch', [1, 2, 3.4, 5]), ('net.arch', {"a": "b"}),
                         ('stream.train.deep', {"a": {"b": ["c", "d", "e"]}}), ('net.arch', 12), ('net.arch', 12.2)]:
            parsed_key, parsed_val = parse_arg(key+':ast='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), type(val))

    def test_not_int_type(self):
        """Test parse_arg raises on bad int value."""
        for key, val in [('common.batch_size', "ahoj"), ('stream.train.seed', [1, 2])]:
            self.assertRaises(ValueError, parse_arg, key+':int='+str(val))

    def test_not_float_type(self):
        """Test parse_arg raises on bad float value."""
        for key, val in [('common.some_number', True), ('net.dropout', "hello"), ('stream.train.float_seed', [1, 2])]:
            self.assertRaises(ValueError, parse_arg, key+':float='+str(val))

    def test_not_bool_type(self):
        """Test parse_arg raises on bad boolean value."""
        for key, val in [('common.quiet', "hello"), ('net.dropout', 0.2), ('stream.train.float_seed', 13),
                         ('stream.train.float_seed', [1, 3])]:
            self.assertRaises(ValueError, parse_arg, key+':bool='+str(val))

    def test_not_ast_type(self):
        """Test parse_arg raises on bad ast value."""
        for key, val in [('common.arch', "hello"), ('net.arch', '[12,3'), ('net.arch', '{"a": }')]:
            self.assertRaises(ValueError, parse_arg, key + ':ast=' + str(val))


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


class ConfigTest(CXTestCaseWithDir):
    """Test case for load_config, config_to_file and config_to_str functions."""

    def test_load_anchorless_config(self):
        """Test loading of a config without yaml anchors."""

        f_name = path.join(self.tmpdir, 'conf.yaml')

        with open(f_name, 'w') as file:
            file.write(_TEST_ANCHORLESS_YAML)

        self.assertDictEqual(load_config(f_name, []), {'e': {'f': 'f', 'h': ['j', 'k']}})
        self.assertDictEqual(load_config(f_name, ['e.f:int=12']), {'e': {'f': 12, 'h': ['j', 'k']}})
        self.assertDictEqual(load_config(f_name, ['e.x:int=12']), {'e': {'f': 'f', 'h': ['j', 'k'], 'x': 12}})

    def test_load_anchored_config(self):
        """Test loading of a config with yaml anchors."""
        f_name = path.join(self.tmpdir, 'conf.yaml')

        with open(f_name, 'w') as file:
            file.write(_TEST_ANCHORED_YAML)

        self.assertDictEqual(load_config(f_name, []), {'a': {'b': 'c', 'd': 11},
                                                       'e': {'f': 'f', 'h': ['j', 'k'], 'b': 'c', 'd': 11}})
        self.assertDictEqual(load_config(f_name, ['a.b:int=12']), {'a': {'b': 12, 'd': 11},
                                                                   'e': {'f': 'f', 'h': ['j', 'k'], 'b': 12, 'd': 11}})
        self.assertDictEqual(load_config(f_name, ['e.b:int=19']),
                             {'a': {'b': 'c', 'd': 11}, 'e': {'f': 'f', 'h': ['j', 'k'], 'b': 19, 'd': 11}})

    def test_dump_config(self):
        """Test config_to_file and config_to_str function."""

        config = {'e': {'f': 'f', 'h': ['j', 'k']}}

        # test if the return path is correct and re-loading does not change the config
        config_path = config_to_file(config, output_dir=self.tmpdir, name=CXF_CONFIG_FILE)
        self.assertTrue(path.exists(config_path))
        self.assertDictEqual(load_config(config_path, []), config)

        # test custom naming
        dump_name = 'my-conf.yaml'
        config_path = config_to_file(config, output_dir=self.tmpdir, name=dump_name)
        self.assertEqual(path.join(self.tmpdir, dump_name), config_path)
        self.assertTrue(path.exists(config_path))

        # test dump to string (effectively, test pyaml)
        yaml_str = config_to_str(config)
        self.assertDictEqual(yaml.load(yaml_str), config)
