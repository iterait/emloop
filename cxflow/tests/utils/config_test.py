"""
Test module for config utils functions (cxflow.utils.config).
"""
from os import path
from collections import OrderedDict

from cxflow.tests.test_core import CXTestCase, CXTestCaseWithDir
from cxflow.utils.config import parse_arg, load_config


class ConfigTestParseArg(CXTestCase):
    """Test case for parse_arg function."""

    def test_default_type(self):
        """Test default type."""
        for key, val in [('common.name', 'BatchSize1'), ('model.name', 'modelie'), ('stream.train.seed', 'none')]:
            parsed_key, parsed_val = parse_arg(key+'='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), str)

    def test_str_type(self):
        """Test str type."""
        for key, val in [('common.name', 'BatchSize1'), ('model.name', 'modelie'), ('stream.train.seed', 'none')]:
            parsed_key, parsed_val = parse_arg(key+'='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), str)

    def test_int_type(self):
        """Test int type."""
        for key, val in [('common.batch_size', 12), ('model.dropout', 0), ('stream.train.seed', 123)]:
            parsed_key, parsed_val = parse_arg(key+'='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), int)

    def test_float_type(self):
        """Test float type."""
        for key, val in [('common.some_int_number', 12.), ('model.dropout', 0.5), ('stream.train.float_seed', 123.456)]:
            parsed_key, parsed_val = parse_arg(key+'='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), float)

    def test_bool_type(self):
        """Test boolean type."""
        for key, val in [('common.quiet', True), ('model.dropout', False), ('stream.train.float_seed', True)]:
            parsed_key, parsed_val = parse_arg(key+'='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), bool)

    def test_ast_type(self):
        """Test ast type."""
        for key, val in [('common.arch', [1, 2, 3.4, 5]), ('model.arch', {"a": "b"}),
                         ('stream.train.deep', {"a": {"b": ["c", "d", "e"]}}),
                         ('model.arch', 12), ('model.arch', 12.2)]:
            parsed_key, parsed_val = parse_arg(key+'='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), type(val))


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
    """Test case for load_config, yaml_to_file and yaml_to_str functions."""

    def test_load_anchorless_config(self):
        """Test loading of a config without yaml anchors."""

        f_name = path.join(self.tmpdir, 'conf.yaml')

        with open(f_name, 'w') as file:
            file.write(_TEST_ANCHORLESS_YAML)

        self.assertDictEqual(load_config(f_name, []), {'e': {'f': 'f', 'h': ['j', 'k']}})
        self.assertDictEqual(load_config(f_name, ['e.f=12']), {'e': {'f': 12, 'h': ['j', 'k']}})
        self.assertDictEqual(load_config(f_name, ['e.x=12']), {'e': {'f': 'f', 'h': ['j', 'k'], 'x': 12}})

    def test_load_anchored_config(self):
        """Test loading of a config with yaml anchors."""
        f_name = path.join(self.tmpdir, 'conf.yaml')

        with open(f_name, 'w') as file:
            file.write(_TEST_ANCHORED_YAML)

        self.assertDictEqual(load_config(f_name, [])['a'], {'b': 'c', 'd': 11})
        self.assertEqual(OrderedDict(load_config(f_name, [])['e']), OrderedDict([('f', 'f'), ('h', ['j', 'k']),
                                                                                 ('b', 'c'), ('d', 11)]))

        self.assertDictEqual(load_config(f_name, ['a.b=12'])['a'], {'b': 12, 'd': 11})
        self.assertEqual(OrderedDict(load_config(f_name, ['a.b=12'])['e']),
                         OrderedDict([('f', 'f'), ('h', ['j', 'k']), ('b', 12), ('d', 11)]))

        self.assertDictEqual(load_config(f_name, ['e.b=19'])['a'], {'b': 'c', 'd': 11})
        self.assertEqual(OrderedDict(load_config(f_name, ['e.b=19'])['e']),
                         OrderedDict([('f', 'f'), ('h', ['j', 'k']), ('b', 19), ('d', 11)]))
