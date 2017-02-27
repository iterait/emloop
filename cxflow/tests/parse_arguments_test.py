from ..entry_point import EntryPoint

import logging
from unittest import TestCase


class ParseArgumentsTest(TestCase):
    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_default_type(self):
        for key, val in [('common.name', 'BatchSize1'), ('net.name', 'netie'), ('stream.train.seed', 'none')]:
            parsed_key, parsed_val = EntryPoint._parse_arg(key+'='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), str)

    def test_str_type(self):
        for key, val in [('common.name', 'BatchSize1'), ('net.name', 'netie'), ('stream.train.seed', 'none')]:
            parsed_key, parsed_val = EntryPoint._parse_arg(key+':str='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), str)

    def test_int_type(self):
        for key, val in [('common.batch_size', 12), ('net.dropout', 0), ('stream.train.seed', 123)]:
            parsed_key, parsed_val = EntryPoint._parse_arg(key+':int='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), int)

        parsed_key, parsed_val = EntryPoint._parse_arg('common.batch_size:int=12.7')
        self.assertTupleEqual(('common.batch_size', 12), (parsed_key, parsed_val))
        self.assertEqual(type(parsed_val), int)

    def test_float_type(self):
        for key, val in [('common.some_int_number', 12), ('net.dropout', 0.5), ('stream.train.float_seed', 123.456)]:
            parsed_key, parsed_val = EntryPoint._parse_arg(key+':float='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), float)

    def test_bool_type(self):
        for key, val in [('common.quiet', 1), ('net.dropout', 0), ('stream.train.float_seed', 1)]:
            parsed_key, parsed_val = EntryPoint._parse_arg(key+':bool='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), bool)

    def test_ast_type(self):
        for key, val in [('common.arch', [1,2,3.4,5]), ('net.arch', {"a": "b"}), ('stream.train.deep', {"a": {"b": ["c", "d", "e"]}}), ('net.arch', 12), ('net.arch', 12.2)]:
            parsed_key, parsed_val = EntryPoint._parse_arg(key+':ast='+str(val))
            self.assertTupleEqual((key, val), (parsed_key, parsed_val))
            self.assertEqual(type(parsed_val), type(val))

    def test_not_int_type(self):
        for key, val in [('common.batch_size', "ahoj"), ('stream.train.seed', [1, 2])]:
            self.assertRaises(AttributeError, EntryPoint._parse_arg, key+':int='+str(val))

    def test_not_float_type(self):
        for key, val in [('common.some_number', True), ('net.dropout', "hello"), ('stream.train.float_seed', [1, 2])]:
            self.assertRaises(AttributeError, EntryPoint._parse_arg, key+':float='+str(val))

    def test_not_bool_type(self):
        for key, val in [('common.quiet', "hello"), ('net.dropout', 0.2), ('stream.train.float_seed', 13), ('stream.train.float_seed', [1, 3])]:
            try:
                self.assertRaises(AttributeError, EntryPoint._parse_arg, key+':bool='+str(val))
            except Exception as e:
                print(type(e))

    def test_not_ast_type(self):
        for key, val in [('common.arch', "hello"), ('net.arch', '[12,3'), ('net.arch', '{"a": }')]:
                self.assertRaises(AttributeError, EntryPoint._parse_arg, key + ':ast=' + str(val))
