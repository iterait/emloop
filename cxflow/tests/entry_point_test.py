from cxflow.entry_point import _train_create_output_dir, _train_create_dataset, _train_load_config
from cxflow.utils.config import config_to_file, load_config

import logging
import os
from os import path
import tempfile
from unittest import TestCase
import shutil
import yaml


class DummyDataset:
    def __init__(self, config_str):
        self.config = yaml.load(config_str)

class EntryPointTest(TestCase):
    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_train_create_output_dir(self):
        temp_dir = tempfile.mkdtemp()
        name = 'my_name'
        output_dir = _train_create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                              output_root=temp_dir,
                                              default_net_name='nothing')

        self.assertEqual(len(os.listdir(temp_dir)), 1)
        self.assertEqual(output_dir, path.join(temp_dir, os.listdir(temp_dir)[0]))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

        shutil.rmtree(temp_dir)

    def test_create_output_dir_no_root(self):
        # check that output root is created if it does not exist
        temp_dir = tempfile.mkdtemp()
        output_root = path.join(temp_dir, 'output_root')
        name = 'my_name'
        output_dir = _train_create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                              output_root=output_root,
                                              default_net_name='nothing')

        # check that output_root exists and it is the only folder in temp_dir
        self.assertEqual(len(os.listdir(temp_dir)), 1)
        self.assertTrue(path.exists(output_root))
        self.assertTrue(path.isdir(output_root))
        # check that output_dir exists and it is the only folder in output_root
        self.assertEqual(len(os.listdir(output_root)), 1)
        self.assertEqual(output_dir, path.join(output_root, path.basename(output_dir)))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

        shutil.rmtree(temp_dir)

    def test_train_create_output_dir_without_net_name(self):
        temp_dir = tempfile.mkdtemp()
        name = 'nothing'
        output_dir = _train_create_output_dir(config={'a': 'b', 'net': {}},
                                              output_root=temp_dir,
                                              default_net_name=name)

        self.assertEqual(len(os.listdir(temp_dir)), 1)
        self.assertEqual(output_dir, path.join(temp_dir, os.listdir(temp_dir)[0]))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

        shutil.rmtree(temp_dir)

    def test_different_dirs(self):
        temp_dir = tempfile.mkdtemp()
        name = 'my_name'
        output_dir_1 = _train_create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                                output_root=temp_dir,
                                                default_net_name='nothing')
        output_dir_2 = _train_create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                                output_root=temp_dir,
                                                default_net_name='nothing')

        self.assertNotEqual(output_dir_1, output_dir_2)
        self.assertEqual(len(os.listdir(temp_dir)), 2)
        shutil.rmtree(temp_dir)

    def test_train_create_dataset(self):
        config = {'dataset': {'module': 'cxflow.tests.entry_point_test', 'class': 'DummyDataset', 'batch_size': 10},
                  'stream': {'train': {'rotate': 20}}, 'hooks': [{'hook_name': 'should_not_be_included'}]}

        expected_config = {'dataset': {'module': 'cxflow.tests.entry_point_test',
                                       'class': 'DummyDataset', 'batch_size': 10},
                           'stream': {'train': {'rotate': 20}}, 'output_dir': 'dummy_dir'}

        dataset = _train_create_dataset(config=config, output_dir='dummy_dir')

        self.assertTrue(isinstance(dataset, DummyDataset))
        self.assertTrue(hasattr(dataset, 'config'))
        self.assertDictEqual(dataset.config, expected_config)

    def test_train_load_config(self):
        temp_dir = tempfile.mkdtemp()

        # test a config call with both dataset and net
        good_config = {'dataset': None, 'net': None}
        config_path = config_to_file(good_config, temp_dir)

        # test assertion when config is incomplete
        missing_net_config = {'dataset': None}
        config_path2 = config_to_file(missing_net_config, temp_dir, 'config2.yaml')
        self.assertRaises(AssertionError, _train_load_config, config_path2, [])

        missing_dataset_config = {'dataset': None}
        config_path3 = config_to_file(missing_dataset_config, temp_dir, 'config3.yaml')
        self.assertRaises(AssertionError, _train_load_config, config_path3, [])

        # test return value
        returned_config = _train_load_config(config_path, [])
        self.assertDictEqual(returned_config, load_config(config_path, []))
        self.assertDictEqual(returned_config, good_config)

        shutil.rmtree(temp_dir)

