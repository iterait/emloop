from cxflow.entry_point import _train_create_output_dir

import logging
import os
from os import path
import tempfile
from unittest import TestCase
import shutil


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
