"""
Test module for cxflow entry point (entry_point.py)
"""
import os
from os import path
from copy import deepcopy
from typing import Mapping, List

import yaml

from cxflow import AbstractNet
from cxflow.entry_point import create_output_dir, create_dataset, train_load_config, create_hooks, create_net, split
from cxflow.hooks.abstract_hook import AbstractHook
from cxflow.hooks.profile_hook import ProfileHook
from cxflow.tests.test_core import CXTestCaseWithDir
from cxflow.utils.config import config_to_file, load_config


class DummyDataset:
    """Dummy dataset which loads the given config to self.config."""
    def __init__(self, config_str):
        self.config = yaml.load(config_str)


class SplitDataset:
    """Simple dataset which records its split method calls."""

    def __init__(self, _):
        _SPLIT_DATASET_INSTANCES.append(self)
        self.split_calls = []

    def split(self, num_splits: int, train: float, valid: float, test: float):
        """Record the call arguments."""
        self.split_calls.append({'n': num_splits, 'tr': train, 'v': valid, 'te': test})

_SPLIT_DATASET_INSTANCES = []


class DummyHook(AbstractHook):
    """Dummy hook which save its **kwargs to self.kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)


class SecondDummyHook(AbstractHook):
    """Second dummy dataset which does nothing."""
    pass


class DummyNet(AbstractNet):
    """Dummy net which serves as a placeholder instead of regular net implementation."""
    def __init__(self, io: dict, **kwargs):  #pylint: disable=unused-argument
        self._input_names = io['in']
        self._output_names = io['out']

    def run(self, batch: Mapping[str, object], train: bool) -> Mapping[str, object]:
        return {o: i for i, o in enumerate(self._output_names)}

    def save(self, name_suffix: str) -> str:
        pass

    @property
    def input_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        return self._input_names

    @property
    def output_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        return self._output_names


class DummyNetWithKwargs(DummyNet):
    """Dummy net which saves kwargs to self.kwargs."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def _create_net(self, **kwargs):  #pylint: disable=unused-argument
        # create a dummy train op and variable
        pass


class DummyNetRestore(DummyNetWithKwargs):
    """Dummy restore net."""
    pass


class EntryPointTest(CXTestCaseWithDir):
    """Entry point functions test case."""

    def test_create_output_dir(self):
        """Test output dir creating and correct naming."""
        # test create output dir with specified net.name
        name = 'my_name'
        output_dir = create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                       output_root=self.tmpdir,
                                       default_net_name='nothing')

        self.assertEqual(len(os.listdir(self.tmpdir)), 1)
        self.assertEqual(output_dir, path.join(self.tmpdir, os.listdir(self.tmpdir)[0]))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

    def test_create_output_dir_no_root(self):
        """Test if output root is created if it does not exist."""
        output_root = path.join(self.tmpdir, 'output_root')
        name = 'my_name'
        output_dir = create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                       output_root=output_root,
                                       default_net_name='nothing')

        # check that output_root exists and it is the only folder in temp_dir
        self.assertEqual(len(os.listdir(self.tmpdir)), 1)
        self.assertTrue(path.exists(output_root))
        self.assertTrue(path.isdir(output_root))
        # check that output_dir exists and it is the only folder in output_root
        self.assertEqual(len(os.listdir(output_root)), 1)
        self.assertEqual(output_dir, path.join(output_root, path.basename(output_dir)))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

    def test_create_output_dir_noname(self):
        """Test create output dir without specified net.name (default_net_name should be used)."""
        name = 'nothing'
        output_dir = create_output_dir(config={'a': 'b', 'net': {}},
                                       output_root=self.tmpdir,
                                       default_net_name=name)

        self.assertEqual(len(os.listdir(self.tmpdir)), 1)
        self.assertEqual(output_dir, path.join(self.tmpdir, os.listdir(self.tmpdir)[0]))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

    def test_different_dirs(self):
        """Test if two calls of train_create_output_dir yields two different dirs."""
        name = 'my_name'
        output_dir_1 = create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                         output_root=self.tmpdir,
                                         default_net_name='nothing')
        output_dir_2 = create_output_dir(config={'a': 'b', 'net': {'name': name}},
                                         output_root=self.tmpdir,
                                         default_net_name='nothing')

        self.assertNotEqual(output_dir_1, output_dir_2)
        self.assertEqual(len(os.listdir(self.tmpdir)), 2)

    def test_create_dataset(self):
        """Test correct config re-wrapping."""
        config = {'dataset': {'module': 'cxflow.tests.entry_point_test', 'class': 'DummyDataset', 'batch_size': 10},
                  'stream': {'train': {'rotate': 20}}, 'hooks': [{'hook_name': 'should_not_be_included'}]}

        expected_config = {'dataset': {'module': 'cxflow.tests.entry_point_test',
                                       'class': 'DummyDataset', 'batch_size': 10}, 'output_dir': 'dummy_dir'}

        dataset = create_dataset(config=config, output_dir='dummy_dir')

        self.assertTrue(isinstance(dataset, DummyDataset))
        self.assertTrue(hasattr(dataset, 'config'))
        self.assertDictEqual(dataset.config, expected_config)

    def test_train_load_config(self):
        """Test correct config loading."""

        # test a config call with both dataset and net
        good_config = {'dataset': None, 'net': None}
        config_path = config_to_file(good_config, self.tmpdir)

        # test assertion when config is incomplete
        missing_net_config = {'dataset': None}
        config_path2 = config_to_file(missing_net_config, self.tmpdir, 'config2.yaml')
        self.assertRaises(AssertionError, train_load_config, config_path2, [])

        missing_dataset_config = {'dataset': None}
        config_path3 = config_to_file(missing_dataset_config, self.tmpdir, 'config3.yaml')
        self.assertRaises(AssertionError, train_load_config, config_path3, [])

        # test return value
        returned_config = train_load_config(config_path, [])
        self.assertDictEqual(returned_config, load_config(config_path, []))
        self.assertDictEqual(returned_config, good_config)

    def test_create_hooks(self):
        """Test hooks creation in train_create_hooks."""

        # test correct kwargs passing
        config = {'hooks': [{'module': 'cxflow.tests.entry_point_test',
                             'class':'DummyHook',
                             'additional_arg': 10}]}
        dataset = 'dataset_placeholder'
        net = 'net_placeholder'
        expected_kwargs = {'dataset': dataset, 'net': net, 'output_dir': self.tmpdir, 'additional_arg': 10}
        hooks = create_hooks(config=config, dataset=dataset, net=net, output_dir=self.tmpdir)
        hook = hooks[0]
        kwargs = hook.kwargs

        self.assertEqual(len(hooks), 1)
        self.assertTrue(isinstance(hook, DummyHook))
        for key in expected_kwargs:
            self.assertIn(key, kwargs)
            self.assertEqual(expected_kwargs[key], kwargs[key])

        # test correct hook order
        two_hooks_config = {'hooks': [{'module': 'cxflow.tests.entry_point_test',
                                       'class': 'DummyHook',
                                       'additional_arg': 10},
                                      {'module': 'cxflow.tests.entry_point_test',
                                       'class': 'SecondDummyHook'}]}
        hooks2 = create_hooks(config=two_hooks_config, dataset=dataset, net=net, output_dir=self.tmpdir)

        self.assertEqual(len(hooks2), 2)
        self.assertTrue(isinstance(hooks2[0], DummyHook))
        self.assertTrue(isinstance(hooks2[1], SecondDummyHook))

        # test auto module
        auto_module_hooks_config = {'hooks': [{'class': 'ProfileHook'}]}
        hooks3 = create_hooks(config=auto_module_hooks_config, dataset=dataset, net=net, output_dir=self.tmpdir)

        self.assertEqual(len(hooks3), 1)
        self.assertTrue(isinstance(hooks3[0], ProfileHook))

        # test bad auto module
        bad_hooks_config = {'hooks': [{'class': 'IDoNotExist'}]}
        self.assertRaises(ValueError, create_hooks,
                          config=bad_hooks_config, dataset=dataset, net=net, output_dir=self.tmpdir)

    def test_create_net(self):
        """Test if net is created correctly."""

        # test correct kwargs passing
        config = {'net': {'module': 'cxflow.tests.entry_point_test',
                          'class': 'DummyNetWithKwargs',
                          'io': {'in': [], 'out': ['dummy']}}}
        dataset = 'dataset_placeholder'
        expected_kwargs = {'dataset': dataset, 'log_dir': self.tmpdir, **config['net']}
        net = create_net(config=config, output_dir=self.tmpdir, dataset=dataset)
        checkpoint_path = net.save('dummy')

        kwargs = net.kwargs  # pylint: disable=no-member
        for key in expected_kwargs.keys():
            self.assertIn(key, kwargs)
            self.assertEqual(expected_kwargs[key], kwargs[key])

        # See issue #50 and #51
        # test net restore without custom restore class
        # restore_config = deepcopy(config)
        # restore_config['net']['restore_from'] = checkpoint_path
        # restored_net = create_net(config=restore_config, output_dir=self.tmpdir + '_restored', dataset=dataset)
        #
        # self.assertTrue(isinstance(restored_net, BaseTFNetRestore))
        # tf.reset_default_graph()

        # # test net restore with custom restore class
        custom_restore_config = deepcopy(config)
        custom_restore_config['net']['restore_from'] = checkpoint_path
        custom_restore_config['net']['restore_module'] = 'cxflow.tests.entry_point_test'
        custom_restore_config['net']['restore_class'] = 'DummyNetRestore'
        restored_net = create_net(custom_restore_config, output_dir=self.tmpdir + '_restored', dataset=dataset)

        self.assertTrue(isinstance(restored_net, DummyNetRestore))

    def test_split(self):
        """Test if split creates the dataset and calls the split function properly."""
        config = {'dataset': {'module': 'cxflow.tests.entry_point_test', 'class': 'SplitDataset'}}
        config_file = config_to_file(config, self.tmpdir, 'config.yaml')
        split(config_file, 7, 5, 3, 1)

        self.assertEqual(len(_SPLIT_DATASET_INSTANCES), 1)
        dataset = _SPLIT_DATASET_INSTANCES[0]
        self.assertIsInstance(dataset, SplitDataset)
        self.assertEqual(len(dataset.split_calls), 1)
        self.assertDictEqual(dataset.split_calls[0], {'n': 7, 'tr': 5, 'v': 3, 'te': 1})
