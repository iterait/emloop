"""
Test module for cxflow common functions (cli/common.py).
"""
import os
from os import path
from copy import deepcopy
from typing import Mapping, List

import yaml

from cxflow import AbstractNet
from cxflow.cli.common import create_output_dir, create_dataset, create_hooks, create_net
from cxflow.hooks.abstract_hook import AbstractHook
from cxflow.hooks.profile_hook import ProfileHook
from cxflow.tests.test_core import CXTestCaseWithDir


class DummyDataset:
    """Dummy dataset which loads the given config to self.config."""
    def __init__(self, config_str):
        self.config = yaml.load(config_str)


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
        super().__init__(**kwargs)

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

    @property
    def restore_fallback_module(self) -> str:
        return ''

    @property
    def restore_fallback_class(self) -> str:
        return ''


class DummyNetWithKwargs(DummyNet):
    """Dummy net which saves kwargs to self.kwargs."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def _create_net(self, **kwargs):  #pylint: disable=unused-argument
        # create a dummy train op and variable
        pass


class DummyNetWithKwargs2(DummyNetWithKwargs):
    """Direct inheritor of `DummyNetWithKwargs`.

    For restoring purposes only."""
    pass


class CLICommonTest(CXTestCaseWithDir):
    """cli common test case."""

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
        config = {'dataset': {'module': 'cxflow.tests.cli.common_test', 'class': 'DummyDataset', 'batch_size': 10},
                  'stream': {'train': {'rotate': 20}}, 'hooks': [{'hook_name': 'should_not_be_included'}]}

        expected_config = {'module': 'cxflow.tests.cli.common_test',
                           'class': 'DummyDataset', 'batch_size': 10, 'output_dir': 'dummy_dir'}

        dataset = create_dataset(config=config, output_dir='dummy_dir')

        self.assertTrue(isinstance(dataset, DummyDataset))
        self.assertTrue(hasattr(dataset, 'config'))
        self.assertDictEqual(dataset.config, expected_config)

    def test_create_hooks(self):
        """Test hooks creation in train_create_hooks."""

        # test correct kwargs passing
        config = {'hooks': [{'module': 'cxflow.tests.cli.common_test',
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
        two_hooks_config = {'hooks': [{'module': 'cxflow.tests.cli.common_test',
                                       'class': 'DummyHook',
                                       'additional_arg': 10},
                                      {'module': 'cxflow.tests.cli.common_test',
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
        config = {'net': {'module': 'cxflow.tests.cli.common_test',
                          'class': 'DummyNetWithKwargs',
                          'io': {'in': [], 'out': ['dummy']}}}
        dataset = 'dataset_placeholder'
        expected_kwargs = {'dataset': dataset, 'log_dir': self.tmpdir, **config['net']}
        net = create_net(config=config, output_dir=self.tmpdir, dataset=dataset)
        checkpoint_path = net.save('dummy')

        kwargs = net.kwargs  # pylint: disable=no-member
        del expected_kwargs['module']
        del expected_kwargs['class']

        for key in expected_kwargs.keys():
            self.assertIn(key, kwargs)
            self.assertEqual(expected_kwargs[key], kwargs[key])

        # test restoring when the net class is found
        restored_net = create_net(config=config, output_dir=self.tmpdir + '_restored', dataset=dataset,
                                  restore_from=self.tmpdir)
        self.assertTrue(isinstance(restored_net, DummyNetWithKwargs))

        # test restoring when the net class is not found
        new_config = deepcopy(config)
        new_config['net']['class'] = 'IDontExist'
        new_config['net']['restore_fallback_module'] = 'cxflow.tests.cli.common_test'
        new_config['net']['restore_fallback_class'] = 'DummyNetWithKwargs2'
        restored_net = create_net(config=new_config, output_dir=self.tmpdir + '_restored', dataset=dataset,
                                  restore_from=self.tmpdir)
        self.assertTrue(isinstance(restored_net, DummyNetWithKwargs2))
