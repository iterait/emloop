"""
Test module for cxflow common functions (cli/common.py).
"""
import os
from os import path
from copy import deepcopy
from typing import Mapping, List

import yaml

from cxflow import AbstractModel
from cxflow.cli.common import create_output_dir, create_dataset, create_hooks, create_model
from cxflow.hooks.abstract_hook import AbstractHook
from cxflow.hooks.log_profile_hook import LogProfile
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


class DummyModel(AbstractModel):
    """Dummy model which serves as a placeholder instead of regular model implementation."""
    def __init__(self, io: dict, **kwargs):  # pylint: disable=unused-argument
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
    def restore_fallback(self) -> str:
        return ''


class DummyModelWithKwargs(DummyModel):
    """Dummy model which saves kwargs to self.kwargs."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def _create_model(self, **kwargs):  # pylint: disable=unused-argument
        # create a dummy train op and variable
        pass


class DummyModelWithKwargs2(DummyModelWithKwargs):
    """Direct inheritor of `DummyModelWithKwargs`.

    For restoring purposes only."""
    pass


class CLICommonTest(CXTestCaseWithDir):
    """cli common test case."""

    def test_create_output_dir(self):
        """Test output dir creating and correct naming."""
        # test create output dir with specified model.name
        name = 'my_name'
        output_dir = create_output_dir(config={'a': 'b', 'model': {'name': name}},
                                       output_root=self.tmpdir,
                                       default_model_name='nothing')

        self.assertEqual(len(os.listdir(self.tmpdir)), 1)
        self.assertEqual(output_dir, path.join(self.tmpdir, os.listdir(self.tmpdir)[0]))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

    def test_create_output_dir_no_root(self):
        """Test if output root is created if it does not exist."""
        output_root = path.join(self.tmpdir, 'output_root')
        name = 'my_name'
        output_dir = create_output_dir(config={'a': 'b', 'model': {'name': name}},
                                       output_root=output_root,
                                       default_model_name='nothing')

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
        """Test create output dir without specified model.name (default_model_name should be used)."""
        name = 'nothing'
        output_dir = create_output_dir(config={'a': 'b', 'model': {}},
                                       output_root=self.tmpdir,
                                       default_model_name=name)

        self.assertEqual(len(os.listdir(self.tmpdir)), 1)
        self.assertEqual(output_dir, path.join(self.tmpdir, os.listdir(self.tmpdir)[0]))
        self.assertTrue(path.exists(output_dir))
        self.assertTrue(path.isdir(output_dir))
        self.assertTrue(name in output_dir)

    def test_different_dirs(self):
        """Test if two calls of train_create_output_dir yields two different dirs."""
        name = 'my_name'
        output_dir_1 = create_output_dir(config={'a': 'b', 'model': {'name': name}},
                                         output_root=self.tmpdir,
                                         default_model_name='nothing')
        output_dir_2 = create_output_dir(config={'a': 'b', 'model': {'name': name}},
                                         output_root=self.tmpdir,
                                         default_model_name='nothing')

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
        config = {'hooks': [{'cxflow.tests.cli.common_test.DummyHook': {'additional_arg': 10}}]}
        dataset = 'dataset_placeholder'
        model = 'model_placeholder'
        expected_kwargs = {'dataset': dataset, 'model': model, 'output_dir': self.tmpdir, 'additional_arg': 10}
        hooks = create_hooks(config=config, dataset=dataset, model=model, output_dir=self.tmpdir)
        hook = hooks[0]
        kwargs = hook.kwargs

        self.assertEqual(len(hooks), 1)
        self.assertTrue(isinstance(hook, DummyHook))
        for key in expected_kwargs:
            self.assertIn(key, kwargs)
            self.assertEqual(expected_kwargs[key], kwargs[key])

        # test correct hook order and hook config with no additional args
        two_hooks_config = {'hooks': [{'cxflow.tests.cli.common_test.DummyHook': {'additional_arg': 10}},
                                      'cxflow.tests.cli.common_test.SecondDummyHook']}
        hooks2 = create_hooks(config=two_hooks_config, dataset=dataset, model=model, output_dir=self.tmpdir)

        self.assertEqual(len(hooks2), 2)
        self.assertTrue(isinstance(hooks2[0], DummyHook))
        self.assertTrue(isinstance(hooks2[1], SecondDummyHook))

        # test module inference
        auto_module_hooks_config = {'hooks': ['LogProfile']}
        hooks3 = create_hooks(config=auto_module_hooks_config, dataset=dataset, model=model, output_dir=self.tmpdir)

        self.assertEqual(len(hooks3), 1)
        self.assertTrue(isinstance(hooks3[0], LogProfile))

        # test non existent class
        bad_hooks_config = {'hooks': ['IDoNotExist']}
        self.assertRaises(ValueError, create_hooks,
                          config=bad_hooks_config, dataset=dataset, model=model, output_dir=self.tmpdir)

    def test_create_model(self):
        """Test if model is created correctly."""

        # test correct kwargs passing
        config = {'model': {'module': 'cxflow.tests.cli.common_test',
                            'class': 'DummyModelWithKwargs',
                            'io': {'in': [], 'out': ['dummy']}}}
        dataset = 'dataset_placeholder'
        expected_kwargs = {'dataset': dataset, 'log_dir': self.tmpdir, **config['model']}
        model = create_model(config=config, output_dir=self.tmpdir, dataset=dataset)
        model.save('dummy')

        kwargs = model.kwargs  # pylint: disable=no-member
        del expected_kwargs['module']
        del expected_kwargs['class']

        for key in expected_kwargs.keys():
            self.assertIn(key, kwargs)
            self.assertEqual(expected_kwargs[key], kwargs[key])

        # test restoring when the model class is found
        restored_model = create_model(config=config, output_dir=self.tmpdir + '_restored', dataset=dataset,
                                      restore_from=self.tmpdir)
        self.assertTrue(isinstance(restored_model, DummyModelWithKwargs))

        # test restoring when the model class is not found
        new_config = deepcopy(config)
        new_config['model']['class'] = 'IDontExist'
        new_config['model']['restore_fallback'] = 'cxflow.tests.cli.common_test.DummyModelWithKwargs2'
        restored_model = create_model(config=new_config, output_dir=self.tmpdir + '_restored', dataset=dataset,
                                      restore_from=self.tmpdir)
        self.assertTrue(isinstance(restored_model, DummyModelWithKwargs2))
