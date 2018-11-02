"""
Test module for **emloop** common functions (cli/common.py).
"""
import os
from os import path
from copy import deepcopy
from typing import Mapping, List
import logging

import yaml
import pytest

from emloop import AbstractModel
from emloop.cli.common import create_output_dir, create_dataset, create_hooks, create_model, run
from emloop.hooks.abstract_hook import AbstractHook
from emloop.hooks.log_profile import LogProfile
from emloop.hooks import StopAfter
from emloop.datasets import AbstractDataset, StreamWrapper


class DummyDataset:
    """Dummy dataset which loads the given config to self.config."""
    def __init__(self, config_str):
        self.config = yaml.load(config_str)


class DummyConfigDataset(AbstractDataset):
    """Dummy dataset which changes config."""
    def __init__(self, config_str: str):
        super().__init__(config_str)
        config = yaml.load(config_str)
        self.train = {'a': 'b'}
        self._configure_dataset(**config)

    def _configure_dataset(self, dataset_config: List[str], **kwargs):
        dataset_config[0], dataset_config[1], dataset_config[2] = \
            dataset_config[1], dataset_config[0], dataset_config[2]

    def train_stream(self):
        yield self.train


class DummyHook(AbstractHook):
    """Dummy hook which save its ``**kwargs`` to ``self.kwargs``."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)


class SecondDummyHook(AbstractHook):
    """Second dummy dataset which does nothing."""
    pass


class DummyConfigHook(AbstractHook):
    """Dummy hook which changes config."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(**kwargs)
        self.change_config(**kwargs)

    def change_config(self, variables: List[str], **kwargs):
        variables[0], variables[1] = variables[1], variables[0]


class DummyModel(AbstractModel):
    """Dummy model which serves as a placeholder instead of regular model implementation."""
    def __init__(self, io: dict, **kwargs):  # pylint: disable=unused-argument
        self._input_names = io['in']
        self._output_names = io['out']
        super().__init__(**kwargs)

    def run(self, batch: Mapping[str, object], train: bool, stream: StreamWrapper) -> Mapping[str, object]:
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


class DummyConfigModel(DummyModel):
    """Dummy model which changes config."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._create_model(**kwargs)
        super().__init__(**kwargs)

    def _create_model(self, architecture: Mapping, **kwargs):
        config = architecture['model_config']
        config[0], config[1], config[2], config[3] = config[1], config[0], config[3], config[2]


def test_create_output_dir(tmpdir):
    """Test output dir creating and correct naming."""
    # test create output dir with specified model.name
    name = 'my_name'
    output_dir = create_output_dir(config={'a': 'b', 'model': {'name': name}},
                                   output_root=tmpdir,
                                   default_model_name='nothing')

    assert len(os.listdir(tmpdir)) == 1
    assert output_dir == path.join(tmpdir, os.listdir(tmpdir)[0])
    assert path.exists(output_dir)
    assert path.isdir(output_dir)
    assert name in output_dir


def test_create_output_dir_no_root(tmpdir):
    """Test if output root is created if it does not exist."""
    output_root = path.join(tmpdir, 'output_root')
    name = 'my_name'
    output_dir = create_output_dir(config={'a': 'b', 'model': {'name': name}},
                                   output_root=output_root,
                                   default_model_name='nothing')

    # check that output_root exists and it is the only folder in temp_dir
    assert len(os.listdir(tmpdir)) == 1
    assert path.exists(output_root)
    assert path.isdir(output_root)
    # check that output_dir exists and it is the only folder in output_root
    assert len(os.listdir(output_root)) == 1
    assert output_dir == path.join(output_root, path.basename(output_dir))
    assert path.exists(output_dir)
    assert path.isdir(output_dir)
    assert name in output_dir


def test_create_output_dir_noname(tmpdir):
    """Test create output dir without specified model.name (default_model_name should be used)."""
    name = 'nothing'
    output_dir = create_output_dir(config={'a': 'b', 'model': {}},
                                   output_root=tmpdir,
                                   default_model_name=name)

    assert len(os.listdir(tmpdir)) == 1
    assert output_dir == path.join(tmpdir, os.listdir(tmpdir)[0])
    assert path.exists(output_dir)
    assert path.isdir(output_dir)
    assert name in output_dir


def test_different_dirs(tmpdir):
    """Test if two calls of train_create_output_dir yields two different dirs."""
    name = 'my_name'
    output_dir_1 = create_output_dir(config={'a': 'b', 'model': {'name': name}},
                                     output_root=tmpdir,
                                     default_model_name='nothing')
    output_dir_2 = create_output_dir(config={'a': 'b', 'model': {'name': name}},
                                     output_root=tmpdir,
                                     default_model_name='nothing')

    assert output_dir_1 != output_dir_2
    assert len(os.listdir(tmpdir)) == 2


def test_create_dataset(tmpdir):
    """Test correct config re-wrapping."""
    config = {'dataset': {'class': 'emloop.tests.cli.common_test.DummyDataset', 'batch_size': 10},
              'stream': {'train': {'rotate': 20}}, 'hooks': [{'hook_name': 'should_not_be_included'}]}

    expected_config = {'batch_size': 10, 'output_dir': 'dummy_dir'}

    dataset = create_dataset(config=config, output_dir='dummy_dir')

    assert isinstance(dataset, DummyDataset)
    assert hasattr(dataset, 'config')
    assert dataset.config == expected_config


def test_create_hooks(tmpdir, caplog):
    """Test hooks creation in :py:class:`emloop.cli.create_hooks."""

    # test correct kwargs passing
    config = {'hooks': [{'emloop.tests.cli.common_test.DummyHook': {'additional_arg': 10}}]}
    dataset = 'dataset_placeholder'
    model = 'model_placeholder'
    expected_kwargs = {'dataset': dataset, 'model': model, 'output_dir': tmpdir, 'additional_arg': 10}
    hooks = create_hooks(config=config, dataset=dataset, model=model, output_dir=tmpdir)
    hook = hooks[0]
    kwargs = hook.kwargs

    assert len(hooks) == 1
    assert isinstance(hook, DummyHook)
    for key in expected_kwargs:
        assert key in kwargs
        assert expected_kwargs[key] == kwargs[key]

    # test correct hook order and hook config with no additional args
    two_hooks_config = {'hooks': [{'emloop.tests.cli.common_test.DummyHook': {'additional_arg': 10}},
                                  'emloop.tests.cli.common_test.SecondDummyHook']}
    hooks2 = create_hooks(config=two_hooks_config, dataset=dataset, model=model, output_dir=tmpdir)

    assert len(hooks2) == 2
    assert isinstance(hooks2[0], DummyHook)
    assert isinstance(hooks2[1], SecondDummyHook)

    # test module inference
    auto_module_hooks_config = {'hooks': ['LogProfile']}
    hooks3 = create_hooks(config=auto_module_hooks_config, dataset=dataset, model=model, output_dir=tmpdir)

    assert len(hooks3) == 1
    assert isinstance(hooks3[0], LogProfile)

    # test non existent class
    bad_hooks_config = {'hooks': ['IDoNotExist']}
    with pytest.raises(ValueError):
        create_hooks(config=bad_hooks_config, dataset=dataset, model=model, output_dir=tmpdir)

    empty_params_hooks_config = {'hooks': [{'emloop.tests.cli.common_test.DummyHook': None}]}
    caplog.clear()
    caplog.set_level(logging.INFO)
    hooks4 = create_hooks(config=empty_params_hooks_config, dataset=dataset, model=model, output_dir=tmpdir)
    assert len(hooks4) == 1
    assert isinstance(hooks4[0], DummyHook)
    assert caplog.record_tuples == [
        ('root', logging.INFO, 'Creating hooks'),
        ('root', logging.WARNING, '\t\t Empty config of `emloop.tests.cli.common_test.DummyHook` hook'),
        ('root', logging.INFO, '\tDummyHook created')
    ]


def test_create_model(tmpdir):
    """Test if model is created correctly."""

    # test correct kwargs passing
    config = {'model': {'class': 'emloop.tests.cli.common_test.DummyModelWithKwargs',
                        'io': {'in': [], 'out': ['dummy']}}}
    dataset = 'dataset_placeholder'
    expected_kwargs = {'dataset': dataset, 'log_dir': tmpdir, **config['model']}
    model = create_model(config=config, output_dir=tmpdir, dataset=dataset)
    model.save('dummy')

    kwargs = model.kwargs  # pylint: disable=no-member
    del expected_kwargs['class']

    for key in expected_kwargs.keys():
        assert key in kwargs
        assert expected_kwargs[key] == kwargs[key]

    # test restoring when the model class is found
    restored_model = create_model(config=config, output_dir=tmpdir + '_restored', dataset=dataset,
                                  restore_from=tmpdir)
    assert isinstance(restored_model, DummyModelWithKwargs)

    # test restoring when the model class is not found
    new_config = deepcopy(config)
    new_config['model']['class'] = 'nonexistingmodule.IDontExist'
    new_config['model']['restore_fallback'] = 'emloop.tests.cli.common_test.DummyModelWithKwargs2'
    restored_model = create_model(config=new_config, output_dir=tmpdir + '_restored', dataset=dataset,
                                  restore_from=tmpdir)
    assert isinstance(restored_model, DummyModelWithKwargs2)


def test_config_file_is_unchanged(tmpdir):
    """Test that config file is not changed during training."""

    config = {'dataset': {'class': 'emloop.tests.cli.common_test.DummyConfigDataset', 'batch_size': 10,
                                   'dataset_config': ['a', 'b', 'c']},
              'stream': {'train': {'rotate': 20}},
              'hooks': [{'emloop.tests.cli.common_test.DummyConfigHook': {'additional_arg': 10,
                                                                          'variables': ['a', 'b']}},
                        {'StopAfter': {'epochs': 1}}],
              'model': {'class': 'emloop.tests.cli.common_test.DummyConfigModel',
                                 'architecture': {'model_config': ['a', 'b', 'c', 'd']},
                                 'io': {'in': [], 'out': ['dummy']}}}

    run(config=config, output_root=tmpdir)

    assert config['dataset']['dataset_config'] == ['a', 'b', 'c']
    assert config['hooks'][0]['emloop.tests.cli.common_test.DummyConfigHook']['variables'] == ['a', 'b']
    assert config['model']['architecture']['model_config'] == ['a', 'b', 'c', 'd']
