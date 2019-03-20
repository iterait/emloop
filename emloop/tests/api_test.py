"""
Test module for **emloop** common functions (api/common.py).
"""
import os
from os import path
from copy import deepcopy
from typing import Mapping, List
import logging
import datetime

import ruamel.yaml
import pytest

from emloop import AbstractModel, MainLoop
from emloop.api import create_output_dir, create_dataset, create_hooks, create_model, create_main_loop
from emloop.hooks.abstract_hook import AbstractHook
from emloop.hooks import StopAfter, LogProfile
from emloop.hooks.training_trace import TrainingTraceKeys
from emloop.datasets import AbstractDataset, StreamWrapper
from emloop.constants import EL_DEFAULT_TRAIN_STREAM, EL_TRACE_FILE
from emloop.types import TimeProfile
from emloop.utils.yaml import load_yaml


class DummyDataset:
    """Dummy dataset which loads the given config to self.config."""
    def __init__(self, config_str):
        self.config = ruamel.yaml.load(config_str)

    def train_stream(self):
        yield {'a': ['b']}


class DummyConfigDataset(AbstractDataset):
    """Dummy dataset which changes config."""
    def __init__(self, config_str: str):
        super().__init__(config_str)
        config = ruamel.yaml.load(config_str)['dataset_config']
        config[0], config[1], config[2] = config[1], config[0], config[2]

    def train_stream(self):
        yield {'a': ['b']}


class DummyEvalDataset(AbstractDataset):
    """Dummy dataset with valid_stream method."""
    def __init__(self, config_str: str):
        super().__init__(config_str)

    def train_stream(self):
        pass

    def valid_stream(self):
        yield {'a': ['b']}


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
    def __init__(self, variables: List[str], **kwargs):
        super().__init__(**kwargs)
        variables[0], variables[1] = variables[1], variables[0]


class DummyEvalHook(AbstractHook):
    """Dummy hook that checks the `after_epoch_profile` is evaluated on `valid` stream."""
    def after_epoch_profile(self, epoch_id: int, profile: TimeProfile, streams: List[str]) -> None:
        """Checks passed in streams parameter is correct."""
        assert streams == ['valid']


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
    def __init__(self, architecture: Mapping, **kwargs):
        super().__init__(**kwargs)
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
    config = {'dataset': {'class': 'emloop.tests.api_test.DummyDataset', 'batch_size': 10},
              'stream': {'train': {'rotate': 20}}, 'hooks': [{'hook_name': 'should_not_be_included'}]}

    expected_config = {'batch_size': 10, 'output_dir': 'dummy_dir'}

    dataset = create_dataset(config=config, output_dir='dummy_dir')

    assert isinstance(dataset, DummyDataset)
    assert hasattr(dataset, 'config')
    assert dataset.config == expected_config


def test_create_hooks(tmpdir, caplog):
    """Test hooks creation in :py:class:`emloop.api.create_hooks."""

    # test correct kwargs passing
    config = {'hooks': [{'emloop.tests.api_test.DummyHook': {'additional_arg': 10}}, 'TrainingTrace']}
    dataset = 'dataset_placeholder'
    model = 'model_placeholder'
    expected_kwargs = {'dataset': dataset, 'model': model, 'output_dir': tmpdir, 'additional_arg': 10}
    hooks = create_hooks(config=config, dataset=dataset, model=model, output_dir=tmpdir)
    hook = hooks[0]
    kwargs = hook.kwargs

    assert len(hooks) == 2
    assert isinstance(hook, DummyHook)
    for key in expected_kwargs:
        assert key in kwargs
        assert expected_kwargs[key] == kwargs[key]

    # test correct hook order and hook config with no additional args
    two_hooks_config = {'hooks': [{'emloop.tests.api_test.DummyHook': {'additional_arg': 10}},
                                   'emloop.tests.api_test.SecondDummyHook', 'TrainingTrace']}
    hooks2 = create_hooks(config=two_hooks_config, dataset=dataset, model=model, output_dir=tmpdir)

    assert len(hooks2) == 3
    assert isinstance(hooks2[0], DummyHook)
    assert isinstance(hooks2[1], SecondDummyHook)

    # test module inference
    auto_module_hooks_config = {'hooks': ['LogProfile', 'TrainingTrace']}
    hooks3 = create_hooks(config=auto_module_hooks_config, dataset=dataset, model=model, output_dir=tmpdir)

    assert len(hooks3) == 2
    assert isinstance(hooks3[0], LogProfile)

    # test non existent class
    bad_hooks_config = {'hooks': ['IDoNotExist']}
    with pytest.raises(ModuleNotFoundError):
        create_hooks(config=bad_hooks_config, dataset=dataset, model=model, output_dir=tmpdir)

    empty_params_hooks_config = {'hooks': [{'emloop.tests.api_test.DummyHook': None}, 'TrainingTrace']}
    caplog.clear()
    caplog.set_level(logging.INFO)
    hooks4 = create_hooks(config=empty_params_hooks_config, dataset=dataset, model=model, output_dir=tmpdir)
    assert len(hooks4) == 2
    assert isinstance(hooks4[0], DummyHook)

    assert ('root', logging.WARNING, '\t\t Empty config of `emloop.tests.api_test.DummyHook` hook') \
      in caplog.record_tuples


def test_create_model(tmpdir):
    """Test if model is created correctly."""

    # test correct kwargs passing
    config = {'model': {'class': 'emloop.tests.api_test.DummyModelWithKwargs',
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


def test_config_file_is_unchanged(tmpdir):
    """
    Test that config file is not changed during training.
    Regarding issue #31: Config YAML in log dir differs https://github.com/iterait/emloop/issues/31
    """

    orig_config = {'dataset': {'class': 'emloop.tests.api_test.DummyConfigDataset', 'batch_size': 10,
                               'dataset_config': ['a', 'b', 'c']},
                   'stream': {'train': {'rotate': 20}},
                   'hooks': [{'emloop.tests.api_test.DummyConfigHook': {'additional_arg': 10, 'variables': ['a', 'b']}},
                             {'StopAfter': {'epochs': 1}}],
                   'model': {'class': 'emloop.tests.api_test.DummyConfigModel',
                                      'architecture': {'model_config': ['a', 'b', 'c', 'd']},
                                      'io': {'in': [], 'out': ['dummy']}}}
    config = deepcopy(orig_config)

    create_main_loop(config=config, output_root=tmpdir).run_training()

    assert orig_config == config


def test_config_file_is_incorrect(tmpdir, caplog):
    """Test that incorrect config file raises error and subsequent system exit."""

    # incorrect dataset arguments
    config = {'dataset': {'class': 'emloop.tests.api_test.DummyDataset', 'output_dir': '/tmp'},
              'hooks': [{'emloop.tests.api_test.DummyHook': {'additional_arg': 1}}, {'StopAfter': {'epochs': 1}}],
              'model': {'class': 'emloop.tests.api_test.DummyModel', 'io': {'in': [], 'out': ['dummy']}}}

    with pytest.raises(Exception):
        create_main_loop(config=config, output_root=tmpdir).run_training()

    # incorrect hooks arguments
    config = {'dataset': {'class': 'emloop.tests.api_test.DummyDataset'},
              'hooks': [{'emloop.tests.api_test.DummyHook': {'additional_arg': 10}, 'StopAfter': {'epochs': 1}}],
              'model': {'class': 'emloop.tests.api_test.DummyModel', 'io': {'in': [], 'out': ['dummy']}}}

    with pytest.raises(Exception):
        create_main_loop(config=config, output_root=tmpdir).run_training()

    # incorrect model arguments
    config = {'dataset': {'class': 'emloop.tests.api_test.DummyDataset'},
              'hooks': [{'StopAfter': {'epochs': 1}}],
              'model': {'io': {'in': [], 'out': ['dummy']}}}

    with pytest.raises(Exception):
        create_main_loop(config=config, output_root=tmpdir).run_training()

    # incorrect main_loop arguments - raises error by default
    config = {'dataset': {'class': 'emloop.tests.api_test.DummyDataset'},
              'hooks': [{'StopAfter': {'epochs': 1}}],
              'model': {'class': 'emloop.tests.api_test.DummyModel', 'io': {'in': [], 'out': ['dummy']}},
              'main_loop': {'non-existent': 'none', 'extra_streams': ['train']}}

    with pytest.raises(Exception):
        create_main_loop(config=config, output_root=tmpdir).run_training()

    # incorrect main_loop arguments - logs warning
    caplog.clear()
    caplog.set_level(logging.WARNING)
    config['main_loop']['on_incorrect_config'] = 'warn'
    create_main_loop(config=config, output_root=tmpdir).run_training()
    assert 'Extra arguments: {\'non-existent\': \'none\'}' in caplog.text

    # incorrect main_loop arguments - ignored
    caplog.clear()
    caplog.set_level(logging.WARNING)
    config['main_loop']['on_incorrect_config'] = 'ignore'
    create_main_loop(config=config, output_root=tmpdir).run_training()
    assert 'Extra arguments: {\'non-existent\': \'none\'}' not in caplog.text


def test_run_with_eval_stream(tmpdir, caplog):
    """Test that eval is called with given stream and not with EL_DEFAULT_TRAIN_STREAM."""
    caplog.set_level(logging.INFO)

    config = {'dataset': {'class': 'emloop.tests.api_test.DummyEvalDataset'},
              'hooks': [{'emloop.tests.api_test.DummyEvalHook': {'epochs': 1}}, {'StopAfter': {'epochs': 1}}],
              'model': {'class': 'emloop.tests.api_test.DummyModel', 'io': {'in': ['a'], 'out': ['dummy']}}}
    
    create_main_loop(config=config, output_root=tmpdir).run_evaluation(stream_name='valid')
    
    assert f'Running the evaluation of stream `{EL_DEFAULT_TRAIN_STREAM}`' not in caplog.text
    assert 'Running the evaluation of stream `valid`' in caplog.text


def test_pass_mainloop_to_hooks(tmpdir):
    """Test that mainloop is passed to all hooks."""
    config = {'hooks': [{'emloop.tests.api_test.DummyHook': {'epochs': 1}},
                        {'emloop.tests.api_test.DummyEvalHook': {'epochs': 1}}]}

    dataset = 'dataset_placeholder'
    model = 'model_placeholder'
    hooks = create_hooks(config=config, dataset=dataset, model=model, output_dir=tmpdir)

    for hook in hooks:
        assert hook._main_loop == None

    main_loop = MainLoop(model=model, dataset=dataset, hooks=hooks)

    for hook in hooks:
        assert hook._main_loop == main_loop

    with pytest.raises(ValueError):
        MainLoop(model=model, dataset=dataset, hooks=hooks)


def test_training_with_list(tmpdir):
    config = {'dataset': {'class': 'emloop.tests.api_test.DummyEvalDataset'},
              'hooks': ['TrainingTrace'],
              'model': {'class': 'emloop.tests.api_test.DummyModel', 'io': {'in': ['a'], 'out': ['dummy']}}}

    main_loop = create_main_loop(config, tmpdir)
    inputs = [{"a" : [i, i+1]} for i in range(0, 10, 2)]
    with main_loop:
        main_loop.epoch(train_streams=[inputs], eval_streams=[])


def test_training_trace(tmpdir, caplog):
    """Test training_trace hook is created automatically even if it is not specified and saves correct file."""
    epochs = 2

    config = {'dataset': {'class': 'emloop.tests.api_test.DummyConfigDataset', 'batch_size': 10,
                          'dataset_config': ['a', 'b', 'c']},
              'stream': {'train': {'rotate': 20}},
              'hooks': [{'emloop.tests.api_test.DummyConfigHook': {'additional_arg': 10, 'variables': ['a', 'b']}},
                        {'StopAfter': {'epochs': epochs}}],
              'model': {'class': 'emloop.tests.api_test.DummyConfigModel',
                                 'architecture': {'model_config': ['a', 'b', 'c', 'd']},
                                 'io': {'in': [], 'out': ['dummy']}}}
    start = datetime.datetime.now()
    create_main_loop(config=config, output_root=tmpdir).run_training()
    end = datetime.datetime.now()

    assert ('root', logging.WARNING, 'TrainingTrace hook added between hooks. Add it to your config.yaml to suppress '
                                     'this warning.') in caplog.record_tuples
    yaml_file = os.path.join(tmpdir, os.listdir(tmpdir)[0], EL_TRACE_FILE)
    assert os.path.exists(yaml_file)
    loaded_yaml = load_yaml(yaml_file)
    assert loaded_yaml[TrainingTraceKeys.EPOCHS_DONE] == epochs
    assert loaded_yaml[TrainingTraceKeys.EXIT_STATUS] == 0
    assert start - loaded_yaml[TrainingTraceKeys.TRAIN_BEGIN] < datetime.timedelta(seconds=1)
    assert end - loaded_yaml[TrainingTraceKeys.TRAIN_END] < datetime.timedelta(seconds=1)
