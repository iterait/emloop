"""
Test module for **emloop train/eval/resume** commands.
"""
import os
import os.path as path
import pytest
from collections import namedtuple

EmloopTraining = namedtuple("Training", "output_dir dataset model hooks main_loop")


@pytest.fixture
def simple_yaml():
    yield """
          model:
            class: Model

          dataset:
            class: Dataset
          """


def test_delete_dir_option_train(tmpdir, mocker, simple_yaml):
    """Test that delete_dir is called (or not called) under any circumstances."""

    orig_config = path.join(tmpdir, 'test.yaml')

    with open(orig_config, 'w') as file:
        file.write(simple_yaml)

    import emloop.cli.train_fn
    mocker.patch.object(emloop.cli.train_fn, 'create_emloop_training',
                        new=lambda _, __, ___, ____: EmloopTraining('dir', None, None, [], None))

    mymocker = mocker.patch('emloop.cli.train_fn.delete_output_dir')

    emloop.cli.train_fn.train(orig_config, [], "", False, "")
    assert not mymocker.called

    emloop.cli.train_fn.train(orig_config, [], "", True, "")
    assert mymocker.called


def test_delete_dir_option_resume(tmpdir, mocker, simple_yaml):
    """Test that delete_dir is called (or not called) under any circumstances."""

    orig_config = path.join(tmpdir, 'test.yaml')

    with open(orig_config, 'w') as file:
        file.write(simple_yaml)

    import emloop.cli.resume_fn
    mocker.patch.object(emloop.cli.resume_fn, 'create_emloop_training',
                        new=lambda _, __, ___, ____: EmloopTraining('dir', None, None, [], None))

    mymocker = mocker.patch('emloop.cli.resume_fn.delete_output_dir')

    emloop.cli.resume_fn.resume(orig_config, "", [], "", False, "")
    assert not mymocker.called

    emloop.cli.resume_fn.resume(orig_config, "", [], "", True, "")
    assert mymocker.called


def test_delete_dir_option_eval(tmpdir, mocker, simple_yaml):
    """Test that delete_dir is called (or not called) under any circumstances."""

    orig_config = path.join(tmpdir, 'test.yaml')

    with open(orig_config, 'w') as file:
        file.write(simple_yaml)

    import emloop.cli.eval_fn
    mocker.patch.object(emloop.cli.eval_fn, 'create_emloop_training',
                        new=lambda _, __, ___, ____: EmloopTraining('dir', None, None, [], None))

    mymocker = mocker.patch('emloop.cli.eval_fn.delete_output_dir')

    emloop.cli.eval_fn.evaluate("", "", orig_config, [], "", False, "")
    assert not mymocker.called

    emloop.cli.eval_fn.evaluate("", "", orig_config, [], "", True, "")
    assert mymocker.called
