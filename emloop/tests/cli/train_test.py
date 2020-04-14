"""
Test module for **emloop train** command (cli/train.py).
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


def test_delete_dir_option(tmpdir, mocker, simple_yaml):
    """Test that delete_dir is called under any circumstances."""

    orig_config = path.join(tmpdir, 'test.yaml')

    with open(orig_config, 'w') as file:
        file.write(simple_yaml)

    import emloop.cli.train_fn
    mocker.patch.object(emloop.cli.train_fn, 'create_emloop_training', new=lambda _, __: EmloopTraining(
        output_dir='adfsgsdfgfds', dataset=None, main_loop=None, model=None, hooks=[]))

    mymocker = mocker.patch('emloop.cli.train_fn.delete_output_dir')

    emloop.cli.train_fn.train(orig_config, [], "", False)
    assert not mymocker.called

    emloop.cli.train_fn.train(orig_config, [], "", True)
    assert mymocker.called
