"""
Test module for **emloop train** command (cli/train.py).
"""
import pytest
import os.path as path
from collections import namedtuple
import os

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
    #
    mocker.patch('emloop.api.delete_output_dir')
    import emloop.cli.train
    mocker.patch.object(emloop.cli.train, 'create_emloop_training', new=lambda _, __: EmloopTraining(output_dir='adfsgsdfgfds', dataset=None, main_loop=None, model=None, hooks=[]))
    emloop.cli.train.train(orig_config, [], "", False)
    assert mocker.assert_called()
