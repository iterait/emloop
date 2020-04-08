"""
Test module for **emloop train** command (cli/train.py).
"""
import pytest
from unittest.mock import Mock

from emloop.api import create_emloop_training, delete_output_dir
from emloop.cli.util import fallback, validate_config, find_config, print_delete_warning
from emloop.utils.config import load_config


def test_delete_dir_option(tmpdir):
    """Test that delete_dir is called under any circumstances."""
    config_path, cl_arguments, output_root = "", [""], ""
    delete_dir = False
    mock = Mock()

    with pytest.raises(Exception):
        try:
            config_path = find_config(config_path)
            config = load_config(config_file=config_path, additional_args=cl_arguments)
            validate_config(config)
            emloop_training = create_emloop_training(config, output_root)
            emloop_training.main_loop.run_training()
        except (Exception, AssertionError) as ex:  # pylint: disable=broad-except
            print('%s', ex)
        finally:
            print(f"Finally {delete_dir}")
            if delete_dir:
                mock.delete_output_dir("")
            sys.exit()

    assert mock.delete_output_dir.call_count == 0

    delete_dir = True
    mock = Mock()

    with pytest.raises(Exception):
        try:
            config_path = find_config(config_path)
            config = load_config(config_file=config_path, additional_args=cl_arguments)
            validate_config(config)
            emloop_training = create_emloop_training(config, output_root)
            emloop_training.main_loop.run_training()
        except (Exception, AssertionError) as ex:  # pylint: disable=broad-except
            print('%s', ex)
        finally:
            print(f"Finally {delete_dir}")
            if delete_dir:
                mock.delete_output_dir("")
            sys.exit()

    assert mock.delete_output_dir.call_count == 1
