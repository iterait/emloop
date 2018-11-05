"""
Module with simple hook saving files to the output dir test case (see :py:class:`emloop.hooks.SaveFile`).
"""
import os
import pytest

from emloop.hooks import SaveFile


def test_saving_file(tmpdir):
    dir = os.path.join(tmpdir, 'files-to-save')
    os.makedirs(dir)

    file1 = os.path.join(dir, 'file1.txt')
    with open(file1, 'w') as file:
        file.write('string')

    file2 = os.path.join(dir, 'file2.csv')
    with open(file2, 'w') as file:
        file.write('string')

    hook = SaveFile(files=[file1, file2], output_dir=tmpdir)

    hook.before_training()
    assert os.path.exists(os.path.join(tmpdir, 'file1.txt'))
    assert os.path.exists(os.path.join(tmpdir, 'file2.csv'))


def test_saving_nonexistent_file(tmpdir):
    file = os.path.join(tmpdir, 'files-to-save', 'file.txt')

    hook = SaveFile(files=[file], output_dir=tmpdir)

    with pytest.raises(AssertionError):
        hook.before_training()
