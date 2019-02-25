"""
Test module for **emloop prune** command (cli/prune.py).
"""
from os import listdir, mkdir, path
from pathlib import Path

from emloop.cli.prune import prune_train_dirs
from emloop.hooks.training_trace import TrainingTraceKeys
from emloop.constants import EL_CONFIG_FILE, EL_LOG_FILE, EL_TRACE_FILE


def test_prune(tmpdir):
    """Test correct logdir prunning."""
    withsubdir = 'withsubdir'
    logdirs = [path.join(tmpdir, dir_) for dir_ in [
        'first',
        'second',
        withsubdir,
        'emptytrace',
        'notrace',
        path.join(withsubdir, 'subdir')
    ]]
    for logdir in logdirs:
        mkdir(logdir)
    # set trace files
    for logdir, epochs in zip(logdirs[:3], [0, 1, 8]):
        with open(path.join(logdir, 'trace.yaml'), "w") as trace:
            trace.write(f'{TrainingTraceKeys.EPOCHS_DONE}: {epochs}')
    Path(path.join(logdirs[3], EL_TRACE_FILE)).touch()
    # create rest of the files from is_train_dir condition
    for logdir in logdirs[:-1]:
        Path(path.join(logdir, EL_CONFIG_FILE)).touch()
        Path(path.join(logdir, EL_LOG_FILE)).touch()

    prune_train_dirs(tmpdir, 0, False)
    assert path.exists(logdirs[-1])
    assert len(listdir(tmpdir)) == 2
    prune_train_dirs(tmpdir, 8, True)
    assert not path.exists(logdirs[-1])
    assert len(listdir(tmpdir)) == 1
    prune_train_dirs(tmpdir, 9, False)
    assert len(listdir(tmpdir)) == 0
