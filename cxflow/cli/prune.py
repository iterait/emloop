"""
Module with `cxflow prune` CLI functionality.
"""
import logging
from os import path, listdir
from shutil import rmtree

from ..constants import CXF_DEFAULT_LOG_DIR, CXF_TRACE_FILE
from .ls import is_train_dir
from ..utils.training_trace import TrainingTrace, TrainingTraceKeys


def _safe_rmtree(dir_: str):
    """Wrap ``shutil.rmtree`` to inform user about (un)success."""
    try:
        rmtree(dir_)
    except OSError:
        logging.warning('\t\t Skipping %s due to OSError', dir_)
    else:
        logging.debug('\t\t Deleted %s', dir_)


def _prune_subdirs(dir_: str) -> None:
    """
    Delete all subdirs in training log dirs.

    :param dir_: dir with training log dirs
    """
    for logdir in [path.join(dir_, f) for f in listdir(dir_) if is_train_dir(path.join(dir_, f))]:
        for subdir in [path.join(logdir, f) for f in listdir(logdir) if path.isdir(path.join(logdir, f))]:
            _safe_rmtree(subdir)


def _prune(dir_: str, epochs: int) -> None:
    """
    Delete all training dirs with incomplete training artifacts or with less than specified epochs done.

    :param dir_: dir with training log dirs
    :param epochs: minimum number of finished epochs to keep the training logs
    :return: number of log dirs pruned
    """
    for logdir in [path.join(dir_, f) for f in listdir(dir_) if path.isdir(path.join(dir_, f))]:
        if not is_train_dir(logdir):
            _safe_rmtree(logdir)
        else:
            trace_path = path.join(logdir, CXF_TRACE_FILE)
            try:
                epochs_done = TrainingTrace.from_file(trace_path)[TrainingTraceKeys.EPOCHS_DONE]
            except (KeyError, TypeError):
                epochs_done = 0
            if not epochs_done or epochs_done < epochs:
                _safe_rmtree(logdir)


def prune_train_dirs(dir_: str, epochs: int, subdirs: bool) -> None:
    """
    Prune training log dirs contained in the given dir. The function is accessible through cxflow CLI `cxflow prune`.

    :param dir_: dir to be pruned
    :param epochs: minimum number of finished epochs to keep the training logs
    :param subdirs: delete subdirs in training log dirs
    """

    if dir_ == CXF_DEFAULT_LOG_DIR and not path.exists(CXF_DEFAULT_LOG_DIR):
        print('The default log directory `{}` does not exist.\n'
              'Consider specifying the directory to be listed as an argument.'.format(CXF_DEFAULT_LOG_DIR))
        quit(1)

    if not path.exists(dir_):
        print('Specified dir `{}` does not exist'.format(dir_))
        quit(1)

    _prune(dir_, epochs)
    if subdirs:
        _prune_subdirs(dir_)
