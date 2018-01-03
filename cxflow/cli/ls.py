"""
Module with `cxflow ls` CLI function and associated util functions.
"""

import os
import os.path as path
from datetime import datetime
from collections import defaultdict
from typing import Iterable, Tuple, List

from tabulate import tabulate
from babel.dates import format_timedelta

from ..utils import load_config, yaml_to_str
from ..constants import CXF_DEFAULT_LOG_DIR, CXF_CONFIG_FILE, CXF_TRACE_FILE, CXF_NA_STR, CXF_LOG_FILE
from ..utils import TrainingTrace, TrainingTraceKeys


def print_boxed(str_: str) -> None:
    """Print the given string in ASCII box."""
    print(tabulate([[str_]], tablefmt='grid'))


def path_total_size(path_: str) -> int:
    """Compute total size of the given file/dir."""
    if path.isfile(path_):
        return path.getsize(path_)
    total_size = 0
    for root_dir, _, files in os.walk(path_):
        for file_ in files:
            total_size += path.getsize(path.join(root_dir, file_))
    return total_size


def humanize_filesize(filesize: int) -> Tuple[str, str]:
    """Return human readable pair of size and unit from the given filesize in bytes."""
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if filesize < 1024.0:
            return '{:3.1f}'.format(filesize), unit+'B'
        filesize /= 1024.0


def is_train_dir(dir_: str) -> bool:
    """Test if the given dir contains training artifacts."""
    return path.exists(path.join(dir_, CXF_CONFIG_FILE)) and \
           path.exists(path.join(dir_, CXF_TRACE_FILE)) and \
           path.exists(path.join(dir_, CXF_LOG_FILE))


def walk_train_dirs(root_dir: str) -> Iterable[Tuple[str, Iterable[str]]]:
    """
    Modify os.walk with the following:
        - return only root_dir and sub-dirs
        - return only training sub-dirs
        - stop recursion at training dirs

    :param root_dir: root dir to be walked
    :return: generator of (root_dir, training sub-dirs) pairs
    """
    if is_train_dir(root_dir):
        yield '', [root_dir]
        return
    for dir_, subdirs, _ in os.walk(root_dir, topdown=True):
        # filter train sub-dirs
        train_subdirs = [subdir for subdir in subdirs if is_train_dir(path.join(dir_, subdir))]

        # stop the recursion at the train sub-dirs
        for subdir in train_subdirs:
            subdirs.remove(subdir)

        yield dir_, train_subdirs


def get_classes(config: dict) -> Tuple[str, str]:
    """
    Return human readable model and dataset classes from the given config.

    :param config: configuration dict
    :return: a tuple of (model.class, dataset.class)
    """
    return config['model']['class'], config['dataset']['class']


def get_model_name(config: dict) -> str:
    """Return model name or `Unnamed`."""
    return config['model']['name'] if 'model' in config and 'name' in config['model'] else 'Unnamed'


def _print_trainings_long(trainings: Iterable[Tuple[str, dict, TrainingTrace]]) -> None:
    """
    Print a plain table with the details of the given trainings.

    :param trainings: iterable of tuples (train_dir, configuration dict, trace)
    """
    long_table = []
    for train_dir, config, trace in trainings:
        start_datetime, end_datetime = trace[TrainingTraceKeys.TRAIN_BEGIN], trace[TrainingTraceKeys.TRAIN_END]
        if start_datetime:
            age = format_timedelta(datetime.now() - start_datetime) + ' ago'
            if end_datetime:
                duration = format_timedelta(end_datetime - start_datetime)
            else:
                duration = CXF_NA_STR
        else:
            age = CXF_NA_STR
            duration = CXF_NA_STR

        epochs_done = trace[TrainingTraceKeys.EPOCHS_DONE] if trace[TrainingTraceKeys.EPOCHS_DONE] else 0

        long_table.append([path.basename(train_dir)] +
                          list(map(lambda fq_name: fq_name.split('.')[-1], get_classes(config))) +
                          [age, duration, epochs_done])

    print(tabulate(long_table, tablefmt='plain'))


def _ls_print_listing(dir_: str, recursive: bool, all_: bool, long: bool) -> List[Tuple[str, dict, TrainingTrace]]:
    """
    Print names of the train dirs contained in the given dir.

    :param dir_: dir to be listed
    :param recursive: walk recursively in sub-directories, stop at train dirs (--recursive option)
    :param all_: include train dirs with no epochs done (--all option)
    :param long: list more details including model name, model and dataset classes,
                 age, duration and epochs done (--long option)
    :return: list of found training tuples (train_dir, configuration dict, trace)
    """
    all_trainings = []
    for root_dir, train_dirs in walk_train_dirs(dir_):
        if train_dirs:
            if recursive:
                print(root_dir + ':')
            trainings = [(train_dir,
                          load_config(path.join(train_dir, CXF_CONFIG_FILE), []),
                          TrainingTrace.from_file(path.join(train_dir, CXF_TRACE_FILE)))
                         for train_dir
                         in [os.path.join(root_dir, train_dir) for train_dir in train_dirs]]
            if not all_:
                trainings = [train_dir for train_dir in trainings if train_dir[2][TrainingTraceKeys.EPOCHS_DONE]]
            if long:
                print('total {}'.format(len(trainings)))
                _print_trainings_long(trainings)
            else:
                for train_dir, _, _ in trainings:
                    print(path.basename(train_dir))
            all_trainings.extend(trainings)
            if recursive:
                print()

        if not recursive:
            break
    return all_trainings


def _ls_print_summary(all_trainings: List[Tuple[str, dict, TrainingTrace]]) -> None:
    """
    Print trainings summary.
    In particular print tables summarizing the number of trainings with
        - particular model names
        - particular combinations of models and datasets

    :param all_trainings: a list of training tuples (train_dir, configuration dict, trace)
    """
    counts_by_name = defaultdict(int)
    counts_by_classes = defaultdict(int)
    for _, config, _ in all_trainings:
        counts_by_name[get_model_name(config)] += 1
        counts_by_classes[get_classes(config)] += 1

    print_boxed('summary')
    print()

    counts_table = [[name, count] for name, count in counts_by_name.items()]
    print(tabulate(counts_table, headers=['model.name', 'count'], tablefmt='grid'))
    print()

    counts_table = [[classes[0], classes[1], count] for classes, count in counts_by_classes.items()]
    print(tabulate(counts_table, headers=['model.class', 'dataset.class', 'count'], tablefmt='grid'))
    print()


def _ls_print_verbose(training: Tuple[str, dict, str]) -> None:
    """
    Print config and artifacts info from the given training tuple (train_dir, configuration dict, trace).

    :param training: training tuple (train_dir, configuration dict, trace)
    """
    train_dir, config, _ = training
    print_boxed('config')
    print(yaml_to_str(config))
    print()

    print_boxed('artifacts')
    _, dirs, files = next(os.walk(train_dir))
    artifacts = [('d', dir) for dir in dirs] + \
                [('-', file_) for file_ in files if file_ not in [CXF_CONFIG_FILE, CXF_LOG_FILE, CXF_TRACE_FILE]]
    artifacts = [(type_, name) + humanize_filesize(path_total_size(path.join(train_dir, name)))
                 for type_, name in artifacts]
    print(tabulate(artifacts, tablefmt='plain', floatfmt='3.1f'))
    print()


def list_train_dirs(dir_: str, recursive: bool, all_: bool, long: bool, verbose: bool) -> None:
    """
    List training dirs contained in the given dir with options and outputs similar to the regular `ls` command.
    The function is accessible through cxflow CLI `cxflow ls`.

    :param dir_: dir to be listed
    :param recursive: walk recursively in sub-directories, stop at train dirs (--recursive option)
    :param all_: include train dirs with no epochs done (--all option)
    :param long: list more details including model name, odel and dataset class,
                 age, duration and epochs done (--long option)
    :param verbose: print more verbose output with list of additional artifacts and training config,
                    applicable only when a single train dir is listed (--verbose option)
    """
    if verbose:
        long = True

    if dir_ == CXF_DEFAULT_LOG_DIR and not path.exists(CXF_DEFAULT_LOG_DIR):
        print('The default log directory `{}` does not exist.\n'
              'Consider specifying the directory to be listed as an argument.'.format(CXF_DEFAULT_LOG_DIR))
        quit(1)

    if not path.exists(dir_):
        print('Specified dir `{}` does not exist'.format(dir_))
        quit(1)

    all_trainings = _ls_print_listing(dir_, recursive, all_, long)

    if long and len(all_trainings) > 1:
        if not recursive:
            print()
        _ls_print_summary(all_trainings)

    if verbose and len(all_trainings) == 1:
        if not recursive:
            print()
        _ls_print_verbose(all_trainings[0])
