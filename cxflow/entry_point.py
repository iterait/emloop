#!/usr/bin/python3 -mentry_point

from .main_loop import MainLoop
from .nets.abstract_net import AbstractNet
from .utils.config import load_config, config_to_str, config_to_file
from .utils.loader import create_object

from argparse import ArgumentParser
from datetime import datetime
from os import path
import logging
import os
import sys
import typing
import traceback


# set up custom logging format
_cxflow_log_formatter = logging.Formatter('%(asctime)s: %(levelname)-8s@%(module)s: %(message)s', datefmt='%H:%M:%S')


def train(config_file: str, cli_options: typing.Iterable[str], output_root: str) -> None:
    """
    Run cxflow training configured from the given file and cli_options.
    Unique output dir for this training is created under the given output_root dir
    wherein all the training outputs are saved.
    :param config_file: path to the training yaml config
    :param cli_options: additional CLI arguments to override or extend the yaml config
    :param output_root: directory wherein output_dir is created
    """

    """
    Step 1:
        - Load yaml configuration and override or extend it with parameters passed in CLI arguments.
        - Check if `net` and `dataset` configs are present
    """
    try:
        logging.info('Loading config')
        config = load_config(config_file=config_file, additional_args=cli_options)
        logging.debug('Loaded config: %s', config)

        assert ('net' in config)
        assert ('dataset' in config)
        if 'hooks' not in config:
            logging.warning('No hooks found in config')
    except Exception as e:
        logging.error('Loading config failed: %s\n%s', e, traceback.format_exc())
        sys.exit(1)

    """
    Step 2:
        - Create output dir
        - Create file logger under the output dir
        - Dump loaded config to the output dir
    """
    try:
        logging.info('Creating output dir')

        # create output dir
        net_name = 'NonameNet'
        if 'name' not in config['net']:
            logging.warning('net.name not found in config, defaulting to: %s', net_name)
        else:
            net_name = config['net']['name']
        output_dirname = '{}_{}'.format(net_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f'))
        output_dir = path.join(output_root, output_dirname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create file logger
        file_handler = logging.FileHandler(path.join(output_dir, 'train_log.txt'))
        file_handler.setFormatter(_cxflow_log_formatter)
        logging.getLogger().addHandler(file_handler)

        # dump config including CLI args
        config_to_file(config=config, output_dir=output_dir)

    except Exception as e:
        logging.error('Failed to create output dir: %s\n%s', e, traceback.format_exc())
        sys.exit(1)

    """
    Step 3:
        - Create dataset
            - yaml string with `dataset`, `stream` and `log_dir` configs is passed to the dataset constructor
    """
    try:
        logging.info('Creating dataset')
        config_str = config_to_str({'dataset': config['dataset'],
                                    'stream': config['stream'],
                                    'output_dir': output_dir})
        dataset = create_object(object_config=config['dataset'], prefix='dataset_', config_str=config_str)
    except Exception as e:
        logging.error('Creating dataset failed: %s\n%s', e, traceback.format_exc())
        sys.exit(1)

    """
    Step 4:
        - Create network
            - Dataset, `log_dir` and net config is passed to the constructor
    """
    try:
        logging.info('Creating network')
        net_config = config['net']
        if 'restore_from' in net_config:
            logging.info('Restoring net from: "%s"', net_config['restore_from'])
            if 'net_module' in net_config or 'net_class' in net_config:
                logging.warning('`net_module` and `net_class` config parameters are provided yet ignored')
            net = AbstractNet(dataset=dataset, log_dir=output_dir, **net_config)
        else:
            logging.info('Creating new net')
            net = create_object(object_config=net_config,
                                prefix='net_', dataset=dataset, log_dir=output_dir, **net_config)
    except Exception as e:
        logging.error('Creating network failed: %s\n%s', e, traceback.format_exc())
        sys.exit(1)

    """
    Step 5:
        - Create all the training hooks
    """
    try:
        logging.info('Creating hooks')
        hooks = []
        if 'hooks' in config:
            for hook_config in config['hooks']:
                hooks.append(create_object(object_config=hook_config,
                                           prefix='hook_', net=net, config=config, **hook_config))
    except Exception as e:
        logging.error('Creating hooks failed: %s\n%s', e, traceback.format_exc())
        sys.exit(1)

    """
    Step 6:
        - Create the main loop object
    """
    try:
        logging.info('Creating main loop')
        main_loop = MainLoop(net=net, dataset=dataset, hooks=hooks)
    except Exception as e:
        logging.error('Creating main loop failed: %s\n%s', e, traceback.format_exc())
        sys.exit(1)

    """
    Step 7:
        - Run the main loop
    """
    try:
        logging.info('Running the main loop')
        main_loop.run(run_test_stream=('test' in config['stream']))
    except Exception as e:
        logging.error('Running the main loop failed: %s\n%s', e, traceback.format_exc())
        sys.exit(1)


def split(config_file: str, num_splits: int, train_ratio: float, valid_ratio: float, test_ratio: float=0):
    logging.info('Splitting to %d splits with ratios %f:%f:%f', num_splits, train_ratio, valid_ratio, test_ratio)

    logging.info('Loading config')
    try:
        config = load_config(config_file=config_file, additional_args=[])
    except Exception as e:
        logging.error('Loading config failed: %s\n%s', e, traceback.format_exc())
        sys.exit(1)

    logging.info('Creating dataset')
    try:
        config_str = config_to_str({'dataset': config['dataset'], 'stream': config['stream']})
        dataset = create_object(object_config=config['dataset'], prefix='dataset_', config_str=config_str)
    except Exception as e:
        logging.error('Creating dataset failed: %s\n%s', e, traceback.format_exc())
        sys.exit(1)

    logging.info('Splitting')
    dataset.split(num_splits, train_ratio, valid_ratio, test_ratio)


def init_entry_point() -> None:
    """
    cxflow entry point for training and dataset splitting.
    """

    # make sure the path contains the current working directory
    sys.path.insert(0, os.getcwd())

    # create parser
    parser = ArgumentParser('cxflow')
    subparsers = parser.add_subparsers(help='cxflow modes')

    # create train subparser
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(subcommand='train')
    train_parser.add_argument('config_file', help='path to the config file')

    # create split subparser
    split_parser = subparsers.add_parser('split')
    split_parser.set_defaults(subcommand='split')
    split_parser.add_argument('config_file', help='path to the config file')
    split_parser.add_argument('-n', '--num-splits', type=int, help='number of splits')
    split_parser.add_argument('-r', '--ratio', type=int, nargs=3, help='train, valid and test ratios')

    # add common arguments
    for p in [parser, train_parser, split_parser]:
        p.add_argument('-v', '--verbose', action='store_true', help='increase verbosity do level DEBUG')
        p.add_argument('-o', '--output-root', default='log', help='output directory')

    # parse CLI arguments
    known_args, unknown_args = parser.parse_known_args()

    # show help if no subcommand was specified.
    if not hasattr(known_args, 'subcommand'):
        parser.print_help()
        quit(1)

    # set up global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG if known_args.verbose else logging.INFO)
    logger.handlers = []  # remove default handlers

    # set up STDERR handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(_cxflow_log_formatter)
    logger.addHandler(stderr_handler)

    if known_args.subcommand == 'train':
        train(config_file=known_args.config_file,
              cli_options=unknown_args,
              output_root=known_args.output_root)

    elif known_args.subcommand == 'split':
        split(config_file=known_args.config_file,
              num_splits=known_args.num_splits,
              train_ratio=known_args.ratio[0],
              valid_ratio=known_args.ratio[1],
              test_ratio=known_args.ratio[2])


if __name__ == '__main__':
    init_entry_point()
