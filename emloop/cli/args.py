from argparse import ArgumentParser
import pkg_resources

from ..constants import EL_DEFAULT_LOG_DIR


def get_emloop_arg_parser(add_common_arguments: bool=False) -> ArgumentParser:
    """
    Create the **emloop** argument parser.

    :return: an instance of the parser
    """
    # create parser
    main_parser = ArgumentParser('emloop',
                                 description='emloop: lightweight framework for machine learning with '
                                             'focus on modularization, re-usability and rapid experimenting.',
                                 epilog='For more info see <https://iterait.github.io/emloop>')

    main_parser.add_argument('--version', action='version', help='Print emloop version and quit.',
                             version='emloop {}'.format(pkg_resources.get_distribution('emloop').version))
    subparsers = main_parser.add_subparsers(help='emloop commands')

    # create train sub-parser
    train_parser = subparsers.add_parser('train', description='Start emloop training from the ``config_file``.')
    train_parser.set_defaults(subcommand='train')
    train_parser.add_argument('config_file', help='path to the config file')

    # create resume sub-parser
    resume_parser = subparsers.add_parser('resume', description='Resume emloop training from the ``config_path``.')
    resume_parser.set_defaults(subcommand='resume')
    resume_parser.add_argument('config_path', help='path to the config file or the directory in which it is stored')
    resume_parser.add_argument('restore_from', nargs='?', default=None,
                               help='information passed to the model constructor (backend-specific); '
                                    'usually a directory in which the trained model is stored')

    # create eval sub-parser
    eval_parser = subparsers.add_parser('eval', description='Evaluate the given model on the specified data stream.')
    eval_parser.set_defaults(subcommand='eval')
    eval_parser.add_argument('stream_name', help='stream name to be evaluated')
    eval_parser.add_argument('model_path', help='model path to be evaluated')
    eval_parser.add_argument('--config', '-c', nargs='?', default=None, help='optional config path to be used')

    # create dataset sub-parser
    dataset_parser = subparsers.add_parser('dataset', description='Invoke arbitrary dataset method.')
    dataset_parser.set_defaults(subcommand='dataset')
    dataset_parser.add_argument('method', help='name of the method to be invoked')
    dataset_parser.add_argument('config_file', help='path to the config file')

    # create grid-search sub-parser
    gridsearch_parser = subparsers.add_parser('gridsearch', description='Do parameter grid search (experimental).')
    gridsearch_parser.set_defaults(subcommand='gridsearch')
    gridsearch_parser.add_argument('script', help='Script to be grid-searched')
    gridsearch_parser.add_argument('params', nargs='*', help='Params to be tested. Format: name:type=[value1,value2]. '
                                                             'Type is optional')
    gridsearch_parser.add_argument('--dry-run', action='store_true', help='Only print command output instead '
                                                                          'of executing it right away')

    # create ls sub-parser
    ls_parser = subparsers.add_parser('ls', description='List training log dirs in the given path.')
    ls_parser.set_defaults(subcommand='ls')
    ls_parser.add_argument('dir', nargs='?', default=EL_DEFAULT_LOG_DIR,
                           help='path to the log directory to be listed')
    ls_parser.add_argument('-l', '--long', action='store_true', help='use long listing format')
    ls_parser.add_argument('-a', '--all', action='store_true', help='include trainings with no epochs done')
    ls_parser.add_argument('-r', '--recursive', action='store_true',
                           help='list all the dirs recursively, stop at training dirs')
    ls_parser.add_argument('-v', '--verbose', action='store_true',
                           help='print more verbose output, applicable only when a single train dir is listed')

    # create prune sub-parser
    prune_parser = subparsers.add_parser('prune', description='Prune training log dirs in the given path without finished epochs.')
    prune_parser.set_defaults(subcommand='prune')
    prune_parser.add_argument('dir', nargs='?', default=EL_DEFAULT_LOG_DIR,
                              help='path to the log directory to be pruned')
    prune_parser.add_argument('-e', '--epochs', default=1, type=int,
                              help='keep only training log dirs having at least this many completed epochs, default 1')
    prune_parser.add_argument('-s', '--subdirs', action='store_true',
                              help='delete all subdirectories in training directories')

    # add common arguments
    if add_common_arguments:
        for parser in [main_parser, train_parser, resume_parser, dataset_parser, eval_parser]:
            parser.add_argument('--output_root', '-o', default='./log', help='output directory')
            parser.add_argument('--verbose', '-v', action='store_true', help='increase verbosity to level DEBUG')

    return main_parser
