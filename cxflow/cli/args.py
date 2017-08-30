from argparse import ArgumentParser
import pkg_resources


def get_cxflow_arg_parser(add_common_arguments: bool=False) -> ArgumentParser:
    """
    Create the **cxflow** argument parser.

    :return: an instance of the parser
    """
    # create parser
    main_parser = ArgumentParser('cxflow',
                                 description='cxflow: lightweight framework for machine learning with '
                                             'focus on modularization, re-usability and rapid experimenting.',
                                 epilog='For more info see <https://cognexa.github.io/cxflow>')

    main_parser.add_argument('--version', action='version', help='Print cxflow version and quit.',
                             version='cxflow {}'.format(pkg_resources.get_distribution('cxflow').version))
    subparsers = main_parser.add_subparsers(help='cxflow commands')

    # create train subparser
    train_parser = subparsers.add_parser('train', description='Start cxflow training from the ``config_file``.')
    train_parser.set_defaults(subcommand='train')
    train_parser.add_argument('config_file', help='path to the config file')

    # create resume subparser
    resume_parser = subparsers.add_parser('resume', description='Resume cxflow training from the ``config_path``.')
    resume_parser.set_defaults(subcommand='resume')
    resume_parser.add_argument('config_path', help='path to the config file or the directory in which it is stored')
    resume_parser.add_argument('restore_from', nargs='?', default=None,
                               help='information passed to the model constructor (backend-specific); '
                                    'usually a directory in which the trained model is stored')

    # create predict subparser
    predict_parser = subparsers.add_parser('predict', description='Run prediction with the given ``config_path``.')
    predict_parser.set_defaults(subcommand='predict')
    predict_parser.add_argument('config_path', help='path to the config file or the directory in which it is stored')
    predict_parser.add_argument('restore_from', nargs='?', default=None,
                                help='information passed to the model constructor (backend-specific); usually a '
                                     'directory in which the trained model is stored')

    # create dataset subparser
    dataset_parser = subparsers.add_parser('dataset', description='Invoke arbitrary dataset method.')
    dataset_parser.set_defaults(subcommand='dataset')
    dataset_parser.add_argument('method', help='name of the method to be invoked')
    dataset_parser.add_argument('config_file', help='path to the config file')

    # create grid-search subparser
    gridsearch_parser = subparsers.add_parser('gridsearch', description='Do parameter grid search (experimental).')
    gridsearch_parser.set_defaults(subcommand='gridsearch')
    gridsearch_parser.add_argument('script', help='Script to be grid-searched')
    gridsearch_parser.add_argument('params', nargs='*', help='Params to be tested. Format: name:type=[value1,value2]. '
                                                             'Type is optional')
    gridsearch_parser.add_argument('--dry-run', action='store_true', help='Only print command output instead '
                                                                          'of executing it right away')

    # add common arguments
    if add_common_arguments:
        for parser in [main_parser, train_parser, resume_parser, predict_parser, dataset_parser]:
            parser.add_argument('--output_root', '-o', default='./log', help='output directory')
            parser.add_argument('--verbose', '-v', action='store_true', help='increase verbosity do level DEBUG')

    return main_parser
