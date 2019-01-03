"""
Config module providing util functions for handling YAML configurations.
"""
import typing

import yaml

from .yaml import load_yaml, reload


def parse_arg(arg: str) -> typing.Tuple[str, typing.Any]:
    """
    Parse CLI argument in format ``key=value`` to ``(key, value)``

    :param arg: CLI argument string
    :return: tuple (key, value)
    :raise: yaml.ParserError: on yaml parse error
    """
    assert '=' in arg, 'Unrecognized argument `{}`. [name]=[value] expected.'.format(arg)

    key = arg[:arg.index('=')]
    value = yaml.load(arg[arg.index('=') + 1:])

    return key, value


def load_config(config_file: str, additional_args: typing.Iterable[str]=()) -> dict:
    """
    Load config from YAML ``config_file`` and extend/override it with the given ``additional_args``.

    :param config_file: path the YAML config file to be loaded
    :param additional_args: additional args which may extend or override the config loaded from the file.
    :return: configuration as dict
    """

    def override_value(config_dict, dict_key, dict_key_prefix, arg_value):
        for key_part in dict_key_prefix:
            config_dict = config_dict[key_part]
        config_dict[dict_key] = arg_value

    config = load_yaml(config_file)

    for key_full, value in [parse_arg(arg) for arg in additional_args]:
        key_split = key_full.split('.')
        key_prefix = key_split[:-1]
        key = key_split[-1]

        conf = config
        override_value(conf, key, key_prefix, value)

        if 'eval' in config:
            eval_conf = config['eval']
            for stream in eval_conf:
                stream_conf = eval_conf[stream]
                override_value(stream_conf, key, key_prefix, value)

    return reload(config)


__all__ = []
