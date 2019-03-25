"""
Config module providing util functions for handling YAML configurations.
"""
from typing import Tuple, Any, Iterable, Optional
import logging
from copy import deepcopy

import ruamel.yaml

from .yaml import load_yaml, reload


def parse_arg(arg: str) -> Tuple[str, Any]:
    """
    Parse CLI argument in format ``key=value`` to ``(key, value)``

    :param arg: CLI argument string
    :return: tuple (key, value)
    :raise: yaml.ParserError: on yaml parse error
    """
    assert '=' in arg, 'Unrecognized argument `{}`. [name]=[value] expected.'.format(arg)

    key = arg[:arg.index('=')]
    value = ruamel.yaml.load(arg[arg.index('=') + 1:], Loader=ruamel.yaml.Loader)

    return key, value


def load_config(config_file: str, additional_args: Iterable[str]=(), override_stream: Optional[str]=None) -> dict:
    """
    Load config from YAML ``config_file``.

    If ``override_stream`` is specified, update config by the respective eval.override_stream section, in particular:
        - hooks section is entirely replaced
        - model and dataset sections are updated

    Extend/override it with the given ``additional_args``.

    :param config_file: path to the YAML config file to be loaded
    :param additional_args: additional args which may extend or override the config loaded from the file
    :param override_stream: stream name whose config section will be updated
    :return: configuration as dict
    """

    config = load_yaml(config_file)

    if override_stream is not None and 'eval' in config and override_stream in config['eval']:
        update_section = config['eval'][override_stream]
        for subsection in ['dataset', 'model', 'main_loop']:
            if subsection in update_section:
                # deepcopy has to be made, otherwise responding eval section is overridden as well
                config[subsection].update(deepcopy(update_section[subsection]))
        if 'hooks' in update_section:
            config['hooks'] = update_section['hooks']
        else:
            logging.warning('Config does not contain `eval.%s.hooks` section. '
                            'No hook will be employed during the evaluation.', override_stream)
            config['hooks'] = []

    for key_full, value in [parse_arg(arg) for arg in additional_args]:
        key_split = key_full.split('.')
        key_prefix = key_split[:-1]
        key = key_split[-1]

        conf = config
        for key_part in key_prefix:
            conf = conf[key_part]
        conf[key] = value

    return reload(config)


__all__ = []
