import ast
import typing
import yaml
import ruamel.yaml
from os import path


def parse_arg(arg: str) -> typing.Tuple[str, typing.Any]:
    """
    Parse CLI argument in format key[:type]=value to (key, value)
    :param arg: CLI argument string
    :return: tuple (key, value[:type])
    """
    assert '=' in arg

    if ':' in arg:
        key = arg[:arg.index(':')]
        typee = arg[arg.index(':') + 1:arg.index('=')]
        value = arg[arg.index('=') + 1:]
    else:
        key = arg[:arg.index('=')]
        typee = 'str'
        value = arg[arg.index('=') + 1:]

    try:
        if typee == 'ast':
            value = ast.literal_eval(value)
        elif typee == 'int':
            value = int(float(value))
        elif typee == 'bool':
            value = bool(int(value))
        else:
            value = eval(typee)(value)
    except (Exception, AssertionError) as e:
        raise AttributeError(
            'Could not convert argument {} of value {} to type {}. Original argument: "{}". Exception: {}'.format(
                key, value, typee, arg, e))

    return key, value


def load_config(config_file: str, additional_args: typing.Iterable[str]) -> dict:
    """
    Load config from `config_file` and apply CLI args `additional_args`.
    :param additional_args: Additional args which may extend or override the config from file.
    :return: configuration as dict
    """

    with open(config_file, 'r') as f:
        config = ruamel.yaml.load(f, ruamel.yaml.RoundTripLoader)

    for key_full, value in [parse_arg(arg) for arg in additional_args]:
        key_split = key_full.split('.')
        key_prefix = key_split[:-1]
        key = key_split[-1]

        conf = config
        for key_part in key_prefix:
            conf = conf[key_part]
        conf[key] = value

    config = yaml.load(ruamel.yaml.dump(config, Dumper=ruamel.yaml.RoundTripDumper))
    return config


def config_to_file(config, output_dir: str, name: str= 'config.yaml') -> str:
    """
    Save the given config to the given path in yaml.
    :param config: configuration dict
    :param output_dir: target output directory
    :param name: target filename
    :return: target path
    """
    dumped_config_f = path.join(output_dir, name)
    with open(dumped_config_f, 'w') as f:
        yaml.dump(config, f)
    return dumped_config_f


def config_to_str(config: dict) -> str:
    """
    Return the given given config as yaml str.
    :param config: configuration dict
    :return: given configuration as yaml str
    """
    return yaml.dump(config)
