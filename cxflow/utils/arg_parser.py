import ast
import logging


def parse_arg(arg: str):
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
        logging.error('Couldn\'t convert argument %s of value %s to type %s. Original argument: "%s". Exception: %s',
                      key, value, typee, arg, e)
        raise AttributeError(
            'Could not convert argument {} of value {} to type {}. Original argument: "{}". Exception: {}'.format(
                key, value, typee, arg, e))

    return key, value
