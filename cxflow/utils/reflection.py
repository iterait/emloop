"""
Module with handy functions which are able to create objects from module and class names.
"""
import importlib
import pkgutil

from types import MappingProxyType
from typing import Tuple, List, Dict, Iterable

_EMPTY_DICT = MappingProxyType({})


def create_object_from_config(config: Dict[str, object], args: Iterable=(),
                              kwargs: Dict[str, object]=_EMPTY_DICT, key_prefix: str=None):
    """
    Create an object instance according to the given config.

    Config dict has to contain module and class names under [key_prefix]module and [key_prefix]class keys.

    If no key_prefix is provided,
    the method attempts to deduce key names so that they contain 'module' and 'class' respectively.

    :param config: config dict
    :param args: args to be passed to the object constructor
    :param kwargs: kwargs to be passed to the object constructor
    :param key_prefix: module and class names key prefix
    :return: created object instance
    """
    if not key_prefix:
        module_matches = [key for key in config.keys() if 'module' in key]
        class_matches = [key for key in config.keys() if 'class' in key]

        if not (len(module_matches) == 1 and len(class_matches) == 1):
            raise ValueError('Failed to deduce module and class names keys. Please provide key_prefix')

        module_key = module_matches[0]
        class_key = class_matches[0]
    else:
        module_key = key_prefix + 'module'
        class_key = key_prefix + 'class'

    assert module_key in config
    assert class_key in config

    return create_object(module_name=config[module_key], class_name=config[class_key], args=args, kwargs=kwargs)


def create_object(module_name: str, class_name: str, args: Iterable=(), kwargs: Dict[str, object]=_EMPTY_DICT):
    """
    Create an object instance of the given class from the given module.
    Args and kwargs are passed to the constructor.

    -----------------------------------------------------
    This mimics the following code:
    -----------------------------------------------------
    from module import class
    return class(*args, **kwargs)
    -----------------------------------------------------

    :param module_name: module name
    :param class_name: class name
    :param args: args to be passed to the object constructor
    :param kwargs: kwargs to be passed to the object constructor
    :return: created object instance
    """
    assert isinstance(module_name, str)
    assert isinstance(class_name, str)

    _module = importlib.import_module(module_name)
    _class = getattr(_module, class_name)
    return _class(*args, **kwargs)


def list_submodules(module_name: str) -> List[str]:
    """List full names of all the submodules in the given module."""
    _module = importlib.import_module(module_name)
    return [module_name+'.'+submodule_name for _, submodule_name, _ in pkgutil.iter_modules(_module.__path__)]


def find_class_module(module_name: str, class_name: str) -> Tuple[List[str], List[Tuple[str, Exception]]]:
    """
    Find sub-modules of the given module that contain the given class.

    Moreover, return a list of sub-modules that could not be imported as a list of (sub-module name, Exception) tuples.

    :param module_name: name of the module to be searched
    :param class_name: searched class name
    :return: a tuple of sub-modules having the searched class and sub-modules that could not be searched
    """
    matched_submodules = []
    erroneous_submodules = []
    for submodule_name in list_submodules(module_name):
        try:  # the sub-module to be included may be erroneous and we need to continue
            submodule = importlib.import_module(submodule_name)
            if hasattr(submodule, class_name):
                matched_submodules.append(submodule_name)
        except Exception as ex:
            erroneous_submodules.append((submodule_name, ex))
    return matched_submodules, erroneous_submodules
