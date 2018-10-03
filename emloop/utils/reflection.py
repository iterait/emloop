"""
Module with handy functions which are able to create objects from module and class names.
"""
import logging
import importlib
import pkgutil

from types import MappingProxyType
from typing import Tuple, List, Dict, Iterable, Any, Optional

_EMPTY_DICT = MappingProxyType({})


def parse_fully_qualified_name(fq_name: str) -> Tuple[Optional[str], str]:
    """
    Parse the given fully-quallified name (separated with dots) to a tuple of module and class names.

    :param fq_name: fully qualified name separated with dots
    :return: ``None`` instead of module if the given name contains no separators (dots).
    """
    last_dot = fq_name.rfind('.')
    if last_dot != -1:
        return fq_name[:last_dot], fq_name[last_dot + 1:]
    else:
        return None, fq_name


def get_attribute(module_name: str, attribute_name: str):
    """
    Get the specified module attribute. It most cases, it will be a class or function.

    :param module_name: module name
    :param attribute_name: attribute name
    :return: module attribute
    """
    assert isinstance(module_name, str)
    assert isinstance(attribute_name, str)

    _module = importlib.import_module(module_name)
    return getattr(_module, attribute_name)


def create_object(module_name: str, class_name: str, args: Iterable=(), kwargs: Dict[str, Any]=_EMPTY_DICT):
    """
    Create an object instance of the given class from the given module.
    Args and kwargs are passed to the constructor.

    This mimics the following code:

    .. code-block:: python

        from module import class
        return class(*args, **kwargs)

    :param module_name: module name
    :param class_name: class name
    :param args: args to be passed to the object constructor
    :param kwargs: kwargs to be passed to the object constructor
    :return: created object instance
    """
    return get_attribute(module_name, class_name)(*args, **kwargs)


def list_submodules(module_name: str) -> List[str]:   # pylint: disable=invalid-sequence-index
    """
    List full names of all the submodules in the given module.

    :param module_name: name of the module of which the submodules will be listed
    """
    _module = importlib.import_module(module_name)
    return [module_name+'.'+submodule_name for _, submodule_name, _ in pkgutil.iter_modules(_module.__path__)]


def find_class_module(module_name: str, class_name: str) \
        -> Tuple[List[str], List[Tuple[str, Exception]]]:   # pylint: disable=invalid-sequence-index
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
        except Exception as ex:  # pylint: disable=broad-except
            erroneous_submodules.append((submodule_name, ex))
    return matched_submodules, erroneous_submodules


def get_class_module(module_name: str, class_name: str) -> Optional[str]:
    """
    Get a sub-module of the given module which has the given class.

    This method wraps `utils.reflection.find_class_module method` with the following behavior:

    - raise error when multiple sub-modules with different classes with the same name are found
    - return None when no sub-module is found
    - warn about non-searchable sub-modules

    .. note::
        This function logs!

    :param module_name: module to be searched
    :param class_name: searched class name
    :return: sub-module with the searched class or None
    """
    matched_modules, erroneous_modules = find_class_module(module_name, class_name)

    for submodule, error in erroneous_modules:
        logging.warning('Could not inspect sub-module `%s` due to `%s` '
                        'when searching for `%s` in sub-modules of `%s`.',
                        submodule, type(error).__name__, class_name, module_name)

    if len(matched_modules) == 1:
        return matched_modules[0]
    if len(matched_modules) > 1:
        # check if all the module attributes point to the same class
        first_class = getattr(importlib.import_module(matched_modules[0]), class_name)
        for matched_module in matched_modules:
            another_class = getattr(importlib.import_module(matched_module), class_name)
            if another_class is not first_class:
                raise ValueError('Found more than one sub-module when searching for `{}` in sub-modules of `{}`. '
                                 'Please specify the module explicitly. Found sub-modules: `{}`'
                                 .format(class_name, module_name, matched_modules))
        return matched_modules[0]
    return None


__all__ = []
