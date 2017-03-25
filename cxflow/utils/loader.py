import importlib


def _create_object(module_name: str, class_name: str, **kwargs):
    module = importlib.import_module(module_name)
    _class = getattr(module, class_name)
    return _class(**kwargs)


def create_object(object_config: dict, prefix: str= '', **kwargs):
    assert(prefix + 'module' in object_config)
    assert(prefix + 'class' in object_config)
    return _create_object(object_config[prefix + 'module'], object_config[prefix + 'class'], **kwargs)

