"""
Test module for reflection utils (cxflow.utils.reflection).
"""
import logging
from unittest import TestCase

from cxflow.utils.reflection import create_object, create_object_from_config


class SimpleClass:  # pylint: disable=too-few-public-methods
    pass


class ClassWithArg:  # pylint: disable=too-few-public-methods
    def __init__(self, x):
        self.ex = x


class ClassWithArgs:  # pylint: disable=too-few-public-methods
    def __init__(self, x, *args):
        self.ex = x
        self.args = args


class ClassWithKwargs:  # pylint: disable=too-few-public-methods
    def __init__(self, x, **kwargs):
        self.ex = x
        self.kwargs = kwargs


class ClassWithArgsAndKwargs:  # pylint: disable=too-few-public-methods
    def __init__(self, x, *args, **kwargs):
        self.ex = x
        self.args = args
        self.kwargs = kwargs


class ClassWithArgsAndKwargsOnly:  # pylint: disable=too-few-public-methods
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class ReflectionTest(TestCase):
    """Test case for create object methods."""

    def __init__(self, *args, **kwargs):
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)

    def test_create_object(self):
        """Test base create object function."""
        module_name = 'cxflow.tests.utils.reflection_test'

        # test type
        try:
            obj0 = create_object(module_name=module_name, class_name='SimpleClass')
            self.assertEqual(type(obj0), SimpleClass)
        except:
            self.fail()

        obj = create_object(module_name=module_name, class_name='ClassWithArg', args=(12,))
        self.assertEqual(type(obj), ClassWithArg)
        self.assertEqual(obj.ex, 12)

        # test args and kwargs
        obj2 = create_object(module_name=module_name, class_name='ClassWithArgs', args=(12, 1, 2, 3))
        self.assertEqual(obj2.ex, 12)
        self.assertTupleEqual(obj2.args, (1, 2, 3))

        obj3 = create_object(module_name=module_name, class_name='ClassWithKwargs', kwargs={'x': 12, 'y': 1, 'z': 2})
        self.assertEqual(obj3.ex, 12)
        self.assertDictEqual(obj3.kwargs, {'y': 1, 'z': 2})

        obj4 = create_object(module_name=module_name, class_name='ClassWithKwargs', args=(12,), kwargs={'y': 1, 'z': 2})
        self.assertEqual(obj4.ex, 12)
        self.assertDictEqual(obj4.kwargs, {'y': 1, 'z': 2})

        obj5 = create_object(module_name=module_name, class_name='ClassWithArgsAndKwargs',
                             args=(12, 1, 2, 3), kwargs={'y': 1, 'z': 2})
        self.assertEqual(obj5.ex, 12)
        self.assertTupleEqual(obj5.args, (1, 2, 3))
        self.assertDictEqual(obj5.kwargs, {'y': 1, 'z': 2})

        obj6 = create_object(module_name=module_name, class_name='ClassWithArgsAndKwargsOnly',
                             args=(1, 2, 3, ), kwargs={'y': 1, 'z': 2})
        self.assertTupleEqual(obj6.args, (1, 2, 3))
        self.assertDictEqual(obj6.kwargs, {'y': 1, 'z': 2})

    def test_create_object_fails(self):
        """Test create object behavior with erroneous aruments."""
        module_name = 'cxflow.tests.utils.reflection_test'

        # test insufficient arguments
        self.assertRaises(TypeError, create_object)
        self.assertRaises(TypeError, create_object, module_name=module_name)
        self.assertRaises(TypeError, create_object, class_name='SimpleClass')

        # test wrong argument types
        self.assertRaises(AssertionError, create_object, module_name=123, class_name='SimpleClass')
        self.assertRaises(AssertionError, create_object, module_name=module_name, class_name=123)

        # test wrong module/class names
        self.assertRaises(ImportError, create_object, module_name=module_name+'xxx', class_name='SimpleClass')
        self.assertRaises(AttributeError, create_object, module_name=module_name, class_name='WrongName')

    def test_create_object_from_config(self):
        """Test create_object_from_config function."""
        module_name = 'cxflow.tests.utils.reflection_test'

        # test type, config read and prefix
        simple_config = {'my_module': module_name, 'my_class': 'SimpleClass'}
        obj = create_object_from_config(simple_config, key_prefix='my_')
        self.assertEqual(type(obj), SimpleClass)

        obsfucated_config = {'my_module': module_name, 'my_class': 'SimpleClass', '2nd_class': 'ClassWithArg'}
        obj2 = create_object_from_config(obsfucated_config, key_prefix='my_')
        self.assertEqual(type(obj2), SimpleClass)

        # test args and kwargs forwarding
        args_and_kwargs_config = {'my_module': module_name, 'my_class': 'ClassWithArgsAndKwargs'}
        obj3 = create_object_from_config(args_and_kwargs_config, key_prefix='my_',
                                         args=(12, 1, 2, 3), kwargs={'y': 1, 'z': 2})
        self.assertEqual(type(obj3), ClassWithArgsAndKwargs)
        self.assertEqual(obj3.ex, 12)
        self.assertTupleEqual(obj3.args, (1, 2, 3))
        self.assertDictEqual(obj3.kwargs, {'y': 1, 'z': 2})

        # test auto config keys
        obj4 = create_object_from_config(simple_config)
        self.assertEqual(type(obj4), SimpleClass)

        multiple_module_config = {'my_module': module_name, 'another_module': module_name, 'my_class': 'SimpleClass'}
        self.assertRaises(ValueError, create_object_from_config, multiple_module_config)
        multiple_class_config = {'my_module': module_name, 'my_class': 'SimpleClass', 'second_class': 'MyClass'}
        self.assertRaises(ValueError, create_object_from_config, multiple_class_config)

        missing_module_config = {'my_class': 'SimpleClass'}
        self.assertRaises(ValueError, create_object_from_config, missing_module_config)
        missing_class_config = {'my_module': module_name}
        self.assertRaises(ValueError, create_object_from_config, missing_class_config)
