"""
Test module for reflection utils (cxflow.utils.reflection).
"""
import sys
import os

from cxflow.utils.reflection import create_object, find_class_module, get_class_module, parse_fully_qualified_name
from cxflow.tests.test_core import CXTestCaseWithDir


class SimpleClass:  # pylint: disable=missing-docstring
    pass


class DuplicateClass:  # pylint: disable=missing-docstring
    pass


class ImportedClass:  # pylint: disable=missing-docstring
    # this class is imported in cxflow.tests.utils.dummy_module
    pass


class ClassWithArg:  # pylint: disable=missing-docstring
    def __init__(self, x):
        self.ex = x


class ClassWithArgs:  # pylint: disable=missing-docstring
    def __init__(self, x, *args):
        self.ex = x
        self.args = args


class ClassWithKwargs:  # pylint: disable=missing-docstring
    def __init__(self, x, **kwargs):
        self.ex = x
        self.kwargs = kwargs


class ClassWithArgsAndKwargs:  # pylint: disable=missing-docstring
    def __init__(self, x, *args, **kwargs):
        self.ex = x
        self.args = args
        self.kwargs = kwargs


class ClassWithArgsAndKwargsOnly:  # pylint: disable=missing-docstring
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class ReflectionTest(CXTestCaseWithDir):
    """Test case for the reflection util functions."""

    def test_parse_fully_qualified_name(self):
        """Test correct parsing of fully qualified names."""

        # test simple name
        module1, class1 = parse_fully_qualified_name('MyClass')
        self.assertIsNone(module1)
        self.assertEqual(class1, 'MyClass')

        # test simple path
        module1, class1 = parse_fully_qualified_name('MyModule.MyClass')
        self.assertEqual(module1, 'MyModule')
        self.assertEqual(class1, 'MyClass')

        # test complex path
        module1, class1 = parse_fully_qualified_name('MyModule.MySubmodule.MyClass')
        self.assertEqual(module1, 'MyModule.MySubmodule')
        self.assertEqual(class1, 'MyClass')

    def test_create_object(self):
        """Test base create object function."""
        module_name = 'cxflow.tests.utils.reflection_test'

        # test type
        obj0 = create_object(module_name=module_name, class_name='SimpleClass')
        self.assertEqual(type(obj0), SimpleClass)

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

    def test_find_class_module(self):
        """Test finding class module."""

        # test correct setup
        matched_modules, erroneous_modules = find_class_module('cxflow.tests.utils', 'SimpleClass')
        self.assertListEqual(matched_modules, ['cxflow.tests.utils.reflection_test'])
        self.assertListEqual(erroneous_modules, [])

        # test non-existent class
        matched_modules2, erroneous_modules2 = find_class_module('cxflow.tests.utils', 'DoesNotExists')
        self.assertListEqual(matched_modules2, [])
        self.assertListEqual(erroneous_modules2, [])

        # test multiple matches
        matched_modules3, erroneous_modules3 = find_class_module('cxflow.tests.utils', 'DuplicateClass')
        self.assertCountEqual(matched_modules3, ['cxflow.tests.utils.reflection_test',
                                                 'cxflow.tests.utils.dummy_module'])
        self.assertListEqual(erroneous_modules3, [])

    def test_find_class_module_errors(self):
        """Test erroneous modules handling in find_class_module function."""

        # create dummy module hierarchy
        module_name = 'my_dummy_module'
        valid_submodule_name = 'my_valid_submodule'
        invalid_submodule_name = 'my_invalid_submodule'
        module_path = os.path.join(self.tmpdir, module_name)

        os.mkdir(module_path)
        with open(os.path.join(module_path, '__init__.py'), 'w') as file:
            file.write('\n')
        with open(os.path.join(module_path, invalid_submodule_name+'.py'), 'w') as file:
            file.write('import ANonExistentModule\n')
        with open(os.path.join(module_path, valid_submodule_name+'.py'), 'w') as file:
            file.write('class MyClass:\n    pass\n')

        sys.path.append(self.tmpdir)

        matched_modules, erroneous_modules = find_class_module(module_name, 'MyClass')

        # test if the correct module is returned despite the erroneous module
        self.assertListEqual(matched_modules, [module_name+'.'+valid_submodule_name])

        # test if the erroneous module is returned correctly
        self.assertEqual(len(erroneous_modules), 1)
        self.assertEqual(erroneous_modules[0][0], module_name+'.'+invalid_submodule_name)
        self.assertIsInstance(erroneous_modules[0][1], ImportError)

    def test_get_class_module(self):
        """Test if get_class_module method wraps the `utils.reflection.find_class_module` method correctly."""

        # test if the module is returned directly
        module = get_class_module('cxflow.hooks', 'LogProfile')
        expected_module = 'cxflow.hooks.log_profile'
        self.assertEqual(module, expected_module)

        # test if None is returned when the class is not found
        module2 = get_class_module('cxflow.hooks', 'IDoNotExist')
        expected_module2 = None
        self.assertEqual(module2, expected_module2)

        # test if exception is raised when multiple modules are matched
        self.assertRaises(ValueError, get_class_module, 'cxflow.tests.utils', 'DuplicateClass')

        # test if any of valid sub-modules is returned if their attribute points to the same class
        # e.g. one sub-module imports a class from another sub-module
        module3 = get_class_module('cxflow.tests.utils', 'ImportedClass')
        possible_submodules = {'cxflow.tests.utils.reflection_test', 'cxflow.tests.utils.dummy_module'}
        self.assertIn(module3, possible_submodules)