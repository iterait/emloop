================
``cxflow.utils``
================

.. automodule:: cxflow.utils


.. currentmodule:: cxflow.utils


Functions
=========

- :py:func:`parse_arg`:
  Parse CLI argument in format ``key[:type]=value`` to ``(key, value)``

- :py:func:`load_config`:
  Load config from YAML ``config_file`` and extend/override it with the given ``additional_args``.

- :py:func:`config_to_file`:
  Save the given config to the given path in YAML.

- :py:func:`config_to_str`:
  Return the given given config as YAML str.

- :py:func:`parse_fully_qualified_name`:
  Parse the given fully-quallified name (separated with dots) to a tuple of module and class names.

- :py:func:`create_object`:
  Create an object instance of the given class from the given module.

- :py:func:`list_submodules`:
  List full names of all the submodules in the given module.

- :py:func:`find_class_module`:
  Find sub-modules of the given module that contain the given class.

- :py:func:`get_class_module`:
  Get a sub-module of the given module which has the given class.


.. autofunction:: parse_arg

.. autofunction:: load_config

.. autofunction:: config_to_file

.. autofunction:: config_to_str

.. autofunction:: parse_fully_qualified_name

.. autofunction:: create_object

.. autofunction:: list_submodules

.. autofunction:: find_class_module

.. autofunction:: get_class_module


Classes
=======

- :py:class:`DisabledLogger`:
  Entirely disable the specified logger in between ``__enter__`` and ``__exit__``.

- :py:class:`DisabledPrint`:
  Disable printing to stdout by redirecting it to ``/dev/null`` in between ``__enter__`` and ``__exit__``.

- :py:class:`Timer`:
  Simple helper which is able to measure execution time of python code.



.. autoclass:: DisabledLogger
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: DisabledLogger
      :parts: 1

   |



.. autoclass:: DisabledPrint
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: DisabledPrint
      :parts: 1

   |



.. autoclass:: Timer
   :show-inheritance:
   :members:
   :private-members:
   :special-members:

   .. inheritance-diagram:: Timer
      :parts: 1

   |



Variables
=========

- :py:data:`_EMPTY_DICT`

.. autodata:: _EMPTY_DICT
   :annotation:

   .. code-block:: guess

      mappingproxy({})
