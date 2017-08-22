====================
``cxflow.constants``
====================

.. automodule:: cxflow.constants


.. currentmodule:: cxflow.constants


Variables
=========

- :py:data:`CXF_LOG_FORMAT`
- :py:data:`CXF_LOG_DATE_FORMAT`
- :py:data:`CXF_FULL_DATE_FORMAT`
- :py:data:`CXF_HOOKS_MODULE`
- :py:data:`CXF_CONFIG_FILE`
- :py:data:`CXF_LOG_FILE`

.. autodata:: CXF_LOG_FORMAT
   :annotation:

   .. code-block:: guess

      '%(asctime)s.%(msecs)06d: %(levelname)-8s@%(module)-15s: %(message)s'

.. autodata:: CXF_LOG_DATE_FORMAT
   :annotation:

   .. code-block:: guess

      '%Y-%m-%d %H:%M:%S'

.. autodata:: CXF_FULL_DATE_FORMAT
   :annotation:

   .. code-block:: guess

      '%Y-%m-%d %H:%M:%S.%f'

.. autodata:: CXF_HOOKS_MODULE
   :annotation:

   .. code-block:: guess

      'cxflow.hooks'

.. autodata:: CXF_CONFIG_FILE
   :annotation:

   .. code-block:: guess

      'config.yaml'

.. autodata:: CXF_LOG_FILE
   :annotation:

   .. code-block:: guess

      'train.log'
