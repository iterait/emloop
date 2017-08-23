==============
``cxflow.cli``
==============

.. automodule:: cxflow.cli


.. currentmodule:: cxflow.cli


Functions
=========

- :py:func:`train`:
  Load config and start the training.

- :py:func:`resume`:
  Load config from the directory specified and start the training.

- :py:func:`grid_search`:
  Build all grid search parameter configurations and optionally run them.

- :py:func:`_build_grid_search_commands`:
  Build all grid search parameter configurations.

- :py:func:`predict`:
  Load config from the directory specified and start the prediction.

- :py:func:`invoke_dataset_method`:
  Create the specified dataset and invoke its specified method.

- :py:func:`get_cxflow_arg_parser`:
  Create the cxflow argument parser.

- :py:func:`create_output_dir`:
  Create output_dir under the given ``output_root`` and

- :py:func:`create_dataset`:
  Create a dataset object according to the given config.

- :py:func:`create_model`:
  Create a model object either from scratch of from the checkpoint in `resume_dir`.

- :py:func:`create_hooks`:
  Create hooks specified in ``config['hooks']`` list.

- :py:func:`run`:
  Run cxflow training configured by the passed `config`.

- :py:func:`find_config`:
  Derive configuration file path from the given path and check its existence.

- :py:func:`validate_config`:
  Assert the config contains both `model` and `dataset` sections.

- :py:func:`fallback`:
  Fallback procedure when a cli command fails.


.. autofunction:: train

.. autofunction:: resume

.. autofunction:: grid_search

.. autofunction:: _build_grid_search_commands

.. autofunction:: predict

.. autofunction:: invoke_dataset_method

.. autofunction:: get_cxflow_arg_parser

.. autofunction:: create_output_dir

.. autofunction:: create_dataset

.. autofunction:: create_model

.. autofunction:: create_hooks

.. autofunction:: run

.. autofunction:: find_config

.. autofunction:: validate_config

.. autofunction:: fallback
