Getting Started
###############

Before we dive in, cxflow must be properly installed.

Requirements
************
In order to run and install cxflow, the following utilities must be installed:

- Python 3.5+.
- `pip` 9.0+

Cxflow depends on additional dependencies which are listed in the ``requirements.txt`` file.
Nevertheless, these dependencies are automatically installed by ``pip`` (see below).

Installation
************

The most simple installation of cxflow is via pip.
This is the recommended option for majority of users.

.. code-block:: bash

    pip install cxflow

Optionally, install additional plugins by installing ``cxflow-<plugin-name>``.
TensorFlow backend for cxflow might be installed by:

.. code-block:: bash

    pip install cxflow-tensorflow

In order to use cxflow nightly builds, install directly from the source code repository (``dev`` branch).

.. code-block:: bash

    pip install -U git+https://github.com/Cognexa/cxflow.git@dev

Developer Installation
**********************

Finally, cxflow might be installed directly for developer purposes.

.. code-block:: bash

    git clone git@github.com:Cognexa/cxflow.git
    cd cxflow
    pip install -e .

When inside the clonned directory, cxflow might be tested

.. code-block:: bash

    python setup.py test
