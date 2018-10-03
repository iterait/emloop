Getting Started
###############

Before we dive in, emloop has to be installed properly.

Requirements
************
In order to run and install emloop, the following utilities must be installed:

- Python 3.5+
- ``pip`` 9.0+

emloop depends on additional dependencies which are listed in the ``requirements.txt`` file.
Nevertheless, these dependencies are automatically installed by ``pip`` (see below).

Installation
************

The simplest way of installing emloop is using pip.
This is the recommended option for majority of users.

.. code-block:: bash

    pip install emloop

Optionally, install additional plugins by installing ``emloop-<plugin-name>``.
TensorFlow backend for emloop can be installed by:

.. code-block:: bash

    pip install emloop-tensorflow

In order to use emloop nightly builds, install it directly from the source code 
repository (``dev`` branch).

.. code-block:: bash

    pip install -U git+https://github.com/iterait/emloop.git@dev

Developer Installation
**********************

Finally, emloop might be installed directly for development purposes.

.. code-block:: bash

    git clone git@github.com:iterait/emloop.git
    cd emloop
    pip install -e .

The emloop test suite can be executed by running the following command in the 
cloned repository:

.. code-block:: bash

    python setup.py test
