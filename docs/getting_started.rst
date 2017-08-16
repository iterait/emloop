Getting Started
###############

Before we dive in, cxflow must be properly installed.

Installation
************

The most simple installation of cxflow is via pip.
This is the recommended option for majority of users.

.. code-block:: bash

    pip install cxflow

Optionally, install additional plugins by running:

.. code-block:: bash

    # pip install cxflow-[plugin-name]
    # e.g.
    pip install cxflow-tensorflow

Alternatively, cxflow might be installed directly from the sources.

.. code-block:: bash

    pip install git+https://github.com/Cognexa/cxflow.git

Developer Installation
**********************

Finally, cxflow might be installed directly for developer purposes.

.. code-block:: bash

    git clone git@github.com:Cognexa/cxflow.git
    cd cxflow
    pip install -e .
    python setup.py test

When inside the clonned directory, cxflow might be tested

.. code-block:: bash

    python setup.py test
