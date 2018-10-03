Tutorial
########

In this tutorial, we are using an example task to demonstrate emloop’s basic
principles.

Introduction
************

**emloop** is a lightweight framework for machine learning which focuses on:

- modularization and re-usability of ML components (datasets, models etc.)
- rapid experimenting with different configurations
- providing convenient instruments to manage and run your experiments

**emloop** does not implement any building blocks, NN layers etc. Instead, you can use
your favorite machine learning framework, such as `TensorFlow
<https://www.tensorflow.org/>`_,
`CNTK <https://cntk.ai/>`_, or `Caffe2 <https://caffe2.ai/>`_. In other words,
**emloop** is back-end agnostic.
Therefore, you don't have to learn a new framework, if you already know one.
In addition, you can easily convert the models you already have by making only
minimal changes.

**emloop** allows (and encourages) you to build modular projects, where the
dataset, the model, and the configuration are separated and reusable. In the following sections,
we will describe how those reusable modules should look like on a simple
example.

Task
****

The tutorial will be demonstrated on a simple task called *majority*.
Given a vector of N bits, which bit is in majority?

Example of few 5-bit vectors:

+--------------+-----------------+----------------+--------------------------+
| input vector | number of zeros | number of ones | bit in majority (target) |
+--------------+-----------------+----------------+--------------------------+
| 00101        | 3               | 2              | 0                        |
+--------------+-----------------+----------------+--------------------------+
| 00000        | 5               | 0              | 0                        |
+--------------+-----------------+----------------+--------------------------+
| 10101        | 2               | 3              | 1                        |
+--------------+-----------------+----------------+--------------------------+
| 11101        | 1               | 4              | 1                        |
+--------------+-----------------+----------------+--------------------------+

.. tip::
   Full example may be found in our
   `emloop examples repository @GitHub <https://github.com/iterait/emloop-examples/tree/master/majority>`_.

Dataset
*******

The very first step in any machine learning task is to load and process the data.
Every **emloop** dataset is expected to implement the interface defined by :py:class:`emloop.datasets.AbstractDataset`.
At the moment, the interfaces defines only the constructor API which accepts a string-encoded YAML.
For regular projects, we recommend extending :py:class:`emloop.datasets.BaseDataset` which decodes the YAML
configuration string for you.

The dataset is meant to wrap all data-related operations.
It is responsible for correct data loading, verification and other useful operations.
The main purpose of the dataset is providing various data streams that will be consequently used for training,
validation and prediction in the production environment.

A typical **emloop** dataset will implement the following:

#. **Training stream:** an iteration of training data batches (``train_stream`` method)
#. **Eval streams:** iterations of additional streams not used for training.
   To provide a stream named <name>, method ``<name>_stream`` needs to return its iterator.
   In our example, we will use *test* stream provided by ``test_stream`` method.
#. **The constructor:** accepts a YAML configuration in the form of a string
   (more on this later). We avoid the need to implement a constructor by
   extending :py:class:`emloop.datasets.BaseDataset`.
#. **Additional methods:** such as ``fetch``, ``split``, or anything else you may need.
   **emloop** is able to call arbitrary dataset methods by invoking ``emloop dataset <method-name>`` command.

To generate the *majority* data and provide the data streams we will implement a ``MajorityDataset``:

.. code-block:: python
    :caption: majority_dataset.py

    import emloop as el
    import numpy.random as npr


    class MajorityDataset(el.BaseDataset):

        def _configure_dataset(self, n_examples: int, dim: int, batch_size: int, **kwargs) -> None:
            self.batch_size = batch_size
            self.dim = dim

            x = npr.random_integers(0, 1, n_examples * dim).reshape(n_examples, dim)
            y = x.sum(axis=1) > int(dim/2)

            self._train_x, self._train_y = x[:int(.8 * n_examples)], y[:int(.8 * n_examples)]
            self._test_x, self._test_y = x[int(.8 * n_examples):], y[int(.8 * n_examples):]

        def train_stream(self) -> el.Stream:
            for i in range(0, len(self._train_x), self.batch_size):
                yield {'x': self._train_x[i: i + self.batch_size],
                       'y': self._train_y[i: i + self.batch_size]}

        def test_stream(self) -> el.Stream:
            for i in range(0, len(self._test_x), self.batch_size):
                yield {'x': self._test_x[i: i + self.batch_size],
                       'y': self._test_y[i: i + self.batch_size]}


Let us describe the functionality of our ``MajorityDataset`` step by step.
We shall begin with the ``_configure_dataset`` method.
This method is called automatically by the dataset constructor, which provides it with the
parameters from the configuration file (configuration will be explained later).
In our case, we need ``n_examples`` (the number of examples in total), ``dim`` (the dimension of the
generated data) and ``batch_size`` (how big our batches will be).

The method randomly generates a dataset of ``n_examples`` vectors of ones and zeros (variable ``x``).
For each of those vectors, it calculates the correct answer (variable ``y``).
Finally, it splits the dataset into training and testing data in the ratio of 8:2.

To sum up, once the dataset is constructed, it features four attributes (``_train_x``,
``_train_y``, ``_test_x`` and ``_test_y``) that represent the loaded data.
Note that you have the option to rename them as desired.

.. note::
    In real-world cases, we usually don't want to generate our data randomly.
    Instead, we can simply load them from a file (e.g. ``.csv``) or from a database.

The train_stream function iterates over the training data.
This function returns an iterator over batches.
Each *batch* is a dictionary with keys ``x`` and ``y``, where the value of ``x`` is a list of
training vectors and the value of ``y`` is the list of the correct answers.
The lists have the length of ``batch_size``.

A batch (with ``batch_size=4``) representing the example above looks like this:

.. code-block:: python

    {
        'x': [
            [0,0,1,0,1],
            [0,0,0,0,0],
            [1,0,1,0,1],
            [1,1,1,0,0]
        ],
        'y': [
            0,
            0,
            1,
            1
        ]
    }

Similarly, there is a ``test_stream`` function that iterates over the testing data.

A single iteration over the whole dataset is called an *epoch*.
We train our machine learning models by iterating through the training stream
for one or more epochs.
The test stream is used only to estimate the performance of the model.

.. note::

    In this example, the training and testing streams are generated randomly and thus,
    they may slightly overlap and bias the performace estimation.

A detailed description of **emloop** datasets might be found in the
:doc:`advanced section <advanced/dataset>`.

Model
*****

With the dataset ready, we now must define the model that is to be trained.
A simple `TensorFlow <https://www.tensorflow.org/>`_ graph can solve our task.
We will use the official `emloop-tensorflow <https://github.com/iterait/emloop-tensorflow>`_ package that provides
convenient TensorFlow integration with **emloop**. Please install this package before you proceed
with this tutorial.

In :py:mod:`emloop_tensorflow`, every model is a python class that is expected to
extend the :py:class:`emloop_tensorflow.BaseModel`.

Let us define a class called ``MajorityNet``.

.. code-block:: python
    :caption: majority_net.py

    import logging

    import emloop_tensorflow as eltf
    import tensorflow as tf
    import tensorflow.contrib.keras as K


    class MajorityNet(eltf.BaseModel):
        """Simple 2-layered MLP for majority task."""

        def _create_model(self, hidden):
            logging.debug('Constructing placeholders matching the model.inputs')
            x = tf.placeholder(dtype=tf.float32, shape=[None, self._dataset.dim], name='x')
            y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

            logging.debug('Constructing MLP model')
            net = K.layers.Dense(hidden)(x)
            y_hat = K.layers.Dense(1)(net)[:, 0]

            logging.debug('Constructing loss and outputs matching the model.outputs')
            tf.pow(y - y_hat, 2, name='loss')
            predictions = tf.greater_equal(y_hat, 0.5, name='predictions')
            tf.equal(predictions, tf.cast(y, tf.bool), name='accuracy')

The only method that is necessary to implement is :py:meth:`emloop_tensorflow.BaseModel._create_model`.
In our case, the ``_create_model`` method creates a simple MLP.
If you know the fundamentals of TensorFlow, it should be easy to understand what is going on.

To be precise, the model registered the following computational graph nodes:

#. Placeholders ``x`` and ``y`` corresponding to a single batch from the stream (only the batch sources ``x`` and ``y`` will be mapped to these placeholders).
#. Variable ``loss`` denoting the mean square error of the model.
#. Variable ``predictions`` denoting the output of the network, i.e., the bit
   predicted to be in majority.
#. Variable ``accuracy`` denoting the fraction of correct predictions in the current batch.

.. caution::
   For each of input/output variables listed in the configuration, there has to
   exist a computational graph node with the corresponding name.
   **emloop-tensorflow** is not able to find the nodes if they are not properly
   named.

The ``_create_model`` method can accept arbitrary arguments - in our case, we allow to configure the number of hidden units.
We will describe the configuration file from which the parameters are taken in the next section.

You can find detailed descriptions of emloop models in the :doc:`advanced section <advanced/model>`.

Configuration
*************

The configuration of the training is the final, most important part of our tutorial.
The configuration or *config* defines which dataset will be used as the data source
and which model will be employed for training.

The configuration file is in the form of a YAML document.
Feel free to use JSON instead, however, YAML makes a lot of things easier.

The YAML document consists of four fundamental sections.
A detailed description of emloop configuration can be found in the :doc:`advanced section <advanced/config>`.


#. dataset
#. model
#. main_loop
#. hooks

Let us describe the sections one by one.

Dataset
=======

In our case, we only need to tell **emloop** which dataset to use.
This is done by specifying the ``class`` of the dataset.
In addition, we will specify the parameters of the dataset (those
are passed to the ``_configure_dataset`` method of the dataset).

.. code-block:: yaml

    dataset:
      class: majority.MajorityDataset
      n_examples: 500
      dim: 11
      batch_size: 4

We can pass arbitrary constants to the dataset that will be hidden in the
``**kwargs`` parameter of the ``_configure_dataset`` method of the dataset.

.. note::
    The whole ``dataset`` section will be passed as a string-encoded YAML to the dataset constructor.
    In the case of using :py:class:`emloop.datasets.BaseDataset`, the YAML is automatically decoded and the individual
    variables are passed to the ``_configure_dataset`` method.

Model
=====

Similarly to the dataset, the model is defined in the ``model`` section.
In our case, we want to specify the ``class`` of the model along with ``optimizer`` and
``hidden`` as required by the ``_create_model`` method of the model.
In addition, we will specify the ``name`` of the network which will be used for naming the
logging directory.

In addition, we have to specify which TensorFlow variable names are the network inputs
and which variable names are on the output.
This can be done by listing their names in the ``inputs`` and ``outputs`` config items.

.. code-block:: yaml

    model:
      name: MajorityExample
      class: majority.MajorityNet

      optimizer:
        class: AdamOptimizer
        learning_rate: 0.001

      hidden: 100

      inputs: [x, y]
      outputs: [accuracy, predictions, loss]

Main Loop
=========

As the model training is executed in epochs, it is naturally implemented as a loop.
This loop (:py:class:`emloop.MainLoop`) can be extended, for example by adding more
streams to the ``train`` stream.
In our case, we also want to evaluate the ``test`` stream, so we will add it to the
``main_loop.extra_streams`` section of the config. **emloop** will then invoke
the ``<name>_stream`` method of the dataset to create the stream. In our case,
the ``test_stream`` method will be invoked.

.. code-block:: yaml
    :caption: evaluate additional streams

    main_loop:
      extra_streams: [test]

Hooks
=====

Hooks can observe, modify and control the training process. In particular, hook actions are triggered after certain events,
such as after a batch or an epoch is completed (more info in :doc:`advanced section <advanced/hook>`).

The hooks to be used are specified in **emloop** configuration similar to the following one:

.. code-block:: yaml
    :caption: hook configuration section

    hooks:
      - ComputeStats:
          variables: [loss, accuracy]
      - LogVariables
      - StopAfter:
          epochs: 10

This section can be read quite naturally. **emloop** will now compute ``loss`` and ``accuracy``
means for each epoch and log the respective values. The training will be stopped after 10 epochs.

.. tip::
    See `API reference <emloop/emloop.hooks.html>`_ for full list of **emloop** hooks.

Using emloop
============

Once the classes and config are implemented, the training can begin.
Let's try it with

.. code-block:: bash

    emloop train majority/config.yaml

The command produces a lot of output.
The first section describes the creation of the components.
The second part presents the output of the hooks.
Finally, our logging hook is the one that produces the information after each epoch.
Now we can easily watch the progress of the training.

After the training is finished, note that there is a new directory ``log/MajorityExample_*``.
This is the logging directory where everything **emloop** produces is stored, including
saved models, the configuration file and various other artifacts.

Let's register one more hook which saves the best model according to the test stream:

.. code-block:: yaml

    - SaveBest:
        stream: test

When we run the training again, we see that the newly created output directory contains
the saved model as well.

Let's resume the training from this model.

.. code-block:: bash

    emloop resume log/MajorityExample_<some-suffix>

It's simple as that.

In case the model is good enough to be used in the production, it is extremely
easy to use emloop for this purpose.
See the configuration :doc:`advanced section <advanced/config>` for more details.
Then, you can just run the following command:

.. code-block:: bash

    emloop eval predict log/MajorityExample_<some-suffix>
