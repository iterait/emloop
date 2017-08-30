Tutorial
########

In this tutorial, we demonstrate  **cxflow** basic principles via solving an example task.

Introduction
************

**cxflow** is a lightweight framework for machine learning which focuses on:

- modularization and re-usability of ML components (datasets, models etc.)
- rapid experimenting with different configurations
- providing convenient instruments to manage and run your experiments

**cxflow** does not implement any building blocks, NN layers etc. Instead, you can use
your favourite machine learning framework, such as `TensorFlow <https://www.tensorflow.org/>`_,
`CNTK <https://cntk.ai/>`_, or `Caffe2 <https://caffe2.ai/>`_; i.e., **cxflow** is back-end agnostic.
Therefore, you don't have to learn a new framework if you already know one.
In addition, you can easily convert the models you already have with only minimal changes.

**cxflow** allows (and encourages) you to build modular projects, wherein the dataset,
the model, and the configuration are separated and reusable. In the following sections,
we will describe how should those reusable modules look like on a simple example.

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
| 11100        | 2               | 3              | 1                        |
+--------------+-----------------+----------------+--------------------------+


Dataset
*******

The very first step in any machine learning task is to load and process the data.
Every **cxflow** dataset is expected to implement the interface defined by :py:class:`cxflow.datasets.AbstractDataset`.
At the moment, the interfaces defines only the constructor API which accepts a string-encoded YAML.
For regular projects, we recommend extending :py:class:`cxflow.datasets.BaseDataset` which decodes the YAML
configuration string for you.

The dataset is meant to wrap all data-related operations.
It is responsible for correct data loading, verification and other useful operations.
The main purpose of the dataset is providing various data streams that will be consequently used for training,
validation and prediction in the production environment.

Typical **cxflow** dataset will implement the following:

#. **Training stream:** an iteration of training data batches (``train_stream`` method)
#. **Eval streams:** iterations of additional <name> data batches not used for training.
   It is possible to provide any <name> stream in the respective ``<name>_stream`` methods.
   In our example, we will use *test* stream provided by ``test_stream`` method.
#. **The constructor:** accepts a YAML configuration in the form of a string
   (more on this later). We avoid the need to implement a constructor by 
   extending :py:class:`cxflow.datasets.BaseDataset`.
#. **Additional methods:** such as ``fetch``, ``split``, or anything else you may need.
   **cxflow** is able to call arbitrary dataset methods by invoking ``cxflow dataset <method-name>`` command.

To generate the *majority* data and provide the data streams we will implement a ``MajorityDataset``:

.. code-block:: python
    :caption: majority.py

    import cxflow as cx
    import numpy.random as npr


    class MajorityDataset(cx.BaseDataset):

        def _configure_dataset(self, n_examples: int, dim: int, batch_size: int, **kwargs) -> None:
            self.batch_size = batch_size

            x = npr.random_integers(0, 1, n_examples * dim).reshape(n_examples, dim)
            y = x.sum(axis=1) > int(dim/2)

            self._train_x, self._train_y = x[:int(.8 * n_examples)], y[:int(.8 * n_examples)]
            self._test_x, self._test_y = x[int(.8 * n_examples):], y[int(.8 * n_examples):]

        def train_stream(self) -> cx.AbstractDataset.Stream:
            for i in range(0, len(self._train_x), self.batch_size):
                yield {'x': self._train_x[i: i + self.batch_size],
                       'y': self._train_y[i: i + self.batch_size]}

        def test_stream(self) -> cx.AbstractDataset.Stream:
            for i in range(0, len(self._test_x), self.batch_size):
                yield {'x': self._test_x[i: i + self.batch_size],
                       'y': self._test_y[i: i + self.batch_size]}


Let us describe the functionality of our ``MajorityDataset`` step by step.
We shall begin with the ``_configure_dataset`` method.
This method is called from the dataset constructor automatically and it is passed the
parameters from the configuration file (configuration will be explained later).
In our case, we need ``n_examples`` (the number of examples in total), ``dim`` (the dimension of the
generated data) and ``batch_size`` (how big our batches will be).

The method randomly generates a dataset of ``n_examples`` vectors of ones and zeros (variable ``x``).
For each of those vectors, it calculates the correct answer (variable ``y``).
Finally, it splits the dataset to training and testing data in the ratio of 8:2.

To sum up, when the dataset is constructed, it features four attributes (``_train_x``,
``_train_y``, ``_test_x`` and ``_test_y``) that represent the loaded data.
Note that it is completely valid option to rename them as desired.

.. note::
    In real-world cases, we usually don't want to generate our data randomly.
    Instead, we can simply load them from file (e.g. ``.csv``) or database.

To iterate over the training data, there is a ``train_stream`` function.
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

Note that in this design, the training and testing streams do not overlap, hence we might
use the training stream for training and the testing stream for an unbiased estimation
of the performace of the model.

A detailed description of **cxflow** datasets might be found in the :doc:`advanced section <advanced>`.

Model
*****

Having the dataset ready, we have to define the model to be trained.
A simple `TensorFlow <https://www.tensorflow.org/>`_ graph can solve our task.
We will use the official `cxflow-tensorflow <https://github.com/Cognexa/cxflow-tensorflow>`_ package that provides
convenient TensorFlow integration with **cxflow**. Please install this package before you proceed
with this tutorial.

In :py:mod:`cxflow_tensorflow`, every model is a python class expected to
extend the :py:class:`cxflow_tensorflow.BaseModel`.

Let us define a class called ``MajorityNet``.

.. code-block:: python
    :caption: majority_net.py

    import logging

    import cxflow_tensorflow as cxtf
    import tensorflow as tf
    import tensorflow.contrib.keras as K


    class MajorityNet(cxtf.BaseModel):
        """Simple 2-layered MLP for majority task."""

        def _create_model(self, hidden):
            logging.debug('Constructing placeholders matching the model.inputs')
            x = tf.placeholder(dtype=tf.float32, shape=[None, 11], name='x')
            y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

            logging.debug('Constructing MLP model')
            net = K.layers.Dense(hidden)(x)
            y_hat = K.layers.Dense(1)(net)[:, 0]

            logging.debug('Constructing loss and outputs matching the model.outputs')
            tf.pow(y - y_hat, 2, name='loss')
            predictions = tf.greater_equal(y_hat, 0.5, name='predictions')
            tf.equal(predictions, tf.cast(y, tf.bool), name='accuracy')

The only method that is really necessary to implement is :py:meth:`cxflow_tensorflow.BaseModel._create_model`.
In our case, ``_create_model`` method creates a simple MLP.
If you know the fundamental basics of TensorFlow, it should be easy to understand what is going on.

To be precise, the model registered the following computational graph nodes:

#. Placeholders ``x`` and ``y`` corresponding to a single batch from the stream (only the batch sources ``x`` and ``y`` will be mapped to these placeholders).
#. Variable ``loss`` denoting the mean square error of the model.
#. Variable ``predictions`` denoting the output of the network, i.e., the supposed bit in majority.
#. Variable ``accuracy`` denoting the fraction of correct predictions in the current batch.

Note that the registration of the nodes is done by the node naming.
The variables that are not named explicitely will not be accessible in the future.

The ``_create_model`` method can accept arbitrary arguments - in our case, we allow to configure the number of hidden units.
We will describe the configuration file from which the parameters are taken in the next section.

Detailed description of **cxflow** models might be found in the :doc:`advanced section <advanced/index>`.

Configuration
*************

Configuration of the training is the final, most important part of our tutorial.
The configuration or *config* defines which dataset will be used as the data source
and which model will be employed for training.

The configuration file is in the form of a YAML document.
Feel free to use JSON instead, but YAML makes a lot of things easier.

The YAML document consists of four fundamental sections.
A detailed description of cxflow configuration can be found in the :doc:`advanced section <advanced/index>`.


#. dataset
#. model
#. main_loop
#. hooks

Let us describe the sections one by one.

Dataset
=======

In our case, we only need to tell cxflow which dataset to use.
This is done by specifying a ``class`` of the dataset.
In addition, we will specify the parameters of the dataset (those
ones passed to dataset's ``_configure_dataset`` method).

.. code-block:: yaml

    dataset:
      class: datasets.MajorityDataset
      n_examples: 500
      dim: 11
      batch_size: 4

We can pass arbitrary other constants to the dataset as they will be hidden in the ``**kwargs``
of the dataset's ``_configure_dataset`` method.

.. note::
    The whole ``dataset`` section will be passed as a string-encoded YAML to the dataset constructor.
    In the case of using :py:class:`cxflow.datasets.BaseDataset`, the YAML is automatically decoded and the individual
    variables are passed to ``_configure_dataset`` method.

Model
=====

Similarly to the dataset, the model is defined in the ``model`` section.
In our case, we want to specify  ``class`` of the model together with ``optimizer`` and
``hidden`` as required from the model's ``_create_model`` method.
In addition, we will specify the network ``name`` which will be used for naming the
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
This loop (:py:class:`cxflow.MainLoop`) can be extended, for example by adding more 
streams to the ``train`` stream.
In our case, we also want to evaluate the ``test`` stream, so we will add it to the
``main_loop.extra_streams`` section of the config. The streams are named by the dataset
methods they are created in. That is, the ``test`` stream corresponds to the
``test_stream`` method of the dataset.

.. code-block:: yaml

    main_loop:
      extra_streams: [test]

Hooks
=====

Hooks are actions which happen on some events, e.g. after a batch or an epoch.
Hooks represent an advanced topic which is covered in the advanced parts of the cxflow
documentation.

For now, we will simply use the following config snippet in order to register a few hooks.

.. code-block:: yaml

    hooks:
    - ComputeStats:
        variables:
          loss: [mean, std]
          accuracy: [mean]
    - LogVariables
    - CatchSigint
    - StopAfter:
        epochs: 10


As it might be observed, we have registered four hooks.
The first one computes various statistics: ``loss`` will be provided with its 
mean and standard deviation, ``accuracy`` will be provided with mean only.

The second hook is the logging hook which simply logs everything it gets to a log file
and to the standard error output (``/dev/stderr`` in case of unix environment).

The third hook makes sure the training safely stops on sigint signal and finishes
the current batch in progress.

The final hook stops the training after 10 epochs.

Using cxflow
============

Once the classes and config are implemented, the training can begin.
Let's try it with

.. code-block:: bash

    cxflow train majority/config.yaml

The command produces a lot of output.
The first section describes the creation of the components.
The second part presents the output of the hooks.
Finally, our logging hook is the one which produces the information after each epoch.
Now we can easily watch the progress of the training.

After the training is finished, note that there is a new directory ``log/MajorityExample_*``.
This is the logging directory where everything cxflow produced is stored, including
saved models, the configuration file and various other artifacts.

Let's register one more hook which saves the best model according to the test stream:

.. code-block:: yaml

    - SaveBest:
        stream: test

When we run the training again, we see that the newly created output directory contains
the saved model as well.

Let's resume the training from this model.

.. code-block:: bash

    cxflow resume log/MajorityExample_<some-suffix>

Simple as that.

In case the model is good enough to be used in the production, it is extremely
easy to use cxflow for this purpose.
See the configuration :doc:`advanced section <advanced/config>` for more details.
The usage is then extremely simple.

.. code-block:: bash

    cxflow predict log/MajorityExample_<some-suffix>
