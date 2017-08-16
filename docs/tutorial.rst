Tutorial
########

This is a simple example introducing cxflow fundamentals.

Introduction
************

Let us describe some fundamental ideas of cxflow.
Cxflow does not implement any building blocks for you (contrary to, e.g.,
`Keras <https://github.com/fchollet/keras>`_). Instead, you can directly use
your favourite machine learning framework, such as `TensorFlow <https://www.tensorflow.org/>`_,
`CNTK <https://cntk.ai/>`_, or `Caffe2 <https://caffe2.ai/>`_. Therefore,
you don't have to learn a new framework if you already know one.
In addition, you can easily convert the models you already have with only minimal changes.

Cxflow allows (and encourages) you to build modular projects, where the dataset,
the model, and the configuration are separated and reusable. In the following sections,
we will describe how should those reusable modules look like on a simple example.

Task
****

The tutorial will be demonstrated on a simple task called *majority*.
Given a vector of N bits, which bit is in majority?

Example:

.. code-block:: bash

   00101 -> 0
   00000 -> 0
   10101 -> 1
   11100 -> 1

Dataset
*******

The very first step in any machine learning task is to load and process the data.
Every `cxflow` dataset is expected to extend the `cxflow.datasets.BaseDataset` 
and to have the following properties:

#. **Training stream** must be implemented. Training stream is an iterable which provides
   batches of the training data.
   The implementation of the training stream is provided in `train_stream` method.
#. Analogously, the dataset might contain **additional streams** such as validation or test
   streams.
   These are implemented in `<name>_stream` methods (e.g.,
   `test_stream`). While additional streams are not mandatory, implementing at
   least the test stream is strongly suggested.
#. **The constructor** accepts a YAML configuration in the form of a string
   (more on this later).
   The purpose of the constructor is to parse the configuration string and
   prepare that cxflow will soon call one of the stream functions.
#. Finally, the dataset might contain other methods, e.g., `fetch`, `split`, or
   anything else you may need.
   Cxflow is able to call arbitrary function on your dataset using `cxflow dataset <name>`
   command.

For our purposes, let us create a class called `MajorityDataset`:

.. code-block:: python

    from cxflow.datasets import BaseDataset
    import numpy.random as npr


    class MajorityDataset(BaseDataset):
        def _init_with_kwargs(self, N: int, dim: int, batch_size: int, **kwargs):
            self.batch_size = batch_size

            x = npr.random_integers(0, 1, N*dim).reshape(N, dim)
            y = x.sum(axis=1) > int(dim/2)

            self.train_x, self.train_y = x[:int(.8*N)], y[:int(.8*N)]
            self.test_x, self.test_y = x[int(.8*N):], y[int(.8*N):]

        def train_stream(self):
            for i in range(0, len(self.train_x), self.batch_size):
                yield {'x': self.train_x[i: i+self.batch_size],
                       'y': self.train_y[i: i+self.batch_size]}

        def test_stream(self):
            for i in range(0, len(self.test_x), self.batch_size):
                yield {'x': self.test_x[i: i+self.batch_size],
                       'y': self.test_y[i: i+self.batch_size]}

Let us describe the functionality of our `MajorityDataset` step by step.
Let's begin with `_init_with_kwargs` method.
The method is called from the dataset constructor automatically and it is passed the
parameters from the configuration file (configuration will be explained later).
In our case, we need `N` (the number of examples in total), `dim` (the dimension of the
generated data) and `batch_size` (how big our batches will be).

The method randomly generates a dataset of `N` vectors of ones and zeros (variable `x`).
For each of those vectors, it calculates the correct answer (variable `y`).
Finaly, it splits the dataset to training and testing data in the ratio of 8:2
(this would be better done in an extra `split` function, but we have omitted this for
the sake of simplicity).

To sum up, when the dataset is constructed, it features four attributes (`train_x`,
`train_y`, `test_x` and `test_y`) that represent the loaded data.
Note that it is completely valid option to rename them as desired.
In real-world cases, we usually don't want to generate our data randomly.
Instead, we can simply load them from file (e.g. `.csv`) or database.

To iterate over the training data, there is a `train_stream` function.
This function returns an iterator over batches.
Each *batch* is a dictionary with keys `x` and `y`, where the value of `x` is a list of
training vectors and the value of `y` is the list of the correct answers.
The lists have the length of `batch_size`.

A batch (with `batch_size=4`) representing the example above looks like this:

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

Similarly, there is a `test_stream` function that iterates over the testing data.

Iteration over the whole dataset is called an *epoch*.
We train our machine learning models by iterating through the training stream for a single
or multiple epochs.
The test stream is used only to estimate the performance of the model.

Note that in this design, the training and testing streams do not overlap, hence we might
use the training stream for training and the testing stream for the unbiased estimation
of the model performance.

Model
*****

After the data are loaded, processed and ready to be used, we have to define the model
to be trained.
Let us define the model using a simple `TensorFlow <https://www.tensorflow.org/>`_ graph.
To make this process simpler, we will use the official 
`cxflow-tensorflow <https://github.com/Cognexa/cxflow-tensorflow>`_ package, that provides
a basic TensorFlow integration to cxflow. Please install this package before you proceed
with this tutorial.

In `cxflow-tensorflow`, every model is a python class expected to
extend the `cxflow_tensorflow.BaseModel`.

Let us define a class called `MajorityNet`.

.. code-block:: python

    import logging
    import tensorflow as tf
    import tensorflow.contrib.keras as K
    from cxflow_tensorflow import BaseModel, create_optimizer


    class MajorityNet(BaseModel):

        def _create_net(self, optimizer, hidden, **kwargs):

            logging.debug('Constructing placeholders')
            x = tf.placeholder(dtype=tf.float32, shape=[None, 11], name='x')
            y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

            logging.debug('Constructing MLP')
            hidden_activations = K.layers.Dense(hidden)(x)
            y_hat = K.layers.Dense(1)(hidden_activations)[:, 0]

            logging.debug('Constructing squared errors')
            sq_err = tf.pow(y - y_hat, 2)

            logging.debug('Constructing loss')
            loss = tf.reduce_mean(sq_err, name='loss')

            logging.debug('Constructing training operation')
            create_optimizer(optimizer).minimize(loss, name='train_op')

            logging.debug('Constructing predictions (argmax)')
            predictions = tf.greater_equal(y_hat, 0.5, name='predictions')

            logging.debug('Constructing accuracy')
            tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(y, tf.bool)),
                                   tf.float32, name='accuracy'))

            logging.debug('Variable initilization')
            self._session.run(tf.global_variables_initializer())
            self._session.run(tf.local_variables_initializer())


When implementing a custom model, make sure to extend the `cxflow.BaseModel` class.
As described above, this tutorial focuses only on TensorFlow model, hence extending
`cxflow_tensorflow.BaseModel` is a good idea.

The only method that is really necessary to implement is `_create_model`.
In our case, `_create_model` method creates a simple MLP.
If you know TensorFlow a little bit, it should be easy to understand what is going on.

To be precise, the model registred the following computational graph nodes:

#. Placeholders *x* and *y* corresponding to a single *x* and *y* batch from the stream.
#. Variable `train_op` denoting the operation performing the training. This operation
   is called by `cxflow` during training.
#. Variable `loss` denoting the mean square error of the model.
#. Variable `predictions` denoting the output of the network, i.e., the supposed bit in majority.
#. Variable `accuracy` denoting the fraction of correct predictions in the current batch.

Note that the registration of the nodes is done by the node naming.
The variables that are not named explicitely will not be accessible in the future.

The `_create_model` method can accept arbitrary arguments - in our case, we accept the
optimization algorithm to be used and the number of hidden units.
We will describe the configuration file from which the parameters are taken in the next section.

Configuration
*************

Configuration of the training is a key and final part of our tutorial.
The configuration (aka *config*) defines which dataset will be used as the data source
and which model will be employed for training.

The configuration file is in the form of a YAML document.
Feel free to use JSON instead, but YAML makes a lot of thing easier.

The YAML document consists of four fundamental sections.

#. dataset
#. model
#. main_loop
#. hooks

Let's dig into them one by one.

Dataset
=======

In our case, we only need to tell cxflow which dataset to use.
This is done by specifying `module` and `class` of the dataset.
In addition, we will specify the parameters of the dataset (those
ones passed to dataset's `_init_with_kwargs` method).

.. code-block:: yaml

    dataset:
      module: datasets.majority_dataset
      class: MajorityDataset
      N: 500
      dim: 11
      batch_size: 4

We can pass arbitrary other constants to the dataset as they will be hidden in the `**kwargs`
of the dataset's `_init_with_kwargs` method.

**Note:** The whole `dataset` section will be passed as a string-encoded YAML
to the dataset constructor.
In the case of using `cxflow.BaseDataset`, the YAML is automatically decoded and the individual
variables are passed to `_init_with_kwargs` method.

Model
=====

Similarly to the dataset, the model is defined in the `net` section.
In our case, we want to specify `module` and `class` of the model together with `optimizer` and
`hidden` as required from the model's `_create_net` method.
In addition, we will specify the network `name` which will be used for naming the
logging directory.

In addition, we have to specify which TensorFlow variable names are the network inputs
and which variable names are on the output.
This is done by `inputs` and `outputs` config items.

.. code-block:: yaml

    model:
      module: models.majority_net
      class: MajorityNet

      name: MajorityExample

      optimizer:
        module: tensorflow.python.training.adam
        class: AdamOptimizer
        learning_rate: 0.001
      hidden: 100

      inputs: [x, y]
      outputs: [accuracy, predictions, loss]

Main Loop
=========

As the model training is executed in epochs, it is naturally implemented as a loop.
This loop (`cxflow.MainLoop`) can be configured, e.g., additional streams to the `train`
stream might be specified.
In our case, we also want to evaluate the `test` stream, so we will add it to the
`main_loop.extra_streams` section of the config.  The streams are named by the dataset
methods they are created in. That is, the `test` stream corresponds to the
`test_stream` method of the dataset.

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
      - class: StatsHook
        variables:
          loss: [mean, std]
          accuracy: [mean]

      - class: LoggingHook
      - class: SigintHook

      - class: EpochStopperHook
        epoch_limit: 10

As it might be observed, we have registered four hooks.
The first one computes various statistics, e.g. `loss` will be provided with its mean and
standard deviation.
`accuracy` will be provided with mean only.

The second hook is the logging hook which simply logs everything it gets to a log file
and to the standard output.

The third hook makes sure the training safely stops on sigint signal and finishes
the current batch in progress.

The final hook stops the training after 10 epochs.

Using cxflow
============

Once the classes and config are implemented, the training might begin.
Let's try it with

.. code-block:: bash

    cxflow train configs/majority.yaml

The command produces a lot of output.
The first section describes the creation of the components.
The second part presents the output of the hooks.
Finally, our logging hook is the one which produces the information after each epoch.
Now we can easily watch the progress of the training.

After the training is finished, note that there is a new directory `log/MajorityExample_*`.
This is the logging directory where everything cxflow produced is stored, including
saved models, the configuration file and various other artifacts.

Let's register one more hook which saves the currently best model based on the test stream.

.. code-block:: yaml

      - class: BestSaverHook

When we run the training again, we see that the newly created output directory contains
the saved model as well.

Let's resume the training from this model.

.. code-block:: bash

    cxflow resume log/MajorityExample_<some-suffix>

Simple as that.

In case the model is good enough to be used in the production, it is extremely
easy to use cxflow for this purpose.
**Note:** the dataset must implement `predict_stream` method.
In addition, the net inputs and outputs should be modified in the configuration file
not to include `loss`, `accuracy` and `y`, since we don't know those in the
producion environment.

.. code-block:: bash

    cxflow predict log/MajorityExample_<some-suffix>

We cover the predicion in the production evironment in the advanced tutorials.
