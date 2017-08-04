Introduction
============

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
====

The tutorial will be demonstrated on a simple task called *majority*.
Given a vector of N bits, which bit is in majority?

Example:

.. code-block:: bash

   00101 -> 0
   00000 -> 0
   10101 -> 1
   11100 -> 1

Dataset
=======

The very first step in any machine learning task is to load and process the data.
Every `cxflow` dataset is expected to extend the `cxflow.datasets.AbstractDataset` 
and to have the following properties:

#. **The constructor** accepts a YAML configuration in the form of a string (configuration
   will be described later). The purpose of the constructor is to prepare data which will
   be used for training and optionally testing.
#. **Training stream** must be implemented. Training stream is an iterable which provides
   batches of the training data. The implementation of the training stream is provided in
   `create_train_stream` method.
#. Analogously, the dataset might contain **additional streams** such as validation or test
   streams. These are implemented in `create_<name>_stream` methods (e.g.,
   `create_test_stream`). While additional streams are not mandatory, implementing at
   least the test stream is strongly suggested.
#. Finally, the dataset might contain `split()` method which is responsible for correct
   data splitting to training and testing subsets. This will be described in detail in
   Advanced section.

For our purposes, let us create a class called `MajorityDataset`:

.. code-block:: python

    from cxflow.datasets import AbstractDataset
    import numpy.random as npr


    class MajorityDataset(AbstractDataset):
        def __init__(self, config: str):
            super().__init__(config)
            N = 500
            dim = 11
            self.batch_size = 4

            x = npr.random_integers(0, 1, N*dim).reshape(N, dim)
            y = x.sum(axis=1) > int(dim/2)

            self.train_x, self.train_y = x[:int(.8*N)], y[:int(.8*N)]
            self.test_x, self.test_y = x[int(.8*N):], y[int(.8*N):]

        def create_train_stream(self):
            for i in range(0, len(self.train_x), self.batch_size):
                yield {'x': self.train_x[i: i+self.batch_size],
                       'y': self.train_y[i: i+self.batch_size]}

        def create_test_stream(self):
            for i in range(0, len(self.test_x), self.batch_size):
                yield {'x': self.test_x[i: i+self.batch_size],
                       'y': self.test_y[i: i+self.batch_size]}

Let us describe the functionality of our `MajorityDataset` step by step.
In the constructor, the dataset randomly generates a dataset of 500 vectors of ones and
zeros (variable *x*). For each of those vectors, it calculates the correct
answer (variable *y*). Finaly, it splits the dataset to training and testing data
in the ratio of 8:2 (this would be better done in an extra `split` function, but
we have omitted this for the sake of simplicity).
Note that we ignore the configuration completely.

To iterate over the training data, there is a `create_train_stream` function. This function
returns an iterator over batches. Each batch is a dictionary with keys *x* and *y*, where
the value of *x* is a list of four training vectors and the value of *y* is the list of
the four correct answers. Similarly, there is a `create_test_stream` function that iterates
over the testing data. Iteration over the whole dataset is called an epoch.
Note that by this design, the training and testing streams do not overlap, hence we might
use the training stream for training and the testing stream for the independent estimation
of the model performance.

Model
=====

After the data are loaded, processed and ready to be used, we have to define the model
to be trained.
Let us define the model using a simple `TensorFlow <https://www.tensorflow.org/>`_ graph.
To make this process simpler, we will use the official 
`cxflow-tensorflow <https://github.com/Cognexa/cxflow-tensorflow>`_ package, that provides
a basic tensorflow integration to cxflow. Please install this package before you proceed
with this tutorial.

In cxflow, every tensorflow-based model is a python class expected to
extend the `cxflow_tf.BaseTFNet`. Let us define a class called `MajorityNet`:

.. code-block:: python

    import logging
    import tensorflow as tf
    import tensorflow.contrib.keras as K
    from cxflow_tf import BaseTFNet, create_optimizer


    class MajorityNet(BaseTFNet):

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
            loss = tf.reduce_mean(sq_err)

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

The only method that is really necessary to implement is the `_create_net`. In our case,
the `_create_net` method creates a simple MLP with the following nodes:

#. Placeholders *x* and *y* corresponding to a single *x* and *y* batch from the stream.
#. Variable `train_op` denoting the operation performing the training. This operation
   is called by `cxflow` during training.
#. Variable `predictions` denoting the output of the network, i.e., the supposed bit in majority.
#. Variable `accuracy` denoting the fraction of correct predictions in the current batch.

The `_create_net` method can accept arbitrary arguments - in our case, we accept the
optimization algorithm to be used and the number of hidden units.
Let us ignore the origin of these parameters for a while and address it in the
Configuration section. For now, let's simply assume they are set correctly.

Note that naming the variables correctly and consistently is mandatory - we will 
use the names in the next section.

Configuration
=============

TODO
