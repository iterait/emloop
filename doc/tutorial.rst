Introduction
============

Let us describe some fundamental ideas of cxflow.
Cxflow does not implement any building blocks for you (contrary to, e.g.,
`Keras <https://github.com/fchollet/keras>`_). Instead, you can directly use
your favourite machine learning framework, such as `TensorFlow <https://www.tensorflow.org/>`_,
`CNTK <https://cntk.ai/>`_, or `Caffe2 <https://caffe2.ai/>`_. Therefore,
you don't have to learn a new framework if you already know one
and you can easily convert the models you already have with only minimal changes.

Cxflow allows (and encourages) you to build modular projects, where the dataset,
the model, and the configuration are separated and reusable. In the following sections,
we will describe how should those reusable modules look like on a simple example.

Task
====

The tutorial will be demonstrated on a simple task called _majority_.
Given a vector of N bits, which bit is in majority?

Example:

.. code-block::

   00101 -> 0
   00000 -> 0
   10101 -> 1
   11100 -> 1

Dataset
=======

The very first step in any machine learning taks is to load and process the data.
Every `cxflow` dataset is expected to extend the `cxflow.datasets.AbstractDataset` 
and to have the following properties:

#. In the constructor, it takes a YAML configuration in the form of a string (configuration
   will be described later).
#. It has to contain a `create_train_stream()` method.
#. It may contain some extra `create_<name>_stream()` methods (e.g., `create_test_stream()`).
#. It may contain a `split()` method.

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
in the ratio of 8:2 (this would be better done in an extra `split()` function, but
we have ommited this for the sake of simplicity).

To iterate over the training data, there is a `create_train_stream()` function. This function
returns an iterator over batches. Each batch is a dictionary with keys *x* and *y*, where
the value of *x* is a list of four training vectors and the value of *y* is the list of
the four correct answers. Similarly, there is a `create_test_stream()` function that iterates
over the testing data. Iteration over the whole dataset is called an epoch. 

Model
=====

Let us define the model using a simple `TensorFlow <https://www.tensorflow.org/>`_,
`CNTK <https://cntk.ai/>`_ graph.
TODO

Configuration
=============
