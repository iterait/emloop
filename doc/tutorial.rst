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
Given a vector of five bits, which bit is in majority?

Example:

.. code-block::
   00101 -> 0
   00000 -> 0
   10101 -> 1
   11100 -> 1

Dataset
=======

To generate the training and the testing data, we need to write a dataset class.
TODO

Model
=====

Let us define the model using a simple `TensorFlow <https://www.tensorflow.org/>`_,
`CNTK <https://cntk.ai/>`_ graph.
TODO

Configuration
=============
