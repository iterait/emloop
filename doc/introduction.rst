Introduction
============

Cxflow is a library for management and training of machine learning models.
Cxflow does not implement any building blocks (contrary to, e.g.,
`Keras <https://github.com/fchollet/keras>`_). Instead, you can directly use
your favourite machine learning framework, such as `TensorFlow <https://www.tensorflow.org/>`_,
`CNTK <https://cntk.ai/>`_, or `Caffe2 <https://caffe2.ai/>`_. That means that
you don't have to learn a new language or a new framework if you already know one
and you can convert the models you already have with only minimal changes.

In machine learning tasks, 
As it usually is with machine learning tasks, you have to The main motivation of cxflow is to avoid the making 
With cxflow

Getting Started
===============

To use cxflow for your project, there are four fundamental components that
need to be set up:

1. Dataset
2. Net
3. Configuration
4. Hooks

These components will be briefly discussed in the following text. 

The components except
Configuration are independent on each other and might be reused across various applications.
