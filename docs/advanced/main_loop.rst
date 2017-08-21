Main Loop
*********

Main loop is the core of the cxflow.
As the name suggests, it is responsible for the main lifecycle of the training of the model.

Dataset/Model Integration
=========================

The first thing cxflow does no matter what command has been executed, is to build the dataset
and pass the section `dataset` from the `config <config.html>`_ to its constructor.
Afterwards, cxflow builds the model and passes the `model` section from the
`config <config.html>`_ to its constructor.
One of the model's arguments is also the dataset itself, so the net can query
it for some information, such as the number of outputs, data size, etc.

After the dataset and the model are created, cxflow calls one of the dataset stream functions
based on whether it is training, testing or predicting (more on this is described in the following
sections).
The selected stream method (e.g., `train_stream`) returns an iterable where each
item corresponds to a single batch of data.

Finally, the stream is iterated through and each batch is fed to the model's `run` method along
with a boolean value indicating whether the model should update or not.
By default, updates only happen when iterating the training stream.
The implementation of `run` method is backend specific.

Training Lifecycle
==================

The lifecycle of the training is very simple. The following are the steps performed by cxflow.

#. Build the dataset and pass the section `dataset` from the `config <config.html>`_ to its constructor.
#. Build the model an pass the section `model` from the `config <config.html>`_ to its constructor.
#. Evaluate the extra streams. Those are usually valid and test streams, depending on 
   your `main_loop.extra_streams` `config <config.html>`_.
   During this phase, the model is not updated and therefore, it is perfectly fine
   to use validation and testing data.
#. Start the training loop.
   In the training loop, the two following steps alternate: update the model based on the training data
   and evaluate the extra streams.

The whole process might be described by the following pseudocode.

.. code-block:: bash

    1. build dataset
    2. build model
    3. evaluate extra streams
    4. while not interrupted:
    5.     train on train stream
    6.     evaluate extra streams

Test Lifecycle
====================

TODO rework based on issue regarding prediction and testing.

Prediction Lifecycle
====================

The lifecycle of the prediction is similar.
The main differences are that only the prediction stream is used and the model is never updated.
In addition, there is only a single pass through the stream and then cxflow terminates.

The whole process might be described by the following pseudocode.

.. code-block:: bash

    1. build dataset
    2. build model
    1. evaluate prediction stream

Hook Integration
================

Main loop is responsible for triggering the registered hooks.
There are multiple events for which the main loop triggers the hook.

The events are as follows:

- **Before training** is triggered before the first epoch, i.e. even before the extra streams are evaluated.
- **After batch** is triggered after each processed batch.
- **After epoch** is triggered at the end od each epoch.
- **After training** is triggered after the final epoch, i.e. right before the training process terminates.
- **After training profile** is triggered after each epoch and it is passed profiling information.

More details might be found in the `hooks section <hook.html>`_.
