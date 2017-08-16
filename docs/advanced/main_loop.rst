Main Loop
*********

Main loop is the core of the cxflow.
As the name suggests, it is responsible for the main lifecycle of the training of the model.

Training Lifecycle
==================

The lifecycle of the training is very simple.
Initially, the extra streams (i.e. usually valid and test streams) are evaluated.
At this point, the model is not updated.
Therefore, it is perfectly fine to use validation and testing data.

Secondly, the training loop starts.
In the training loop, two following steps alternate: updating model based on the training data and evaluating
the extra streams.

The whole process might be described by pseudocode as follows.

.. code-block:: bash

    1. Evaluate extra streams
    2. while not interrupted:
    3.     train on train stream
    4.     evaluate extra streams

Prediction Lifecycle
====================

The lifecycle of the prediction is analogous.
The main differences are that the prediction stream is used instead of the training stream and the model is never
updated.
In addition, there is only a single pass throughout the stream.

The whole process might be described by pseudocode as follows.

.. code-block:: bash

    1. Evaluate prediction streams
    2. Evaluate extra streams

Dataset/Model Integration
=========================

When main loop is about to either train or evaluate a stream, the following process is initiated.
Firstly, the stream is created by calling the proper method of the corresponding dataset.

Then, the stream is iterated through so that a batch is obtained in each iteration.
When the batch is fetched, the model's method `run` is invoked with two parameters.
The former is the batch, the latter is a boolean value indicating whether the model should update or not depending on
whether training stream is processed or not.

The implementation of `run` method is backend specific.

Hook Integration
================

Main loop is responsible for triggering the registered hooks.
There are multiple event for which the main loop triggers the hook.
More details might be found in the hooks section.

The events are as follows:

- **Before training** is triggered before first epoch, i.e. even before the extra streams are evaluated.
- **After batch** is triggered after each batch is processed.
- **After epoch** is triggered at the end od each epoch.
- **After training** is triggered after the final epoch, i.e. right before the training process terminated.
- **After training profile** is triggered after each epoch and is passed profiling information.
