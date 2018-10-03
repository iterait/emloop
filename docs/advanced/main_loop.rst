Main Loop
*********

Main loop is the core of **emloop** responsible for
the main lifecycle of model training.

Dataset/Model Integration
=========================

No matter what command has been executed, **emloop** always starts by creating
the dataset and passing the ``dataset`` section from the `config <config.html>`_ 
to its constructor. Consequently, **emloop** continues with creating the model and passing the
``model`` section from the `config <config.html>`_ to its constructor.
One of the arguments of the model is also the dataset itself, so the model can query
it for information such as the number of outputs, data size, etc.

After the dataset and the model are created, **emloop** calls one of the dataset stream functions
based on whether it is training, testing or evaluation (more on this is described in the following
sections).
The selected stream method (e.g., ``train_stream``) returns an iterable where each
item corresponds to a single batch of data.

Finally, the stream is iterated through and each batch is fed to the model's :py:meth:`emloop.models.AbstractModel.run`
method along with a boolean value indicating whether the model should update or not.
By default, updates only happen when iterating the training stream.

.. note::
    The implementation of :py:meth:`emloop.models.AbstractModel.run` method is backend specific.

Training Lifecycle
==================

The lifecycle of the training is very simple. Here are the steps performed by 
**emloop**.

#. Create the dataset and pass the ``dataset`` section from the `config
   <config.html>`_ to its constructor.
#. Create the model an pass the ``model`` section from the `config
   <config.html>`_ to its constructor.
#. Evaluate the extra streams. Those are usually valid and test streams, depending on
   your ``main_loop.extra_streams`` `config <config.html>`_.
   During this phase, the model is not updated and therefore, it is perfectly fine
   to use validation and testing data.
#. Start the training loop.
   In the training loop, the two following steps alternate: update the model based on the training data
   and evaluate the extra streams.

The whole process can be described by the following pseudocode.

.. code-block:: bash

    1. create dataset
    2. create model
    3. evaluate all the streams
    4. while not interrupted:
    5.     train on train stream
    6.     evaluate extra streams


Evaluation Lifecycle
====================

Any stream may be evaluated leaving the model intact. There is only a single pass through the stream before emloop
terminates.

The whole process can be described by the following pseudocode.

.. code-block:: bash

    1. create dataset
    2. restore model
    3. evaluate the specified stream

Hook Integration
================

The main loop is responsible for triggering the registered hooks.
There are multiple events for which the main loop triggers the hook.

The events are as follows:

- **Before training** is triggered prior to the first epoch, i.e. even before 
  the extra streams are evaluated.
- **After batch** is triggered after each processed batch.
- **After epoch** is triggered at the end of each epoch.
- **After training** is triggered after the final epoch, i.e. right before the training process terminates.
- **After training profile** is triggered after each epoch and receives 
  profiling information.

More details might be found in the `hooks section <hook.html>`_.
