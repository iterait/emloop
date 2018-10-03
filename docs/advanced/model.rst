Model
*****

The model is the second component of the **emloop** environment.
It defines the machine learning part of the whole workflow.
The model object is defined by the :py:class:`emloop.models.AbstractModel` interface.

The model constructor accepts a dataset instance, path to the logging directory and the information whether
(and from where) the model should be restored or whether a new one should be created.

Additionally, every model has to define two properties, ``input_names`` and ``output_names``.
These properties should return the corresponding lists of input and output variable names.
The input names are expected to exist in every batch as the keys to the batch dictionary.
The output names are the variables, which the model computes and outputs, and which may be used
by statistical hooks and alike.
In the case of our animal recognition example from the `dataset <dataset.html>`_ tutorial,
the input names would contain ``image`` and ``animal``, while the output
names would contain ``predicted_animal`` and ``loss``.

Running the Model
-----------------

The most important method of the model is the :py:meth:`emloop.models.AbstractModel.run`.
This method evaluates the model on a single batch given as the first parameter.
The second parameter is a boolean variable determining whether the model should update (train) on this batch or not.

Note that the trained model is not persistent, as it is only stored in the memory.
The persistence of the model is provided by the :py:meth:`emloop.models.AbstractModel.save` method, which dumps the model to the
filesystem (although this behavior is model-specific and you may implement it as you wish in you own models).
The :py:meth:`emloop.models.AbstractModel.save` method shall accept only a single parameter -the name for the dumped file(s).

The pseudocode of model training, evaluation and saving may look as follows.
Note that this loop is automatically managed by :py:class:`emloop.MainLoop` and we publish this snippet just in order to
demonstrate the process.

.. code-block:: python

    # `model` construction should be here

    for epoch_id in range(10):
        for train_batch in dataset.train_stream():
            model.run(batch=train_batch, train=True)

        for test_batch in dataset.test_stream():
            model.run(batch=test_batch, train=False)

        model.save(name_suffix=str(epoch_id))


Restoring the Model
-------------------

Once the model is successfully saved, it can be also restored.
This is done when the training is about to continue (``emloop resume``)
or in a production environemt (``emloop eval <stream_name>``).
Both commands expect a single positional argument specifying from where the model shall be loaded.
This argument is called ``restore_from`` and it is passed to the model constructor (see below).

If the ``restore_from`` argument is passed to the constructor, the model attempts to restore itself.
Most often, it will consider the argument to be a file path and loads the file, yet the implementation
is model-specific and may be implemented differently.

To restore the model, **emloop** needs to know what class should be instantiated to be able to call
its constructor with the given ``restore_from`` argument.
The class is inferred from the dumped configuration file in the output directory, specifically from the
``model.class`` entry.
However, there are cases in which the original class cannot be constructed (somebody deleted source codes
with the model object implementations etc.).
For these cases, each model should implement a :py:meth:`emloop.models.AbstractModel.restore_fallback` method, which usually points
to a backend-specific base class, which is able to restore the saved files of all its subclasses.
For instance, in the ``emloop-tensorflow`` backend, the :py:meth:`emloop.models.AbstractModel.restore_fallback` class returns
:py:class:`emloop_tensorflow.BaseModel`, which it is able to load any checkpoint
without the need for the original model source codes.
