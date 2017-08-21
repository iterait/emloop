Model
*****

Model is the second component of the cxflow environment.
It defines the machine learning part of the whole workflow.
The model object is defined by `cxflow.AbstractModel` API.

Firstly, the model constructor accepts a dataset instance, path to the logging directory and the information of
if (and from where) the model should be restored.

Secondly, `input_names` and `output_names` properties must be defined.
These properties return the corresponding lists of input and output variable names.
The input names are expected by the model, i.e. must be keys of the given batches.
The output names are the variables which the model computes and outputs.
In the case of our animal recognition example, the input names would contain `image` and `animal` while the output
names would contain `predicted_animal` and `loss`.

Running the Model
-----------------

The most important method of the model is `run`.
This method simulated an execution of the model on a single batch which is the first parameter.
The second parameter is a boolean variable determining whether the model will update (train) on this batch or not.

Note that the model is not persistent.
The persistence of the model might be ensured by invoking the `save` method which dumps the model (usually) to the
filesystem.
The method requires a suffix of the dumped file.

The pseudocode of the model training, evaluation and saving follows.
Note that this loop is automatically managed by `cxflow.MainLoop` and we publish this snippet just in order to
demonstrate the process.

.. code-block:: python

    for epoch_id in range(10):
        for train_batch in dataset.train_stream():
            model.run(batch=train_batch, train=True)

        for test_batch in dataset.test_stream():
            model.run(batch=test_batch, train=False)

        model.save(name_suffix=str(epoch_id))


Restoring the Model
-------------------

Once the model is successfully saved, it might be also restored.
This is done when the training is about to continue (`cxflow resume`) or for production environemt (`cxflow predict`).
Both commands expect the `restore_from` positional argument which specify the backend-agnostic argument.
This argument is passed to the model constructor (see above).

If this parameter is passed, the network attempt to restore the model parameters by employing the information passed
in `restore_from`.
The detail behavior is backend-specific, however, in most cases it uses the backend saving API.

In order to restore the model, cxflow needs to know what model object should be constructed first.
This information is inferred from the dumped configuration file in the output directory.
However, there are cases in which the model object cannot be constructed (somebody deleted source codes with the
model object implementations etc.).
For these cases, so called `restore_fallback` property is defined by each model.
This is usually the backend-specific baseclass which is able to restore all its subclasses.
