Hooks
*****

In this short tutorial, we learn how to use standard cxflow hooks and even how to write new ones.

Cxflow hooks allow to observe, modify and act upon the training process.
Hooks shall do their work in one of the following events invoked by the cxflow `main loop <main_loop.html>`_:

- **before_training** invoked once before entering the training loop, `no args`
- **after_batch** invoked after each batch regardless of the stream, `(stream_name, batch_data)`
- **after_epoch** invoked after each epoch, `(epoch_id, epoch_data)`
- **after_epoch_profile** special event with training profiling data, invoked after each epoch, `(epoch_id, epoch_profile)`
- **after_training** invoked once after the trainig finishes, `no args`

Before we dig into details, we peek on how to use some of the standard hooks available in cxflow framework.

In your `configuration <config.html>`_, hooks are listed under `hooks` entry, for example:

.. code-block:: yaml

    hooks:
      - ComputeStats:
          variables:
            loss: [mean]

      - LogVariables

would instruct cxflow to create two hooks which will keep track of the mean loss during the training.
In fact, the `ComputeStats` stores the loss from every batch and means the accumulated values after
each epoch.
Subsequently, the `LogVariables` logs all the variables available in the `epoch_data`, which
in the example above is only the mean loss computed by `ComputeStats` hook.

The names of the hooks are nothing more than the names of their respective classes.
For hooks that are built-in inside cxflow, only the class name needs to be specified,
however, for hooks outside of cxflow, you also have to specify their module. For instance,
for a class called `MyHook` inside a module `my_project.hooks`, you would write:

.. code-block:: yaml

    hooks:
      - my_project.hooks.MyHook

Any additional arguments specified in the config file are passed to the hook's constructor.
For instance, the following config:

.. code-block:: yaml

    hooks:
      - my_project.hooks.MyHook:
          arg1: 10
          arg2: ['a','b']

will be roughly translated to

.. code-block:: python

    from my_project.hooks import MyHook
    hook = MyHook(arg1=10, arg2=['a', 'b'])
    # use hook in the cxflow main_loop

In addition to the specified arguments, cxflow supplies the constructor with the model,
the dataset and the log output directory.
Hence, the hook creation looks actually more like this:

.. code-block:: python

    hook = MyHook(model=model, dataset=dataset, output_dir=output_dir, arg1=10, arg2=['a', 'b'])

Every hook may override any of the event handling methods specified above. Some hooks may be quite simple.
For example, a hook that would stop the training after the specified number of epochs can be written as follows:

.. code-block:: python

    import logging
    from cxflow.hooks.abstract_hook import AbstractHook, TrainingTerminated

    class EpochStopperHook(AbstractHook):
        def __init__(self, epoch_limit: int, **kwargs):
            super().__init__(**kwargs)
            self._epoch_limit = epoch_limit

        def after_epoch(self, epoch_id: int, **kwargs) -> None:
            if epoch_id >= self._epoch_limit:
                logging.info('EpochStopperHook triggered')
                raise TrainingTerminated('Training terminated after epoch {}'.format(epoch_id))

Now, lets take a closer look on the `after_batch` and `after_epoch` events where the majority
of hooks will operate.

`after_batch` event
===================

This event is invoked after every batch regardless of what stream is being processed.
In fact, the stream name will be available in the `stream_name` argument.

The second and last argument named `batch_data` is a dict of stream sources and model outputs.

Imagine a dataset that provides streams with two sources, `images` and `labels` and a model which
takes the `images` and outputs its own `preditions`.
In this case, the `batch_data` would contain the following dict

.. code-block:: python

    {
      'images': ['1st image', '2nd image'...],
      'labels': [5, 2,...],
      'prediction': [5, 1,...]
    }

Now, the hook decides how to process this data. Usually, it is useful to accumulate the data over
the whole epoch and process them in the `after_epoch` event all at once.
Luckily, you do not have to implement this behavior on your own, it is already
available in our :py:class:`cxflow.hooks.AccumulateVariables` hook from which
you may derive your own hook.

`after_epoch` event
===================

The `after_epoch` event is even more simple.
The event accepts two arguments, `epoch_id`, representing the epoch number, and
`epoch_data`, which is an object shared between the hooks.

Initially, the `epoch_data` object is an empty dict with stream name entries.
E.g., with train, valid and test streams it initially looks as following:

.. code-block:: python

    {
      'train': {},
      'valid': {},
      'test': {}
    }

Now, for instance, our `ComputeStats` from the first example computes the mean over the
accumulated loss data and stores the result to the given `epoch_data`. So after
the `ComputeStats` hook has been called, the `epoch_data` will look as follows:

.. code-block:: python

    {
      'train': {'loss': {'mean': 0.2}},
      'valid': {'loss': {'mean': 0.32}},
      'test': {'loss': {'mean': 0.35}
    }

The `LogVariables` already expects this structure and logs everything it gets.

**Note that the hooks order matters! We would see nothing with the `LogVariables` placed
before the `ComputeStats`.**

Regular hook configuration
==========================

Altogether, the hook system provides instruments to carefully watch and manage your training.

A good starting point for your own hook configuration may be the following config:

.. code-block:: yaml
  
    hooks:
      # compute classification statistics such as accuracy of f1 score
      - cxflow_scikit.ClassificationInfoHook:
          predicted_variable: predictions
          gold_variable: labels

      # compute mean loss over each epoch
      - ComputeStats:
          variables:
            loss: [mean]

      # log the results to the standard python logging, csv and tensorboard
      - LogVariables
      - WriteCSV
      - LogProfile
      - cxflow_tensorflow.hooks.WriteTensorboard

      # save the best model
      - SaveBest

      # allow interrupting with CTRL+C
      - CatchSigint

      # stop after 100 epochs
      - StopAfter:
          epochs: 100
