Hooks
*****

In this short tutorial, we learn how to use standard cxflow hooks and even how to write new ones.

CXflow hooks allow to observe, modify and act upon the training process. Hooks shall do their work in one of the following events invoked by the cxflow main loop:

- **before_training** invoked once before entering the training loop, `no args`
- **after_batch** invoked after each batch regardless the stream, `(stream_name, batch_data)`
- **after_epoch** invoked after each epoch, `(epoch_id, epoch_data)`
- **after_epoch_profile** special event with training profiling data, invoked after each epoch, `(epoch_id, epoch_profile)`
- **after_training** invoked once after the trainig finishes, `no args`

Before we dig into details, we peek on how to use some of the standard hooks available in cxflow framework.

In your configuration, hooks are listed under `hooks` entry, for example:

.. code-block:: yaml

    # net & dataset configurations
    hooks:
      - class: StatsHook
        variables:
          loss: [mean]

      - class: LoggingHook

would instruct cxflow to create two hooks which will keep track of the mean loss during the training.
In fact, the `StatsHook` stores the loss from every batch and means the values after each epoch.
Subsequently, the `LoggingHook` logs all the variables available in the `epoch_data`, that is the mean loss in our case.

Note that we had to specify the exact class name of the hook to be constructed.
As the hook modules were not specified explicitly, cxflow assumes that the hook classes can be found under `cxflow.hooks` module.

Any additional arguments are passed to the hook `__init__` method. The following config

.. code-block:: yaml

    hooks:
      - class: MyHookClass
        module: my_project.my_module
        arg1: 10
        arg2: ['a','b']

will be roughly translated to

.. code-block:: python

    from my_project.my_module import MyHookClas
    hook = MyHookClass(arg1=10, arg2=['a', 'b'])
    # use hook in the cxflow main_loop

In addition to the specified args, cxflow supplies the `__init__` method with the previously created net, dataset and output directory.
Hence, the hook creation looks more like this:

.. code-block:: python

    hook = MyHookClass(net=net, dataset=dataset, output_dir=output_dir, arg1=10, arg2=['a', 'b'])

Every hook may override any of the event handling method specified above. Some hooks may be quite simple.
For example, a hook that would stop the training after the specified number of epochs can be written as follows:

.. code-block:: python

    import logging
    from cxflow.abstract_hook import AbstractHook, TrainingTerminated

    class EpochStopperHook(AbstractHook):
        def __init__(self, epoch_limit: int, **kwargs):
            super().__init__(**kwargs)
            self._epoch_limit = epoch_limit

        def after_epoch(self, epoch_id: int, **kwargs) -> None:
            if epoch_id >= self._epoch_limit:
                logging.info('EpochStopperHook triggered')
                raise TrainingTerminated('Training terminated after epoch {}'.format(epoch_id))

Now, lets take a closer look on the after_batch and after_epoch events where the majority of hooks will operate.

`after_batch` event
===================

This event is invoked after every batch regardless of what stream is being processed.
In fact, the stream name will be available in the `stream_name` argument.

The second and last argument named `batch_data` is a dict of stream sources and net outputs.

Imagine a dataset that provides streams with `images` and `classes` sources and a net which takes the `images` and outputs its own `preditions`.
In this case, the `batch_data` would contain the following dict

.. code-block:: python

    {
     'images': ['1st image', '2nd image'...],
     'classes': [5, 2,...],
     'prediction': [5, 1,...]
    }

Now, the hook decides how to process this data. It may be useful to accumulate the data over the whole epoch and process them in the after_epoch event.
Luckily, you do not have to implement this behavior on your own, it is already available in our `AccumulatingHook` from which you may derive your own hook.

`after_epoch` event
===================

The after epoch event is even more simple. The event identifies the epoch with the `epoch_id` argument and provides an `epoch_data` object to share the computed data between the hooks.

Initially, this object is an empty dict with stream name entries. E.g., with train, valid and test streams we get

.. code-block:: python

    {
     'train': {},
     'valid': {},
     'test': {}
    }

Now our `StatsHook` from the previous example computes the mean over the accumulated loss data and stores the result to the given `epoch_data` which leaves us with

.. code-block:: python

    {
     'train': {'loss': {'mean': 0.2}},
     'valid': {'loss': {'mean': 0.32}},
     'test': {'loss': {'mean': 0.35}
    }

The `LoggingHook` already expects this structure and logs everything it gets.

**Note that the hooks order matters! We would see nothing with the `LoggingHook` placed before the `StatsHook`.**

## Regular hook configuration
Altogether, the hook system provides instruments to carefully watch and manage your training.

A good starting point for your own hook configuration may be the following config:

.. code-block:: yaml
  
    hooks:
      # compute classification statistics such as accuracy of f1 score
      - class: ClassificationInfoHook
        predicted_variable: predictions
        gold_variable: labels

      # compute mean loss over each epoch
      - class: StatsHook
        variables:
          loss: [mean]

      # log the results to the standard python logging, csv and tensorboard
      - class: LoggingHook
      - class: CSVHook
      - class: ProfileHook
      - class: TensorBoardHook

      # save the best model
      - class: BestSaverHook

      # allow interrupting with CTRL+C
      - class: SigintHook

      # stop after 100 epochs
      - class: EpochStopperHook
        epoch_limit: 100
