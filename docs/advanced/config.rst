Configuration
*************

Configuration is a crucial part of **cxflow** linking all the components together with just a few lines of YAML.

If you aren't comfortable with YAML, JSON documents are supported as well (YAML is actually a superset of JSON).
However, JSON does not support advanced features such as `anchors <https://learnxinyminutes.com/docs/yaml/>`_
or simple comments.

Each configuration file is divided into following sections:

- ``dataset``
- ``model``
- ``hooks``
- ``main_loop``
- ``eval``

wherein only ``dataset`` and ``model`` sections are mandatory.

Dataset
=======

The dataset section contains, unsurprisingly, the dataset configuration. First of all, one need to specify the
fully qualified name of the dataset in the ``class`` entry.

All the remaining parameters are passed to the dataset constructor in the form of a string-encoded YAML.

The common dataset arguments can include augmentation setting for the training 
stream, the data root directory or a batch size, to name but a few.

.. code-block:: yaml
    :caption: example dataset configuration

    dataset:
      class: datasets.my_dataset.MyDataset

      data_root: /var/my_data
      batch_size: 16
      augment:
        rotate: true     # enable random rotations
        blur_prob: 0.05  # probability of blurring

See `dataset documentation <dataset.html>`_ for more information.

Model
=====

The model section specifies the configuration of the model to be trained.
Again, it contains a ``class`` that **cxflow** uses for constructing the model.

In addition, ``inputs`` and ``outputs`` entries are required as well.
These arguments define what sources will be obtained from the dataset stream and which will
be provided by the model.
The remaining parameters are directly passed to the model constructor from where they might
(or might not) be used.

For example, an image-processing deep neural network might require the input image resolution,
the number of hidden neurons or a learning rate as in the following example:

.. code-block:: yaml
    :caption: model configuration with additional parameters

    model:
      class: models.my_model.MyModel
      inputs: [image, animal]
      outputs: [prediction, loss, accuracy]

      width: 800
      height: 600
      learning_rate: 0.003
      n_hidden_neurons: 512

.. note::

    Note that in addition to the passed variables from the configuration, the
    dataset object is automatically passed to the model.
    Therefore, the model might use attributes and methods of the dataset as well.
    For example, the dataset might compute the number of target classes and the network can build the
    classifier in a way that suits the given dataset.
    Another example from natural language processing would be a number of distinct tokens for which the network
    train their embeddings.


See `model documentation <model.html>`_ for more information.

Hooks
=====

The hooks section is optional yet omnipresent. It contains a list of hooks to be registered in the main loop (see
`main loop <main_loop.html>`_ documentation) to save your model, terminate the training, visualize results etc.

Hooks are specified by their fully qualified names as in the following example:

.. code-block:: yaml
    :caption: non-parametrized hooks

    hooks:
      - LogProfile  # cxflow.log_profile.LogProfile
      - cxflow_tensorflow.TensorBoardHook
      - my_hooks.my_hook.MyHook

.. tip::

    Standard **cxflow** hooks from ``cxflow.hooks`` module may be referenced only by their names,
    e.g.: ``LogProfile`` instead of ``cxflow.log_profile.LogProfile``.

In some cases, we need to configure the hooks being created with additional parameters. To do so, simply define a
dictionary of parameters which will be passed to the hook constructor. E.g.:

.. code-block:: yaml
    :caption: parametrized hooks

    hooks:
      - cxflow_scikit.ClassificationInfoHook:
          predicted_variable: predictions
          gold_variable: labels

      - ComputeStats:
          variables: [loss]

Main Loop
=========

The ``main_loop`` section is optional. Any parameter specified there is forwarded to the
:py:class:`cxflow.MainLoop` constructor which takes the following arguments:

.. automethod:: cxflow.MainLoop.__init__

.. code-block:: yaml
    :caption: main loop configuration

    main_loop:
      extra_streams: [valid, test]
      skip_zeroth_epoch: True

Evaluation
==========

Naturally, the evaluation (sometimes referred as *prediction* or *inference*)
of the model on new unannotated data differs from its training.
In this phase, we don't know the ground truth, hence the dataset sources are different.
In such a situation, some of the metrics are impossible to measure, e.g. accuracy, which requires the
ground truth. Most likely, we also need a different set of hooks to process the model outputs.

For this reason, one can override the configuration with a special ``eval`` section.
For each data stream, a sub-section (e.g.: ``eval.my_stream``) is expected to match the overall configuration structure,
i.e. it **may** contain the ``model``, ``dataset``, ``hooks`` and/or  ``main_loop`` sections.

In the following example, we use all the original settings but the model inputs and outputs are overridden. Furthermore,
a different list of hooks is specified. Yet another example is available in our
`examples repository @GitHub <https://github.com/Cognexa/cxflow-examples/tree/master/imdb>`_.

.. code-block:: yaml
    :caption: eval section of **cxflow** configuration

    ...
    eval:
      predict:  # configuration for the predict_stream
        model:
          inputs: [images]
          outputs: [predictions]

        hooks:
          - hooks.inference_logging_hook.InferenceLoggingHook:
              variables: [ids, predictions]

Evaluation of the predict stream can then be invoked with:

```
cxflow eval predict path/to/model
```

Conclusion
==========

The main motivation for this type of configuration is its modularity.
The developer might easily produce various general models that will be trained or evaluated
on different datasets, just by changing a few lines in the configuration file.

With this approach, the whole process of developing machine learning models is modularized.
Once the interface (the names and the types of the data sources) are defined, the development
of the model and the dataset might be done separately.
In addition, the individual components are resusable for further experiments.

Furthermore, the configuration is backed up to the log directory.
Therefore, it is clear what combination of model and dataset was used in the experiment,
including all the parameters.

By registering custom hooks, the training and inference process might be arbitrarily
changed. For instance, the results may be saved into a file/database, or they can be streamed
