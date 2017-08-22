Configuration
*************

Configuration is the key part of cxflow.
It is possible to link a dataset and appropriate model by just a few lines of a YAML document.

If you aren't comfortable with YAML, JSON documents are supported as well.
However, JSON does not support advanced features such as `anchors <https://learnxinyminutes.com/docs/yaml/>`_
or simple comments.

Each configuration file is structured in the following sections.

- ``dataset``
- ``model``
- ``hooks``
- ``main_loop``
- ``infer``

While ``dataset`` and ``model`` sections are mandatory, the remaining ones are not.
Nevertheless, ``hooks`` are strongly recommended to be specified.

Dataset
=======

Dataset section specifies the configuration of the dataset.
It contains a ``class`` entry which points to the module and the class name of the dataset, separated
by a dot.
The remaining parameters are passed to the dataset constructor in the form of a string-encoded YAML.

See `dataset documentation <dataset.html>`_ for more information.

The common dataset arguments might include e.g., augmentation setting for the training stream,
the directory from whence the data will be loaded or a batch size.

Example:

.. code-block:: yaml

    dataset:  &dataset
      class: datasets.my_dataset.MyDataset

      data_root: /var/my_data
      batch_size: 16
      augment:
        rotate: true     # enable random rotations
        blur_prob: 0.05  # probability of blurring

Model
=====

Model section specifies the configuration of the model to be trained or inferred.
Again, it contains a ``class`` entry which is employed by cxflow in order to construct the model object.

In addition, ``inputs`` and ``outputs`` entries are required as well.
These arguments define what sources will be obtained from the dataset stream and which will
be provided by the model.
The remaining parameters are directly passed to the model constructor from whence they might
(or might not) be used. See `model documentation <model.html>`_ for more information.

The common arguments might include model-specific parameters.
For example, an image-processing deep neural network might require the input image resolution,
the number of hidden neurons or a learning rate.

Example:

.. code-block:: yaml

    model:  &model
      class: models.my_model.MyModel
      inputs: [image, animal]
      outputs: [prediction, loss, accuracy]

      width: 800
      height: 600
      learning_rate: 0.003
      n_hidden_neurons: 512

Note that in addition to the passed variables from the configuration, the model is automatically
passed the dataset object.
Therefore, it might use dataset attributes and methods as well.
For example, dataset might compute the number of target classes and the network can build the
classifier in a way it suits the given dataset.
Another example from natural language processing would be a number of distinct tokens for which the network
train their embeddings.

Hooks
=====

Hooks section is an optional but a suggested section.
The section contains a list of hooks to be registered in the main loop (see
`main loop <main_loop.html>`_ documentation).

The list contains two types of entries.
The first type specifies only the hook class name and optionally its module.
In the case there is only the class name, the hook will be automatically
loaded from the ``cxflow.hooks`` module.
In case the hook class is fully specified including the module, e.g. ``my_module.my_hook``,
this hook will be loaded as expected.

Example:

.. code-block:: yaml

    hooks:
      - LogProfile  # cxflow.log_profile.LogProfile
      - cxflow_tensorflow.TensorBoardHook
      - my_hooks.my_hook.MyHook

Second possible type of hook entries is a dict in the form of ``Hook -> {config}``.
Again, the ``Hook`` can be either standard hook name or a fully qualified name in the
form of ``module.class``.
The nested is passed to directly to the hook's constructor in the form of ``**kwargs``.

Example:

.. code-block:: yaml

    hooks:
      - cxflow_scikit.ClassificationInfoHook:
          predicted_variable: predictions
          gold_variable: labels

      - ComputeStats:
          variables:
          loss: [mean]

Both syntaxes might be mixed up arbitrarily.
The reason for this approch is that the parameter-less hooks or the ones with convenient
default values can be registred very easily.
However, if there is the need, hooks might be configured at will.

Main Loop
=========

Main loop section specifies various settings of the main loop.
Currently, the following parameters are supported.

- ``extra_streams``: A list of additional streams that will be evalueted during training or inferred
                   during ``cxflow infer``.
- ``on_unused_sources``: Behavior of the main loop when the dataset provides batches with sources not
                       registered in model's ``inputs``. By default (``warn``), main loop warns the developer.
                       Remaining options are ``ignore`` which suppresses the warning and ``error`` which
                       terminates the process immediately.
- ``fixed_batch_size``: If this option is specified, the main loop will enforce the batches fed to the model will
                      contain exactly the specified number of examples. Incorrectly sized batches will be skipped
                      with a warning.
- ``skip_zeroth_epoch``: If set to ``True``, the evaluation of ``extra_streams`` before the first training epoch will
                       be skipped.

Example:

.. code-block:: yaml

    main_loop:
      extra_streams: [valid, test]
      skip_zeroth_epoch: True

Inference
=========

Naturally, the inference (evaluation) of the model on new unanotated data differs from its training.
In this phase, we don't know the ground truth, hence the dataset sources are different.
In such a situation, some of the metrics are impossible to measure, e.g., accuracy which requires the
ground truth.

For this reason, a special section ``infer`` is introduced.
It matches the overall configuration structure, i.e. it must contain the ``model`` and the ``dataset`` sections.
Analogously, the ``hooks`` section is optional as well as ``main_loop``.

If ``cxflow infer`` is invoked, the rest of the configuration is ignored and only the ``infer`` section is used.
In other cases, the ``infer`` section is ignored.
The main advantage of this approach is that the user doesn't have to define ``infer`` when they experiment with the
models.
This can be done after the model is developed, fine-tuned and ready for production.

As it might be observed, the inference sections such as ``model`` and ``dataset`` are almost identical to the top level ones.
YAML can reduce configuration duplicity by using `anchors <https://learnxinyminutes.com/docs/yaml/>`_.
Note that we've already defined anchors ``&dataset`` and ``&model`` in the snippets above.

Now, we can import them and rewrite only the arguments which differ.
In the following example, we reuse the whole dataset as is.
The model is almost the same but we need to specify different ``inputs`` and ``outputs`` since the inference stream will
no longer provide the target class (``animal``).
The model itself is supposed to infer the ``animal`` instead.
Finally, we define a completely different set of hooks.

.. code-block:: yaml

    infer:
      dataset:
        <<: *dataset

      model:
        <<: *model
        inputs: [images]
        outputs: [predictions]

      hooks:
        - hooks.inference_logging_hook.InferenceLoggingHook:
            variables: [ids, predictions]

Conclusion
==========

The main motivation for this type of configuration is its modularity.
The developer might easily produce various general models that will be trained or evaluated
on different datsets just by changing a few lines in the configuration file.

By this approach, the whole process of developing machine learning models is modularized.
Once the interface (the names and the types of the data sources) are defined, the development
of the model and the dataset might be done separately.
In addition, the individual components are resusable for further experiments.

Furhtermore, the configuration is backed up to the log directory.
Therefore, it is clear what combination of model and dataset was used in the experiment,
including all the parameters.

By registering custum hooks, the training and inference process might be arbitrarily
changed. For instance, the results may be saved into a file/database, or they can be
deployed and served on you webpage.
