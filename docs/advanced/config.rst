Configuration
*************

Configuration is the key part of cxflow.
It is possible to link a dataset and appropriate model by just a few lines of a YAML document.

If you aren't comfortable with YAML, JSON documents are supported as well.
However, advanced features such as `anchors <https://learnxinyminutes.com/docs/yaml/>`_ or simple comments will not
be enabled.

Each configuration file is structured in the following sections.

- `dataset`
- `model`
- `hooks`
- `main_loop`
- `predict`

While `dataset` and `model` sections are mandatory, the remaining ones are not.
Nevertheless, `hooks` are strongly recommended to be specified.

Dataset
=======

Dataset section specifies the configuration of the dataset.
It consists of `module` and `class` which are employed by cxflow in order to construct the dataset object.
The remaining parameters are passed to the dataset constructor in form of string-encoded YAML.

See `dataset documentation <dataset.html>`_ for more information.

The common dataset arguments might include e.g. setting of the augmentations for training stream, directory from whence
the data will be loaded or batch size.

Example:

.. code-block:: yaml

    dataset:  &dataset
      module: datasets.my_dataset
      class: MyDataset

      data_root: /var/my_data
      batch_size: 16
      augment:
        rotate: true     # enable random rotations
        blur_prob: 0.05  # probability of blurring

Model
=====

Model section specifies the configuration of the model to be trained or inferred.
Again, it consists of `module` and `class` which are employed by cxflow in order to construct the model object.
In addition, `inputs` and `outputs` are required.
These arguments define what sources will be obtained from the dataset stream and which will be provided by the model.
The remaining parameters are directly passed to the model constructor from whence they might be used.

See `model documentation <model.html>`_ for more information.

The common arguments might include model-specific parameters.
For example, an image-processing deep neural network might require input image resolution, number of hidden neurons
or learning rate.

Example:

.. code-block:: yaml

    model:  &model
      module: models.my_model
      class: MyModel
      inputs: [image, animal]
      outputs: [prediction, loss, accuracy]

      width: 800
      height: 600
      learning_rate: 0.003
      n_hidden_neurons: 512

Note that in addition to the passed variables from the configuration, the model is automatically passed the dataset
object.
Therefore, it might use dataset attributes and methods.
For example, dataset might compute the number of target classes and the network can build the classifier in a way it
suits every dataset.
Another example (NLP motivated) would be a number of distinct tokens for which the network train their embeddings.


Hooks
=====

Hooks section is an optional but suggested section.
The section contains a list of hooks to be registered in the main loop (see `main loop <main_loop.html>`_ documentation).
The list contains two types of entries.

First, only the hook name might be specified.
In case the hook name is fully specified, e.g. `module.class`, this hook will be loaded.
If only class name is provided, e.g. `class`, the module will be automatically inferred from the `cxflow.hooks` module.

Example:

.. code-block:: yaml

    hooks:
      - LogProfile  # cxflow.log_profile.LogProfile
      - cxflow_tensorflow.TensorBoardHook
      - my_hooks.my_hook.MyHook

Second, a dict in the form of `Name -> {config}` might be specified.
Again, name can be either standard hook name or a fully qualified name in the form of `module.class`.
The nested config is passed to directly to the hook constructor.

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
The reason for this approch is that the parameter-less hooks or the ones with sufficient default values can be registred
extremely easily.
On the contrary, hooks might be configured as required.

Main Loop
=========

Main loop section specifies various behavior of the main loop.
Currently, the following parameters are supported.

- `extra_streams`: a list of additional streams that will be evalueted during training or inferred during `predict`.
- `on_unused_sources`: behavior of the main loop when the dataset provides additional sources to the ones the model
                       requires. By default (`warn`), main loop warns the developer. Remaining options are `ignore`
                       which suppresses the warning and `error` which terminates the process immediately.
- `fixed_batch_size`: if this option is specified, the main loop will enforce the batches flowing to the model will
                      contain exactly the specified number of examples. The incorrectly sized batches will be ignored
                      and the warning will be provided.
- `skip_zeroth_epoch`: if set to `True`, the evaluation before the first training epoch will be skipped.


.. code-block:: yaml

    main_loop:
      extra_streams: [valid, test]
      skip_zeroth_epoch: True

Prediction
==========
Naturally, the prediction (inference) of the whole system differ from the training time.
In prediction phase, we don't know the ground truth, hence the dataset sources are different.
This automatically lead to a fact, that not all metrics are suitable to be measured, e.g. accuracy which requires the
ground truth.

For this reason, a special section `predict` is introduced.
It matches the overall configuration structure, i.e. it must contain `model` and `dataset` sections.
Analogously,`hooks` section is optional as well as `main_loop`.

If `cxflow predict` is invoked, the rest of the configuration is ignored and only the `predict` section is used.
In other cases, the prediction section is ignored.
The main advantage of this approach is that the user doesn't have to define `predict` when they experiment with the
models.
This can be done after the model is developed, fine-tuned and ready for production.

As it might be observed, the predict sections such as `model` and `dataset` are almost identical to the top level ones.
YAML can reduce configuration duplicity by using the anchors.
Note that we've already defined anchors `&dataset` and `&model` in the snippets above.

Now, we can import them and rewrite only arguments which differ.
In the following example, we reuse the whole dataset as is.
The model is almost the same but we need to specify different `inputs` and `outputs` since the prediction stream will
no longer provide the target class (`animal`).
The model supposed to do it instead.
Finally, we define a completely different set of hooks.

.. code-block:: yaml

    predict:
      dataset:
        <<: *dataset

      model:
        <<: *model
        inputs: [images]
        outputs: [predictions]

      hooks:
        - hooks.prediction_logging_hook.PredictionLoggingHook:
            variables: [ids, predictions]

Conclusion
==========

The main motivation for this type of configuration is its modularity.
The developer might easily produce various general models that will be trained or evaluated on various models just by
changing few lines in the configuration (dataset section).
Analogously, a single dataset might be approached by different models just by changing the model section.

By this approach, the whole process of developing machine learning models is modularized.
Once the interface (names and types of the data sources) are defined, the development of the model and the dataset might
be done separately.
In addition, the components are automatically resusable for further experiments.

The configuration is stored in the cxflow output directory in a simple YAML file.
Therefore, it is clear what combination of model (with precise parameters) and dataset (again, with precise parameters)
was used in the experiment.

By registering a custum hook, the progress of the training might be saved into a database of the intermediate
results/artifacts can be saved to a hard-drive or automatically deployed.
