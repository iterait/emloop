Python API
**********

Besides the CLI utilities, emloop also provides Python API for nonstandard techniques and rapid experimentation.
However, the users are encouraged to use the CLI interface and standard techniques whenever possible.

Before running the training, there are five things that need to be initialized: the `model <model.html>`_, `dataset <dataset.html>`_, `hooks <hooks.html>`_, output directory, and the `main loop <main_loop.html>`_. The following functions make the process straightforward:

- :py:meth:`emloop.api.create_output_dir`
- :py:meth:`emloop.api.create_dataset`
- :py:meth:`emloop.api.create_model`
- :py:meth:`emloop.api.create_hooks`
- :py:meth:`emloop.api.create_main_loop`

All these functions require a configuration dictionary that can be retrieved e.g., from the usual `YAML config file <config.html>`_ via :py:meth:`emloop.api.load_yaml`.

When everything is properly initialized, the main loop object behaves as a `context manager <https://docs.python.org/3/reference/datamodel.html#context-managers>`_ and it can run a single training epoch or a single evaluation epoch using its methods :py:meth:`emloop.main_loop.MainLoop.epoch`.

The following example demonstrates how to evaluate the model every tenth epoch of its training:

.. code-block:: python

    import emloop as el

    config = el.load_yaml("config.yaml")

    with el.create_main_loop(config, "./log") as main_loop:
        for epoch in range(100):
            if epoch % 10 == 0:
                main_loop.epoch(train_streams=["train"], valid_streams=["valid"])
            else:
                main_loop.epoch(train_streams=["train"], valid_streams=[])

Streams passed to method :py:meth:`emloop.main_loop.MainLoop.epoch` can be strings (as in the previous example) or any Iterable objects.

Following example shows training on list of objects.

.. code-block:: python

    import emloop as el

    config = el.load_yaml("config.yaml")

    batch1 = {"house_size" : [131, 96], "price" : [8000000, 6600000]}
    batch2 = {"house_size" : [152, 127], "price" : [13000000, 8100000]}

    with el.create_main_loop(config, "./log") as main_loop:
        main_loop.epoch(train_streams=[batch1, batch2],
                        valid_streams=[])

Note that `house_size` and `price` must be inputs of the model declared in the `config.yaml`.
