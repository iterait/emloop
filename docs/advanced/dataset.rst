Dataset
*******

Dataset is an object representing the data to be fed into the model.
In general, the dataset is an instance of arbitrary class which is able handle
a string-encoded YAML configuration in its constructor.
Note that this configuration is parsed from the `config file <config.html>`_ section ``dataset``.

BaseDataset
-----------

Inheriting from ``cxflow.BaseDataset`` is recommended as it parses the YAML string automatically.
Instead of the constructor, the developer is expected to implement ``_init_with_kwargs`` method.
This method is supposed to accept the parameters from the config file.

Let us demonstrate ``cxflow.BaseDataset`` on an example. First, we will create a class ``MyDataset`` located
in ``datasets.my_dataset`` module:

.. code-block:: python

     from cxflow import BaseDataset

     class MyDataset(BaseDataset):
         def _init_with_kwargs(batch_size: int, augment: dict, **kwargs):
           # ...

This class is interested in two arguments, ``batch_size`` and ``augment``. Any other argument
it is given is ignored and hidden in the ``**kwargs``.

Second, we define the ``dataset`` section in the config file:

.. code-block:: yaml

    dataset:
      module: datasets.my_dataset
      class: MyDataset
      batch_size: 16
      augment:
        rotate: true     # enable random rotations
        blur_prob: 0.05  # probability of blurring
      width: 800
      height: 600

Data Processing
---------------

First, let us define a **data source**.
We refer to the data source as to a unit (or type) of which the (training) example consists.
Let's take a look at an artificial image-classification task in which we are supposed to
classify the input image into various animal classes (dog, cat, rabbit, ...).
In this setting, the sources could be e.g., ``image`` and ``animal``.
The former is in the form of a tensor with the shape of 800x600x3 (i.e. a regular RGB 800x600 image).
The latter is a string describing the animal depicted in the image.

Second, let us define a **batch of data**.
We refer to a batch as to a collection of training examples.
Note that the examples in the batch are represented source-wise, hence it
is a Python ``dict`` mapping the source name to a collection of corresponding
source of examples.
The example of a single batch may look as follows (batch size is set ot four):

.. code-block:: python

    batch = {
      'image': [img1, img2, img3, img4],
      'animal': ['cat', 'cat', 'dog', 'rabbit']
    }

Finally, **data stream** is an iterable of batches.

The streams should be defined in dataset methods conveniently named ``<name>_stream``.

An example of a simple training stream is as follows.
The datasets contains a ``train_stream`` method which loads batches via function ``load_training_batch``.
For simplicity, we assume this function is already implemented.

.. code-block:: python

    def train_stream(self):
        for i in range(10):
            yield load_training_batch(num=i)

The training stream is used only when training is invoked either by ``cxflow train`` or ``cxflow resume`` commands.

Analogously, additional methods such as ``valid_stream`` and ``test_stream`` can be easily implemented.
If they are registered in the config file under ``main_loop.extra_streams``, they will be evaluated
along with the train stream. The configuration may look as follows:

.. code-block:: yaml

    main_loop:
      extra_streams: [valid, test]

The extra streams, however, *are not* used for training, that is, the model is not updated while iterating them.

During prediction (i.e., ``cxflow predict`` CLI command), only the ``predict_stream`` method is employed in order
to provide the data to be inferred.

TODO THE FOLLOWING SENTENCE MAY CHANGE.
Extra streams might be inferred as well when registred as described above (including training stream).

Additional Methods
------------------

The dataset may contain various additional methods as well.
For example, is can contain ``fetch`` method which checks whether the dataset has all the data it requires.
If not, it may download them from the internet/databse/drive.

Additional useful method could be ``statistics`` which prints various statistics of the
provided data, plot some figures etc.
Sometimes, we need to split the whole dataset into training, validation and testing sets.
For this purpose, we may want to implement ``split`` function.

The suggested methods are completely arbitrary and they may or may not be implemented.
The key concept is to keep data-related function encapsuled together in the dataset object,
so that one don't need to implement several separate script for fetching/visualization/statistics etc.

An elegant way of executing the dataset methods is via ``cxflow dataset <method-name> <config-file>``.
It constructs the dataset specified in the config file and invokes the proper method.

A typical pipeline contains the following commands.
We leave them without further comments as they are self-describing.

- ``cxflow dataset fetch config/my-data.yaml``
- ``cxflow dataset checksum config/my-data.yaml``
- ``cxflow dataset print_statistics config/my-data.yaml``
- ``cxflow dataset plot_histogram config/my-data.yaml``
- ``cxflow train config/my-data.yaml``
- ``cxflow predict config/my-data.yaml``

The Philosophy of Laziness
--------------------------

In our experience, the best practice for the dataset is to implement it as lazy as possible.
That is, constructor should not perform any time-consuming operation such as loading and decoding the data.
Instead, the data should be loaded and encoded in the first moment they are really necessary (e.g.,
in the ``train_stream`` method).

The main reason for laziness is that the dataset doesn't know for which purpose it was constructed.
It might be queried to provide the training data or only to print some simple checksums.
In the cases of extremely big datasets, it is useless and annoying to waste the time by loading the data
without their actual use.
