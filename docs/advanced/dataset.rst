Dataset
*******

Dataset is an object representing the data used in cxflow.
In general, the dataset is an instance of arbitrary class which is able handle the string-encoded YAML configuration
in its constructor.
Note that this configuration is parsed from the config file section `dataset`.

BaseDataset
-----------

Usage of `cxflow.BaseDataset` is recommended as it parses the YAML string automatically.
Instead of the constructor, the developer is expected to implement `_init_with_kwargs`.
This method is supposed to accept a subset of parameters that are configured in the config file.

The following example demonstrates this behavior.
Firstly, we define the dataset section in the config file.
The constructed dataset will be an instance of `MyDataset` located in `datasets.my_dataset` module.
The configuration passed to the constructor will contain information of batch size and a dict of augmentation settings.

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

Secondly, we use `cxflow.BaseDataset` for automatic parsing of the configuration by implementing `_init_with_kwargs`
method.
Note that even though the example configuration specifies four attributes, our dataset is interested in only two of them
(`batch_size` and `augment`), leaving `width` and `height` in the `**kwargs`.

.. code-block:: python

     from cxflow import BaseDataset

     class MyDataset(BaseDataset):
         def _init_with_kwargs(batch_size: int, augment: dict, **kwargs):
           # ...

Data Processing
---------------

Firstly, let us define **data source**.
We refer to the data source as to a unit (or type) of which the (training) example consists.
Let's take a look at artificial image-classification task in which we are supposed to classify the input image
into various animal classes (dog, cat, rabbit, ...).
In this setting, the sources could be e.g. `image` and `animal`.
The former is in form of a tensor with shape of 800x600x3 (i.e. regular RGB 800x600 image).
The latter is a string describing the animal depicted at the image.

Secondly, let us define **batch of data**.
We refer to a batch as to a collection of training examples.
Note that the examples are represented source-wise, hence it is a Python `dict` mapping the source name to a collection
of corresponding source of the examples.
The example of a single batch follows (batch size is set ot four):

.. code-block:: python

    batch = {
      'image': [img1, img2, img3, img4],
      'animal': ['cat', 'cat', 'dog', 'rabbit']
    }

Finally, let us define the **data stream**.
Assuming we have a set of data we want to use for a model training.
The stream is a an iterable (preferably a generator) which yields batches when iterating through it.

The streams are defined in parameter-less methods with a name convention `<name>_stream`.
An example of a simple training stream definition is as follows.
The datasets contains `train_stream` method which, when iterated through, yields batches of training data via
function `load_training_batch`.
For simplicity, we assume this function is already implemented and returns the training batches.

.. code-block:: python

    def train_stream(self):
        for i in range(10):
            yield load_training_batch(num=i)

The training stream is used only when the training is invoked (either by `cxflow train` or `cxflow resume` commands).
Analogously, additional methods such as `valid_stream` and `test_stream` can be easily implemented.
These streams *are not* used for training.
Instead, the developer might register them in the configuration file as the *extra streams* which will be used only for
evaluation.

.. code-block:: yaml

    main_loop:
      extra_streams: [valid, test]

During prediction (`cxflow predict`), `predict_stream` method is employed in order to provide data to be inferred.
Extra streams might be inferred as well when registred as described above (including training stream).

Additional Methods
------------------

The dataset is supposed to represent a full data wrapper including various additional methods.
For example, the dataset can contain `fetch` method which checks whether the dataset has all the data it requires.
If not, it downloads them from the internet/databse/drive.

Additional useful method could be `statistics` which prints various statistics of the provided data, plot some figures
etc.
Sometimes, we need to split the whole dataset into training, validation and testing sets.
Implementing `split` function is straightforward.

The suggested methods are completely arbitrary and the develop might implement additional ones or none at all.
The key concept is to keep data-related function encapsuled together in the dataset object, so that one don't need to
implement several script for fetching/visualization/statistics etc. anymore.

An elegant way of executing the dataset methods is via `cxflow dataset <method-name> <config-file>`.
It constructs the dataset specified in the config file and invokes the proper method.

A typical pipeline contain the following commands.
We leave them without further comments as they are self-describing.

- `cxflow dataset fetch congif/my-data.yaml`
- `cxflow dataset checksum congif/my-data.yaml`
- `cxflow dataset print_statistics congif/my-data.yaml`
- `cxflow dataset plot_histogram congif/my-data.yaml`
- `cxflow train congif/my-data.yaml`
- `cxflow predict congif/my-data.yaml`

The Philosophy of Laziness
--------------------------

Best practise of dataset is implement them as lazy as possible.
That is, constructor can be executed without time-consuming operation such as loading and decoding the data.

The main reason for the laziness is that the dataset doesn't know for which reasons it was constructed.
It might provide training data as well as only print some checksums etc.
In the cases of extremely big datasets, it is useless to waste the time by loading the data without their actual use.
