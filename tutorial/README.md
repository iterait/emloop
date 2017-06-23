# cxflow Tutorial
**This tutorial is a draft and requires a review**

This document provides the description of **cxflow** usage.

## Fundamentals
**cxflow** is designed as a modular system.
It consists of four fundamental components:

- Dataset
- Net
- Configuration
- Hook

These components will be discussed in the following text.
The components except Configuration are independent on each other and might be reused across various applications.

We demonstrate the whole tutorial on a very simple task of *majority*. Given a vector of five bits, which bit is in majority?

**Example:**
```
00101 -> 0
00000 -> 0
10101 -> 1
11100 -> 1
```

## Project structure
While the project structure is completely arbitrary, it is strongly suggested to use the following directory layout.

```
TutorialExample/
    config/
    datasets/
    hooks/
    nets/
```

## Dataset
The very first step in any machine learning taks is to load and process the data.
This is handled by tha Dataset component.
Each Dataset is expected to extend `cxflow.datasets.AbstractDataset` which means

- it might be constructed by passing a YAML configuration (in form of string)
- it contains multiple `create_train_stream` methods
- it contains multiple `create_<name>_stream` methods
- optionally it contains `split` method

Let's (for now) ignore the dataset construction and focus on the desired functionality.
The datasets are basically wrappers for so called streams which provide the epoch data (i.e. one data pass).
Each epoch consists of multiple steps, each taking only part of the training data - so called batches which are dictionaries in form on `{source: values}`.

**Example:**

In our Majority example we train a simple relation.
Each training example is a tuple of `(11-bits, majority-bit)`.
The whole training dataset contains ~400 examples and the test dataset ~100 examples.
We set the batch size to 4.
One training epoch will then contain 100 training steps, each processing a single batch.

One such batch might look like this:

```python
{
    'x': [0, 1, 1, 1],
    'y': [
                [0,0,0,0,0,0,1,0,0,1,0],
                [0,1,1,0,1,0,0,1,1,1,1],
                [0,0,1,1,1,0,1,0,1,1,0],
                [1,1,1,0,1,0,1,0,0,1,1]
             ],
}
```

Here, the sources are named `x` and `y`.
Note that each source provides a list of its instances.

End-of-example




Similarly, we define another streams such as test.
Note that the streams might provide different data among epochs but it is not advised to do so except the training stream.

Working example of such dataset could look like this:

TODO: code


## Net
After the data are loaded, processed and ready to be used, the model itself must be defined.
The net is expected to extend `cxflow.AbstractNet` which means to define what input it takes (e.g. `image` and `label`),
what outputs it provides (`predicted_label` and `accuracy`).
Next, method for saving the data must be defined.
Finally, `run` method must be implemented which handles the processing of a single batch.

While `AbstractNet` is very general, from now on we will focus on TensorFlow nets only.
For that purpose, [cxflow-tensorflow](https://gitlab.com/Cognexa/cxflow-tensorflow) must be installed, which 
provides `cxflow_tf.BaseTFNet`.
In this tutorial, we will extend this one.

When using `cxflow_tf.BaseTFNet`, everythings happen silently in the background and the only remaining part is to  specify the itself.
This is done in `_create_net` method which must be implemented.
Note that `_create_net` accepts arbitrary arguments - in our case used optimizer and number of hidden units.
We ignore the origin of these parameters for a while and address it in the Configuration section.
For now, let's simply assume we have them already set.

First, let's define the standard TensorFlow placeholders, i.e. the inputs of the network.
```python
x = tf.placeholder(dtype=tf.float32, shape=[None, 11], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')
```

Then let's construct simple MLP via new Keras API
```python
hidden_activations = K.layers.Dense(hidden)(x)
y_hat = K.layers.Dense(1)(hidden_activations)[:, 0]
```

Then compute squared error and its mean (named as `loss`):
```python
sq_err = tf.pow(y - y_hat, 2, name='sq_err')
loss = tf.reduce_mean(sq_err)
```

Create optimizer via cxflow API and name it `train_op`. This ensures that the network updates its parameters in a way we want.
```python
create_optimizer(optimizer).minimize(loss, name='train_op')
```
Finally, create variables for network predictions and accuracy:
```python
predictions = tf.greater_equal(y_hat, 0.5, name='predictions')
tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(y, tf.bool)), tf.float32, name='accuracy'))
```
And initialize all variables:
```python
self._session.run(tf.global_variables_initializer())
self._session.run(tf.local_variables_initializer())
```

Note that naming the variables correctly and consistently is mandatory - we will se the example usage of the names in the next section.

##Configuration
Now we have our own dataset and a network.
How to put it together?
This is managed by a config. file, e.g. `configs/simple.yaml`.

Each config file consists of four fundamental sections

- dataset
- net
- main_loop
- hooks

Each of them manages an automatical creation of a underlying component.

In order to create our dataset, we simply add `dataset` section with proper module and class of it:
```yaml
dataset:
  dataset_module: datasets.majority_dataset
  dataset_class: MajorityDataset
```

Simiarly, the network can be easily configured by 
```yaml
  net_module: nets.majority_net
  net_class: MajorityNet
```

The remaining question is solve the problem of passing the parameters to the network `_create_net` method.
This is done by passing all remaining values from `net` config section to `_create_net`.
Similarly, the whole configuration section of the dataset is passed as a string to the dataset constructor.
This means that the dataset can be implemented in any programming language that can be called from python.

**Example:**
We want to set number of hidden neurons to 100.
```yaml
  hidden: 100
```

Similarly, to configure the optimizer:
```yaml
  optimizer:
    module: tensorflow.python.training.adam
    class: AdamOptimizer
    learning_rate: 0.001
```

end-of-example

Finally, we have to specify inputs and outputs of the network.
This is done by `io` section:
```yaml
  io:
    in:
      - x
      - y
    out:
      - predictions
      - sq_err
      - accuracy
```
In this setting, we instruct the network to expect inputs `x` and `y`.
**cxflow** will make sure the proper dataset provides these sources and links them to the proper TensorFlow placeholders.
Similarly, the output variables are the ones that will be evaluated by the computational graph.
This is the part where the correct naming plays important role - `io` in the config is precisely mapped to the
variable names in the graph.

Next config section deals with the training main loop itself.
Here, additional streams might be listed in order to be evaluated.
```yaml
main_loop:
  extra_streams: [test]
```
The last section of the configuration describes hooks.
Please see a separate tutorial on hooks in order to implement your own one.

Briefly - hooks are tiny classes that are invoked on some special events (e.g. after each epoch, after training etc.).
They are used for terminating the training, for reporting progress to the STDOUT or saving results to the database.

We start with the most complicated hook - `StatsHook`.
This hook is responsible for various statistics about `io` variables.
As a demonstration we show how to add a `StatsHook` that computes mean batch accuracy after each epoch and mean squared error
including the standard error.
```yaml
hooks:
  - class: StatsHook
    variables:
      sq_err: [mean, std]
      accuracy: [mean]
```

In addition, we simply add another hooks that make some pretty logging (`StatsHook`).
Moreover, we want a nice way of terminating the training manualy (`SigintHook`).
```yaml
  - class: LoggingHook
  - class: SigintHook
```

Finally, let's add a hook that terminates the training after 10 epochs.
```yaml
  - class: EpochStopperHook
    epoch_limit: 10
```

## Hook
See te separate tutorial

## Training Itself
Finally, we have dataset, net and configuration implemented.
Now we can finally run it.
```bash
$ cxflow train -v configs/simple.yaml
```

If some of the parameters in the configuration need to be overwritten from to command-line, it can be easily done:

```bash
$ cxflow train -v configs/simple.yaml net.optimizer.learning_rate:float=0.1
```
## cxflow Advantages
Components such as hooks, datasets and nets might be easily reused in other applications.
As an example, let's assume our dataset is fine and we want to experiment with different networks.
Simply create new networks with unique name and run it.

The configuration files are plain text filex which might be easily versioned via e.g. git.
In addition, everything is really flexible.
Even TensorFlow isn't mandatory (but it is pretty easy to demonstrate cxflow on it).

## Conclusion
We demonstrated a solution to a simple task.
More real-world problem - MNIST - lies in its [own repository](https://gitlab.com/Cognexa/cxMNIST).

Next steps

- Fuel dataset backend
- Hooks tutorial
- cxtream (a C++ high-perf dataset backend)
