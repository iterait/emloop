"""
Test module for base tensorflow nets (cxflow.nets.tf_net).
"""
import os
from os import path

import tensorflow as tf
import numpy as np

from cxflow.nets.tf_net import BaseTFNet, BaseTFNetRestore, create_optimizer
from cxflow.tests.test_core import CXTestCaseWithDirAndNet


class DummyNet(BaseTFNet):
    """Dummy tf net with empty graph."""

    def _create_net(self, **kwargs):
        """Create dummy tf net."""

        # defining dummy variable as otherwise we would not be able to create the net saver
        tf.Variable(name='dummy', initial_value=[1])

        # defining a dummy train_op as otherwise we would not be able to create the net
        tf.no_op(name='train_op')


class TrainOpNet(BaseTFNet):
    """Dummy tf net with train op saved in self."""

    def _create_net(self, **kwargs):
        """Create dummy tf net."""

        # defining dummy variable as otherwise we would not be able to create the net saver
        tf.Variable(name='dummy', initial_value=[1])

        # defining a dummy train_op as otherwise we would not be able to create the net
        self.defined_train_op = tf.no_op(name='train_op')


class NoTrainOpNet(BaseTFNet):
    """Dummy tf net without train op."""

    def _create_net(self, **kwargs):
        """Create dummy tf net."""

        # defining dummy variable as otherwise we would not be able to create the net saver
        tf.Variable(name='dummy', initial_value=[1])

        # defining an op that is not named `train_op`
        tf.no_op(name='not_a_train_op')


class SimpleNet(BaseTFNet):
    """Simple net with input and output tensors."""

    def _create_net(self, **kwargs):
        """Create simple tf net."""

        self.input1 = tf.placeholder(tf.int32, shape=[None, 10], name='input')
        self.input2 = tf.placeholder(tf.int32, shape=[None, 10], name='second_input')

        self.const = tf.Variable([2]*10, name='const')

        self.output = tf.multiply(self.input1, self.const, name='output')

        self.sum = tf.add(self.input1, self.input2, name='sum')

        # defining a dummy train_op as otherwise we would not be able to create the net
        self.defined_train_op = tf.no_op(name='train_op')

        self.session.run(tf.global_variables_initializer())


class TrainableNet(BaseTFNet):
    """Trainable tf net."""

    def _create_net(self, **kwargs):
        """Create simple trainable tf net."""

        self.input = tf.placeholder(tf.float32, shape=[None, 10], name='input')
        self.target = tf.placeholder(tf.float32, shape=[None, 10], name='target')

        self.var = tf.Variable([2] * 10, name='var', dtype=tf.float32)

        self.output = tf.multiply(self.input, self.var, name='output')

        loss = tf.reduce_mean(tf.squared_difference(self.target, self.output))

        # defining a dummy train_op as otherwise we would not be able to create the net
        create_optimizer({'module': 'tensorflow.python.training.adam',
                          'class': 'AdamOptimizer', 'learning_rate': 0.1}).minimize(loss, name='train_op')

        self.session.run(tf.global_variables_initializer())


class BasetTFNetTest(CXTestCaseWithDirAndNet):
    """
    Test case for BaseTFNet.

    Note: do not forget to reset the default graph after every net creation!
    """

    def test_init_asserts(self):
        """Test if the init arguments are correctly asserted."""

        good_io = {'in': [], 'out': ['dummy']}
        DummyNet(dataset=None, log_dir='', io=good_io)
        tf.reset_default_graph()

        # test assertion on missing in/out
        missing_in = {'out': ['dummy']}
        missing_out = {'in': []}
        empty_out = {'in': [], 'out': []}
        self.assertRaises(AssertionError, DummyNet, dataset=None, log_dir='', io=missing_in)
        tf.reset_default_graph()
        self.assertRaises(AssertionError, DummyNet, dataset=None, log_dir='', io=missing_out)
        tf.reset_default_graph()
        self.assertRaises(AssertionError, DummyNet, dataset=None, log_dir='', io=empty_out)
        tf.reset_default_graph()

        # test assertion on negative thread count
        DummyNet(dataset=None, log_dir='', io=good_io, threads=2)
        self.assertRaises(AssertionError, DummyNet, dataset=None, log_dir='', io=good_io, threads=-2)
        tf.reset_default_graph()

    def test_finding_train_op(self):
        """Test finding train op in graph."""

        good_io = {'in': [], 'out': ['dummy']}

        # test whether train_op is found correctly
        trainop_net = TrainOpNet(dataset=None, log_dir='', io=good_io)
        self.assertEqual(trainop_net.defined_train_op, trainop_net.train_op)
        tf.reset_default_graph()

        # test whether an error is raised when no train_op is defined
        self.assertRaises(ValueError, NoTrainOpNet, dataset=None, log_dir='', io=good_io)
        tf.reset_default_graph()

    def test_io_mapping(self):
        """Test if net.io is translated to output/input names."""

        good_io = {'in': ['input', 'second_input'], 'out': ['output', 'sum']}
        net = SimpleNet(dataset=None, log_dir='', io=good_io)
        self.assertListEqual(net.input_names, good_io['in'])
        self.assertListEqual(net.output_names, good_io['out'])
        tf.reset_default_graph()

        # test if an error is raised when certain input/output tensor is not found
        missing_input_tensor = {'in': ['input', 'second_input', 'third_input'], 'out': ['output', 'sum']}
        missing_output_tensor = {'in': ['input', 'second_input'], 'out': ['output', 'sum', 'sub']}
        self.assertRaises(ValueError, SimpleNet, dataset=None, log_dir='', io=missing_input_tensor)
        tf.reset_default_graph()
        self.assertRaises(ValueError, SimpleNet, dataset=None, log_dir='', io=missing_output_tensor)
        tf.reset_default_graph()

    def test_get_tensor_by_name(self):
        """Test if _get_tensor_by_name works properly."""

        good_io = {'in': ['input', 'second_input'], 'out': ['output', 'sum']}
        net = SimpleNet(dataset=None, log_dir='', io=good_io)
        self.assertEqual(net.get_tensor_by_name('sum'), net.sum)
        self.assertRaises(KeyError, net.get_tensor_by_name, name='not_in_graph')
        tf.reset_default_graph()

    def test_run(self):
        """Test tf net run."""

        good_io = {'in': ['input', 'second_input'], 'out': ['output', 'sum']}
        net = SimpleNet(dataset=None, log_dir='', io=good_io)
        valid_batch = {'input': [[1]*10], 'second_input': [[2]*10]}

        # test if outputs are correctly returned
        results = net.run(batch=valid_batch, train=False)
        for output_name in good_io['out']:
            self.assertTrue(output_name in results)
        self.assertTrue(np.allclose(results['output'], [2]*10))
        self.assertTrue(np.allclose(results['sum'], [3]*10))
        tf.reset_default_graph()

        # test variables update if and only if train=True
        trainable_io = {'in': ['input', 'target'], 'out': ['output']}
        trainable_net = TrainableNet(dataset=None, log_dir='', io=trainable_io)
        batch = {'input': [[1]*10], 'target': [[0]*10]}

        # single run with train=False
        orig_value = trainable_net.var.eval(session=trainable_net.session)
        trainable_net.run(batch, train=False)
        after_value = trainable_net.var.eval(session=trainable_net.session)
        self.assertTrue(np.allclose(orig_value, after_value))

        # multiple runs with train=False
        for _ in range(100):
            trainable_net.run(batch, train=False)
        after_value = trainable_net.var.eval(session=trainable_net.session)
        self.assertTrue(np.allclose(orig_value, after_value))

        # single run with train=True
        trainable_net.run(batch, train=True)
        after_value = trainable_net.var.eval(session=trainable_net.session)
        self.assertFalse(np.allclose(orig_value, after_value))

        # multiple runs with train=True
        trainable_net.run(batch, train=True)
        for _ in range(1000):
            trainable_net.run(batch, train=True)
        after_value = trainable_net.var.eval(session=trainable_net.session)
        self.assertTrue(np.allclose([0]*10, after_value))


class BasetTFNetRestoreTest(CXTestCaseWithDirAndNet):
    """
    Test case for BaseTFNetRestore.

    Additionally, save method of BaseTFNet is tested.

    Note: do not forget to reset the default graph after every net creation!
    """

    def test_restore(self):
        """Test net saving and restoring."""

        # test net saving
        trainable_io = {'in': ['input', 'target'], 'out': ['output']}
        trainable_net = TrainableNet(dataset=None, log_dir=self.tmpdir, io=trainable_io)
        batch = {'input': [[1] * 10], 'target': [[0] * 10]}
        for _ in range(1000):
            trainable_net.run(batch, train=True)
        saved_var_value = trainable_net.var.eval(session=trainable_net.session)
        checkpoint_path = trainable_net.save('')

        index_path = checkpoint_path + '.index'
        meta_path = checkpoint_path + '.meta'
        checkpoint_file_path = path.join(path.dirname(checkpoint_path), 'checkpoint')

        self.assertTrue(path.exists(index_path))
        self.assertTrue(path.exists(meta_path))
        self.assertTrue(path.exists(checkpoint_file_path))
        tf.reset_default_graph()

        # test restoring
        restored_net = BaseTFNetRestore(dataset=None, log_dir='', io=trainable_io, restore_from=checkpoint_path)

        var = restored_net.graph.get_tensor_by_name('var:0')
        var_value = var.eval(session=restored_net.session)
        self.assertTrue(np.allclose(saved_var_value, var_value))


class TFBaseNetSaverTest(CXTestCaseWithDirAndNet):
    """
    Test case for correct usage of tensorflow saver in BaseTFNet.
    """

    def test_keep_checkpoints(self):
        """
        Test if the checkpoints are kept.

        This is regression test for issue #71 (tensorflow saver is keeping only the last 5 checkpoints).
        """
        dummy_net = SimpleNet(dataset=None, log_dir=self.tmpdir, io={'in': [], 'out': ['output']})

        checkpoints = []
        for i in range(20):
            checkpoints.append(dummy_net.save(str(i)))

        for checkpoint in checkpoints:
            self.assertTrue(path.exists(checkpoint+'.index'))
            self.assertTrue(path.exists(checkpoint+'.meta'))
            data_prefix = path.basename(checkpoint)+'.data'
            data_files = [file for file in os.listdir(path.dirname(checkpoint)) if file.startswith(data_prefix)]
            self.assertGreater(len(data_files), 0)
