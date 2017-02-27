import abc
import logging
import typing
from collections import defaultdict
from os import path
from datetime import datetime

import tensorflow as tf


FuelIterator = typing.NewType('FuelIterator', typing.Iterable[typing.Mapping[str, typing.Any]])


class AbstractNet:
    TRAIN = 1
    VALID = 2
    TEST = 3

    def __init__(self, name: str, learning_rate: float, log_root: str='log', optimizer: str='adam',
                 device: str='/cpu:0', threads: int=4, restore_from=None, **kwargs):

        self.name = name
        self.learning_rate = learning_rate
        self.log_root = log_root
        self.device = device
        self.threads = threads

        # Save kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

        # Extended init
        self.extended_init(**kwargs)

        # Save mandatory parameters
        self.log_dir = path.join(self.log_root, '{}_{}'.format(self.name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

        # Define the optimizer
        self.optimizer = AbstractNet.build_optimizer(optimizer_name=optimizer)(learning_rate=self.learning_rate)

        # Set default attributes
        self.to_evaluate = []   # list of strings
        self.train_op = None

        with tf.device(self.device):
            self.create_net()
            assert hasattr(self, 'train_op'), '`Net::create_net` must define `train_op`'

            self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=self.threads,
                                                            intra_op_parallelism_threads=self.threads,
                                                            allow_soft_placement=True))

        self.saver = tf.train.Saver()

        if restore_from:
            logging.info('Restoring model from %s', restore_from)
            self.saver.restore(self.session, restore_from)
        else:
            logging.info('Creating a brand new model (no restoring)')
            try:  # TF 0.12.1+
                self.session.run(tf.global_variables_initializer())
                self.session.run(tf.local_variables_initializer())
            except AttributeError:  # TF 0.11.0-
                self.session.run(tf.initialize_all_variables())
                self.session.run(tf.initialize_local_variables())

        try:  # TF 0.12.1+
            self.summary_writer = tf.summary.FileWriter(logdir=self.log_dir,
                                                         graph=self.session.graph,
                                                                 flush_secs=10)
        except AttributeError:  # TF 0.11.0-
            self.summary_writer = tf.train.SummaryWriter(logdir=self.log_dir,
                                                         graph=self.session.graph,
                                                         flush_secs=10)

    def save_checkpoint(self, epoch_id: int):
        save_path = self.saver.save(self.session, path.join(self.log_dir, 'model_{}.ckpt'.format(epoch_id)))
        return save_path

    @staticmethod
    def build_optimizer(optimizer_name: str):
        optimizer_name = optimizer_name.lower()

        if optimizer_name == 'adam':
            return tf.train.AdamOptimizer
        elif optimizer_name == 'adadelta':
            return tf.train.AdadeltaOptimizer
        elif optimizer_name == 'adagrad':
            return tf.train.AdagradOptimizer
        elif optimizer_name == 'sgd':
            return tf.train.GradientDescentOptimizer
        elif optimizer_name == 'momentum':
            return tf.train.MomentumOptimizer
        elif optimizer_name == 'proximaladagrad':
            return tf.train.ProximalAdagradOptimizer
        elif optimizer_name == 'proximalsgd':
            return tf.train.ProximalGradientDescentOptimizer
        elif optimizer_name == 'rmsprop':
            return tf.train.RMSPropOptimizer
        else:
            raise ValueError('Unsupported optimizer')

    @abc.abstractmethod
    def create_net(self) -> None:
        pass

    def extended_init(self, **kwargs):
        pass

    def register_binary_confussion_matrix(self, predicted, actual):
        self.to_evaluate += ['tp', 'tn', 'fp', 'fn']

        predicted = tf.cast(predicted, dtype=tf.bool)
        is_label_one = tf.cast(actual, dtype=tf.bool)
        is_label_zero = tf.logical_not(is_label_one)

        self.tp = tf.reduce_mean(tf.cast(tf.logical_and(predicted, is_label_one), tf.float32))
        self.fp = tf.reduce_mean(tf.cast(tf.logical_and(predicted, is_label_zero), tf.float32))
        self.tn = tf.reduce_mean(tf.cast(tf.logical_and(predicted, is_label_zero), tf.float32))
        self.fn = tf.reduce_mean(tf.cast(tf.logical_and(predicted, is_label_one), tf.float32))

    def register_nary_confussion_matrix(self, n, predicted, gold):
        for i in range(n):
            for j in range(n):
                self.to_evaluate.append('cm_{}_{}'.format(i, j))

                is_predicted_i = tf.cast(tf.equal(predicted, i*tf.ones_like(predicted)), dtype=tf.bool)
                is_gold_j = tf.cast(tf.equal(gold, j*tf.ones_like(predicted)), dtype=tf.bool)

                setattr(self, 'cm_{}_{}'.format(i, j),
                        tf.reduce_mean(tf.cast(tf.logical_and(is_predicted_i, is_gold_j), tf.float32)))

    @staticmethod
    def _get_activation(activation_name: str) -> typing.Callable[[tf.Tensor], tf.Tensor]:
        if activation_name == 'ReLU':
            return tf.nn.relu
        if activation_name == 'Identity':
            return tf.identity

        raise NotImplementedError

    @staticmethod
    def mlp(x: tf.Tensor, dims: typing.Iterable[int], activations: typing.Iterable[str],
            dropout: typing.Iterable[float]=None, std=0.001):

        logging.debug('MLP: input: %s', x.get_shape())
        x = tf.contrib.layers.flatten(x)

        for i, (dim, activation_name) in enumerate(zip(dims, activations)):
            W = tf.get_variable(name='W{}'.format(i),
                                shape=[x.get_shape()[1], dim],
                                initializer=tf.random_normal_initializer(mean=0, stddev=std))
            logging.debug('MLP: W_%d: %s', i, W.get_shape())
            h = tf.get_variable(name='h{}'.format(i),
                                shape=[dim],
                                initializer=tf.constant_initializer(0.0))
            logging.debug('MLP: h_%d: %s', i, h.get_shape())

            activation = AbstractNet._get_activation(activation_name)
            x = activation(tf.matmul(x, W) + h)

            if dropout:
                if float(dropout[i]) > 0:
                    x = tf.nn.dropout(x, keep_prob=1-float(dropout[i]))

            logging.debug('MLP: after layer %d: %s', i, x.get_shape())

        return x
