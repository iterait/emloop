from ..datasets.abstract_dataset import AbstractDataset

import tensorflow as tf

import abc
import logging
import typing
from os import path


class AbstractNet:

    def __init__(self, dataset: AbstractDataset, log_dir: str, name: str, io: dict, learning_rate: float,
                 optimizer: str='adam', device: str='/cpu:0', threads: int=4, restore_from=None,
                 ignore_extra_sources: bool=True, skip_incomplete_batches: bool=False, **kwargs):
        """
        Abstract net which should be an ancestor to all nets.

        At first, all kwargs are saved as attributes. Then, overloaded `extended_init` is called. Then, the model is
        created (or restored) via `create_net`.

        :param dataset: dataset to be used
        :param log_dir: training directory (for logging purposes)
        :param name: name of the training
        :param io: dict containing `in` and `out` which are mapped to list of strings representing placeholders and
                   the provided outputs, respectively.
        :param learning_rate: learning rate
        :param optimizer: @see _build_optimizer for options
        :param device: {/cpu:0, /gpu:0}
        :param threads: number of threads to be used
        :param restore_from: name of checkpoint to be restored from or None if new model should be created
        :param ignore_extra_sources: if set to True (default), warning will be raised when the stream batch contains
                                     a superset of requires sources. If set to False, an error will be raised.
        :param skip_incomplete_batches: if set to True, the incomplete (in terms of batch_size) batches will not be
                                        processed. If set to False (default), the batches will be processed normally.
                                        This option is useful for TensorFlow constructs that requires fixed batch size,
                                        which is rare nowadays (but still present in some advanced RNNs).
        :param kwargs: will be saved as attributes
        """

        self.dataset = dataset
        self.name = name
        self.io = io
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.device = device
        self.threads = threads
        self.ignore_extra_sources = ignore_extra_sources
        self.skip_incomplete_batches = skip_incomplete_batches

        assert 'in' in self.io
        assert 'out' in self.io
        assert self.learning_rate > 0
        assert self.threads > 0

        # Save kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

        if not restore_from:
            # Extended init
            self.extended_init(dataset=dataset, log_dir=log_dir, name=name, learning_rate=learning_rate,
                               optimizer=optimizer, device=device, threads=threads, restore_from=restore_from, **kwargs)

        # Define the optimizer
        self.optimizer = AbstractNet._build_optimizer(optimizer_name=optimizer)(learning_rate=self.learning_rate)

        # Set default attributes
        self.train_op = None

        if restore_from:
            with tf.device(self.device):
                logging.debug('Creating empty session')
                self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=self.threads,
                                                                intra_op_parallelism_threads=self.threads,
                                                                allow_soft_placement=True))
                if not restore_from.endswith('.ckpt'):
                    logging.warning('`restore_from` does not end with `.ckpt` suffix. Automatically adding it. Consider'
                                    'explicitely adding the `.ckpt` suffix to the config.')
                    restore_from += '.ckpt'

                logging.debug('Loading meta graph')
                self.saver = tf.train.import_meta_graph(restore_from + '.meta')

                logging.debug('Restoring model')
                self.saver.restore(self.session, restore_from)

                try:
                    logging.debug('Loading `train_op`')
                    self.train_op = tf.get_default_graph().get_operation_by_name('train_op')
                except Exception as e:
                    logging.error('Cannot load `train_op`. Make sure the dumped model has a training operation'
                                  'named `train_op`.')
                    raise e

                logging.debug('Loading net IO')
                for tensor_name in set(self.io['in'] + self.io['out']):
                    try:
                        setattr(self, tensor_name, tf.get_default_graph().get_tensor_by_name(tensor_name+':0'))
                    except Exception as e:
                        logging.error('Cannot find tensor "%s" in model loaded from "%s".', tensor_name, restore_from)
                        raise e

        else:
            # Create model placed to device
            with tf.device(self.device):
                logging.debug('Creating net')
                self.create_net()

                assert hasattr(self, 'train_op'), '`Net::create_net` must define `train_op`'
                assert self.train_op.name == 'train_op', '`train_op` must be named as "train_op"'

                logging.debug('Creating session')
                self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=self.threads,
                                                                intra_op_parallelism_threads=self.threads,
                                                                allow_soft_placement=True))

            for tensor_name in set(self.io['in'] + self.io['out']):
                assert hasattr(self, tensor_name), 'IO defined tensor `{}` but it was not defined ' \
                                                   'in the net.'.format(tensor_name)
                tensor = getattr(self, tensor_name)
                assert tensor.name == tensor_name+':0', 'Tensor stored in variable `{}` has different name `{}`. ' \
                                                        'This will prevent correct restoring of the ' \
                                                        'model.'.format(tensor_name, tensor.name)

            logging.debug('Creating Saver')
            self.saver = tf.train.Saver()

            logging.debug('Init. variables')
            try:  # TF 0.12.1+
                self.session.run(tf.global_variables_initializer())
                self.session.run(tf.local_variables_initializer())
            except AttributeError:  # TF 0.11.0-
                self.session.run(tf.initialize_all_variables())
                self.session.run(tf.initialize_local_variables())

        logging.debug('Creating TensorBoard writer')
        try:  # TF 0.12.1+
            self.summary_writer = tf.summary.FileWriter(logdir=self.log_dir,
                                                        graph=self.session.graph,
                                                        flush_secs=10)
        except AttributeError:  # TF 0.11.0-
            self.summary_writer = tf.train.SummaryWriter(logdir=self.log_dir,
                                                         graph=self.session.graph,
                                                         flush_secs=10)

    def save_checkpoint(self, name: str) -> str:
        """Save TensorFlow checkpoint with named suffix."""
        save_path = self.saver.save(self.session, path.join(self.log_dir, 'model_{}.ckpt'.format(name)))
        return save_path

    @staticmethod
    def _build_optimizer(optimizer_name: str):  # TODO: return type (be carefull TF0.10-1.0
        """
        Create a TensorFlow optimizer. Supported arguments are {adam, adadelta, adagrad, sgd, momentum,
        proximaladagrad, proximalsgd, rmsprop}"""
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
        """
        This method describes the architecture of the networks. All placeholders, train_op and logged statistics must
        be saved as attributes.
        """
        pass

    def extended_init(self, **kwargs):
        """This method is invoked after all kwargs are saved but before the model is created."""
        pass

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
