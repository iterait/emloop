"""
Module with cxflow trainable nets defined in tensorflow.

Provides BaseTFNet which manages net config, api and unifies tf graph <=> cxflow touch points.

Furthermore, this module exposes BaseTFNetRestore class
which is able to restore arbitrary cxflow nets from tf checkpoint.
"""
import logging
from os import path
from abc import abstractmethod, ABCMeta
from typing import Dict, Callable, List, Mapping, Any

import tensorflow as tf

from ..datasets.abstract_dataset import AbstractDataset
from ..third_party.tensorflow.freeze_graph import freeze_graph
from ..utils.reflection import create_object_from_config, get_class_module
from .abstract_net import AbstractNet

TF_OPTIMIZERS_MODULE = 'tensorflow.python.training'


def create_optimizer(optimizer_config: Dict[str, Any]):
    """
    Create tf optimizer according to the given config.

    When `module` entry is not present in the optimizer_config,
    the function attempts to find it under the TF_OPTIMIZER_MODULE.
    :param optimizer_config: dict with at least `class` and `learning_rate` entries
    :return: optimizer
    """
    assert 'learning_rate' in optimizer_config
    assert 'class' in optimizer_config
    kwargs = optimizer_config.copy()
    learning_rate = kwargs.pop('learning_rate')
    if 'module' not in kwargs:
        optimizer_module = get_class_module(TF_OPTIMIZERS_MODULE, optimizer_config['class'])
        if optimizer_module is not None:
            optimizer_config['module'] = optimizer_module
        else:
            raise ValueError('Can\'t find the optimizer module for class `{}` under `{}`. Please specify it explicitly.'
                             .format(optimizer_config['class'], TF_OPTIMIZERS_MODULE))
    else:
        kwargs.pop('module')
    kwargs.pop('class')
    return create_object_from_config(optimizer_config, args=(learning_rate,), kwargs=kwargs)


def create_activation(activation_name: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Create tf activation with the given name.
    :param activation_name: one of {Relu, Identity, Softmax}
    :return: activation
    """
    if activation_name == 'ReLU':
        return tf.nn.relu
    if activation_name == 'Identity':
        return tf.identity

    raise NotImplementedError


class BaseTFNet(AbstractNet, metaclass=ABCMeta):   # pylint: disable=too-many-instance-attributes
    """
    Base tensorflow network enforcing uniform net API which is trainable in cxflow main loop.

    All tf nets should be derived from this class and override _create_net method.
    """

    def __init__(self,   # pylint: disable=too-many-arguments
                 dataset: AbstractDataset, log_dir: str, io: dict, device: str='/cpu:0', threads: int=4, **kwargs):
        """
        Create new cxflow trainable tf net.

        :param dataset: dataset to be trained with
        :param log_dir: path to the logging directory (wherein models should be saved)
        :param io: net `in`put and `out`put names; `out`put names cannot be empty
        :param device: tf device to be trained on
        :param threads: number of threads to be used by tf
        :param kwargs: additional kwargs which are passed to the _create_net method
        """
        assert 'in' in io
        assert 'out' in io
        assert io['out']
        assert threads > 0

        self._dataset = dataset
        self._log_dir = log_dir
        self._train_op = None
        self._graph = None
        self._input_names = io['in']
        self._output_names = io['out']
        self._tensors = {}

        with tf.device(device):
            logging.debug('Creating session')
            self._session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                             intra_op_parallelism_threads=threads,
                                                             allow_soft_placement=True))

            logging.debug('Creating net')
            self._create_net(**kwargs)

            logging.debug('Finding train_op in the created graph')
            self._graph = tf.get_default_graph()
            try:
                self._train_op = self._graph.get_operation_by_name('train_op')
            except (KeyError, ValueError, TypeError) as ex:
                raise ValueError('Cannot find train op in graph. Train op must be named `train_op`.') from ex

            logging.debug('Finding io tensors in the created graph')
            for tensor_name in set(self._input_names + self._output_names):
                full_name = tensor_name + ':0'
                try:
                    tensor = self._graph.get_tensor_by_name(full_name)
                except (KeyError, ValueError, TypeError) as ex:
                    raise ValueError('Tensor `{}` defined as input/output was not found. It has to be named `{}`.'
                                     .format(tensor_name, full_name)) from ex

                if tensor_name not in self._tensors:
                    self._tensors[tensor_name] = tensor

        logging.debug('Creating Saver')

    @property
    def input_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        """List of tf tensor names listed as net inputs."""
        return self._input_names

    @property
    def output_names(self) -> List[str]:   # pylint: disable=invalid-sequence-index
        """List of tf tensor names listed as net outputs."""
        return self._output_names

    @property
    def graph(self) -> tf.Graph:
        """Tf graph object."""
        return self._graph

    @property
    def session(self) -> tf.Session:
        """Tf session object."""
        return self._session

    @property
    def train_op(self) -> tf.Operation:
        """Net train op."""
        return self._train_op

    def get_tensor_by_name(self, name) -> tf.Tensor:
        """
        Get the tf tensor with the given name.

        Only tensor previously defined as net inputs/outputs in net.io can be accessed.
        :param name: tensor name
        :return: tf tensor
        """
        if name in self._tensors:
            return self._tensors[name]
        else:
            raise KeyError('Tensor named `{}` is not within accessible tensors.'.format(name))

    def run(self, batch: Mapping[str, object], train: bool) -> Mapping[str, object]:
        """
        Feed-forward the net with the given batch as feed_dict.
        Fetch and return all the net outputs as a dict.
        :param batch: batch dict source_name->values
        :param train: flag whether parameters update (train_op) should be included in fetches
        :return: outputs dict
        """
        # setup the feed dict
        feed_dict = {}
        for placeholder_name in self._input_names:
            feed_dict[self.get_tensor_by_name(placeholder_name)] = batch[placeholder_name]

        # setup fetches
        fetches = [self._train_op] if train else []
        for output_name in self._output_names:
            fetches.append(self.get_tensor_by_name(output_name))

        # run the computational graph for one batch
        batch_res = self._session.run(fetches=fetches, feed_dict=feed_dict)

        if train:
            batch_res = batch_res[1:]

        return dict(zip(self._output_names, batch_res))

    def save(self, name_suffix: str) -> str:
        """
        Save current tensorflow graph to a checkpoint named with the given name suffix.

        The checkpoint will be locaced in self.log_dir directory.
        :param name_suffix: saved checkpoint name suffix
        :return: path to the saved checkpoint
        """
        graph_path = path.join(self._log_dir, 'model_{}.graph'.format(name_suffix))
        checkpoint_path = path.join(self._log_dir, 'model_{}.ckpt'.format(name_suffix))
        frozen_graph_path = path.join(self._log_dir, 'model_{}.pb'.format(name_suffix))

        tf.train.write_graph(self._session.graph_def, '', graph_path, as_text=False)

        tf.train.Saver().save(self._session, checkpoint_path)

        with tf.Graph().as_default():
            freeze_graph(input_graph=graph_path,
                         input_checkpoint=checkpoint_path,
                         output_node_names=self._output_names,
                         output_graph=frozen_graph_path)

        return checkpoint_path

    @abstractmethod
    def _create_net(self, **kwargs) -> None:
        """
        Create network according to the given config.

        -------------------------------------------------------
        cxflow framework requires the following
        -------------------------------------------------------
        1. define training op named as 'train_op'
        2. input/output tensors have to be named according to net.io config
        3. initialize/restore variables through self._session
        -------------------------------------------------------

        :param kwargs: net configuration
        """
        pass


class BaseTFNetRestore(BaseTFNet):
    """
    Generic tf net restore class used when no custom restore class is provided.
    """

    def _create_net(self, restore_from: str, **kwargs) -> None:
        """
        Restore tf net from the given checkpoint.
        :param restore_from: path to the checkpoint
        :param kwargs: additional **kwargs are ignored
        """
        logging.debug('Loading meta graph')
        saver = tf.train.import_meta_graph(restore_from + '.meta')
        logging.debug('Restoring model')
        saver.restore(self._session, restore_from)
