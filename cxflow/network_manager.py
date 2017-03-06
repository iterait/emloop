from .datasets.abstract_dataset import AbstractDataset

from .hooks.abstract_hook import AbstractHook, TrainingTerminated
from .nets.abstract_net import AbstractNet

import numpy as np

from collections import defaultdict
import logging
import typing


class NetworkManager:
    """Train the network, manage hooks etc."""

    def __init__(self, net: AbstractNet, dataset: AbstractDataset,
                 dont_ignore_extra_sources=False, dont_ignore_incomplete_batches=False,
                 hooks: typing.Iterable[AbstractHook]=[]):
        """
        :param net: trained network
        :param dataset: loaded dataset
        :param dont_ignore_extra_sources: if set to true, the manager will raise an error if the dataset stream provides
                                          more data sources than expected by the network. If set to false, such
                                          situation will be ignored, which is the default behavior.
        :param dont_ignore_incomplete_batches: if set to false (default), the manager will skip incomplete batches (usually only
                                               the last one). This is useful when specifieng model batch size directly
                                               in the placeholders.
        :param hooks: list of hooks
        """

        self.net = net
        self.dataset = dataset
        self.dont_ignore_extra_sources = dont_ignore_extra_sources
        self.dont_ignore_incomplete_batches = dont_ignore_incomplete_batches
        self.hooks = hooks

    def _run_batch(self, train: bool, **kwargs) -> typing.Mapping[str, np.ndarray]:
        """Process a single batch (either train or eval)."""

        # setup the feed dict
        feed_dict = {}
        for placeholder_name, placeholder_value in kwargs.items():
            try:
                feed_dict[getattr(self.net, placeholder_name)] = placeholder_value
            except AttributeError as e:
                if self.dont_ignore_extra_sources:
                    raise e

        # setup fetches
        fetches = [self.net.train_op] if train else []
        fetches += [getattr(self.net, to_eval) for to_eval in self.net.to_evaluate]

        # run the computational graph for one batch
        batch_res = self.net.session.run(fetches=fetches, feed_dict=feed_dict)

        # zip the string names with results
        if train:
            return dict(zip(self.net.to_evaluate, batch_res[1:]))
        else:
            return dict(zip(self.net.to_evaluate, batch_res))

    def train_batch(self, **kwargs) -> typing.Mapping[str, np.ndarray]:
        """Train a single batch."""

        return self._run_batch(train=True, **kwargs)

    def evaluate_batch(self, **kwargs) -> typing.Mapping[str, np.ndarray]:
        """Evaluate a single batch."""

        return self._run_batch(train=False, **kwargs)

    def _run_epoch(self, stream: AbstractDataset.Stream, train: bool, batch_size: int, stream_type: str,
                   batch_limit: int = None):
        """
        Iterate through the stream
        :param stream: Iterable stream
        :param train: if set to true, the network will be trained
        :param batch_size: batch size
        :param stream_type: {train, valid, test}
        :param batch_limit: in set to a number, only this number of batches will be used, ignoring the others.
        :return: epoch summary results
        """

        n_batches = 0
        summed_results = defaultdict(float)

        for bid, d in enumerate(stream):
            if not self.dont_ignore_incomplete_batches:
                if len(d[list(d.keys())[0]]) != batch_size:
                    continue

            n_batches += 1
            batch_result = self._run_batch(train=train, **d)

            for hook in self.hooks:
                hook.after_batch(stream_type=stream_type, results=batch_result)

            for name, value in batch_result.items():
                summed_results[name] += value

            if batch_limit and bid >= batch_limit:
                break

        for name in summed_results.keys():
            summed_results[name] /= n_batches

        return summed_results

    def train_by_stream(self, stream: AbstractDataset.Stream, batch_size: int, batch_limit: int = None):
        """Given a stream and batch size, train the network on this stream."""

        return self._run_epoch(stream=stream, train=True,
                               batch_size=batch_size, batch_limit=batch_limit, stream_type='train')

    def evaluate_stream(self, stream: AbstractDataset.Stream, batch_size: int, stream_type: str, batch_limit: int = None):
        """Given a stream and batch size, evaluate the network on this stream."""

        return self._run_epoch(stream=stream, train=False,
                               batch_size=batch_size, batch_limit=batch_limit, stream_type=stream_type)

    def run_main_loop(self, batch_size: int, eval_batch_size_multiplier: float=1, **kwargs) -> None:
        """
        Start the main loop
        :param batch_size: batch size
        :param eval_batch_size_multiplier: valid/test batch size multiplier
        """

        train_batch_size = batch_size
        eval_batch_size = int(batch_size * eval_batch_size_multiplier)

        epoch_id = 0

        for hook in self.hooks:
            hook.before_training(**kwargs)

        valid_results = self.evaluate_stream(stream=self.dataset.create_valid_stream(), batch_size=eval_batch_size,
                                             stream_type='valid', **kwargs)
        test_results = self.evaluate_stream(stream=self.dataset.create_test_stream(), batch_size=eval_batch_size,
                                            stream_type='test', **kwargs)

        for hook in self.hooks:
            hook.before_first_epoch(valid_results=valid_results, test_results=test_results)

        while True:
            epoch_id += 1

            train_results = self.train_by_stream(stream=self.dataset.create_train_stream(), batch_size=train_batch_size,
                                                 **kwargs)
            valid_results = self.evaluate_stream(stream=self.dataset.create_valid_stream(), batch_size=eval_batch_size,
                                                 stream_type='valid', **kwargs)
            test_results = self.evaluate_stream(stream=self.dataset.create_test_stream(), batch_size=eval_batch_size,
                                                stream_type='test', **kwargs)

            try:
                for hook in self.hooks:
                    hook.after_epoch(epoch_id=epoch_id, train_results=train_results, valid_results=valid_results,
                                     test_results=test_results)
            except TrainingTerminated as e:
                logging.info('Training terminated by a hook: %s', e)
                break

        for hook in self.hooks:
            hook.after_training(**kwargs)
