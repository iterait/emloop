from .datasets.abstract_dataset import AbstractDataset

from .hooks.abstract_hook import AbstractHook, TrainingTerminated
from .nets.abstract_net import AbstractNet

import numpy as np

from collections import defaultdict
import logging
import typing

FuelIterator = typing.NewType('FuelIterator', typing.Iterable[typing.Mapping[str, typing.Any]])


class NetworkManager:
    """
    Train and evaluate network
    """

    def __init__(self, net: AbstractNet, dataset: AbstractDataset,
                 dont_ignore_extra_sources=False, dont_ignore_incomplete_batches=False,
                 hooks: typing.Iterable[AbstractHook]=[]):
        """
        Construct the manager
        """
        self.net = net
        self.dataset = dataset
        self.dont_ignore_extra_sources = dont_ignore_extra_sources
        self.dont_ignore_incomplete_batches = dont_ignore_incomplete_batches
        self.hooks = hooks

    def _run_batch(self, train: bool, **kwargs) -> typing.Mapping[str, np.ndarray]:
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
        return self._run_batch(train=True, **kwargs)

    def evaluate_batch(self, **kwargs) -> typing.Mapping[str, np.ndarray]:
        return self._run_batch(train=False, **kwargs)

    def _run_epoch(self, epoch_iterator: FuelIterator, train: bool, batch_size: int, stream_type: str,
                   batch_limit: int = None):
        n_batches = 0
        summed_results = defaultdict(float)

        for bid, d in enumerate(epoch_iterator):
            if not self.dont_ignore_incomplete_batches:
                if len(d[list(d.keys())[0]]) != batch_size:
                    continue

            n_batches += 1
            batch_result = self._run_batch(train=train, **d)

            for hook in self.hooks:
                hook.after_batch(net=self.net, stream_type=stream_type, results=batch_result)

            for name, value in batch_result.items():
                summed_results[name] += value

            if batch_limit and bid >= batch_limit:
                break

        for name in summed_results.keys():
            summed_results[name] /= n_batches

        return summed_results

    def train_by_stream(self, stream, batch_size: int, batch_limit: int = None):
        return self._run_epoch(epoch_iterator=stream, train=True,
                               batch_size=batch_size, batch_limit=batch_limit, stream_type='train')

    def evaluate_stream(self, stream, batch_size: int, stream_type: str, batch_limit: int = None):
        return self._run_epoch(epoch_iterator=stream, train=False,
                               batch_size=batch_size, batch_limit=batch_limit, stream_type=stream_type)

    def run_main_loop(self, batch_size: int, eval_batch_size_multiplier: float=1, **kwargs) -> None:
        train_batch_size = batch_size
        eval_batch_size = int(batch_size * eval_batch_size_multiplier)

        epoch_id = 0

        for hook in self.hooks:
            hook.before_training(net=self.net, **kwargs)

        valid_results = self.evaluate_stream(stream=self.dataset.create_valid_stream(), batch_size=eval_batch_size,
                                             stream_type='valid', **kwargs)
        test_results = self.evaluate_stream(stream=self.dataset.create_test_stream(), batch_size=eval_batch_size,
                                            stream_type='test', **kwargs)

        for hook in self.hooks:
            hook.before_first_epoch(net=self.net, valid_results=valid_results, test_results=test_results)

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
                    hook.after_epoch(net=self.net, epoch_id=epoch_id, train_results=train_results,
                                     valid_results=valid_results, test_results=test_results)
            except TrainingTerminated as e:
                logging.info('Training terminated by a hook: %s', e)
                break

        for hook in self.hooks:
            hook.after_training(net=self.net, **kwargs)
