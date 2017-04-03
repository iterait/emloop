"""
cxflow main loop for training nets.

The MainLoop requires AbstractNet, AbstractDataset and a list of AbstractHooks.
Having all that, it manages iterating through streams, training and hooks execution.
"""
import sys
import logging
import typing
from collections import defaultdict

from .datasets.abstract_dataset import AbstractDataset
from .nets.abstract_net import AbstractNet
from .hooks.abstract_hook import AbstractHook, TrainingTerminated
from .utils.profile import Timer


class MainLoop:
    """Train the network, manage hooks etc."""

    def __init__(self, net: AbstractNet, dataset: AbstractDataset, hooks: typing.Iterable[AbstractHook]=(),
                 on_unused_sources: str='warn', fixed_batch_size: int=None):
        """
        :param net: trained network
        :param dataset: loaded dataset
        :param hooks: list of hooks
        :param on_unused_sources: action to take when stream provides unused sources {'ignore', 'warn', 'error'}
        :param fixed_batch_size: if specified, main_loop removes all batches that do not have the specified size
        """
        assert on_unused_sources in {'ignore', 'warn', 'error'}

        self._net = net
        self._dataset = dataset
        self._hooks = hooks
        self._on_unused_sources = on_unused_sources
        self._fixed_batch_size = fixed_batch_size
        self._extra_sources_warned = False
        self._epoch_profile = {}

    def _check_sources(self, batch: typing.Dict[str, object]) -> None:
        """
        Check for unused and missing sources.
        :param batch: batch to be checked
        """
        unused_sources = [source for source in batch.keys() if source not in self._net.input_names]
        missing_sources = [source for source in self._net.input_names if source not in batch.keys()]
        # check stream sources
        if unused_sources:
            if self._on_unused_sources == 'warn' and not self._extra_sources_warned:
                logging.warning('Some sources provided by the stream do not match net placeholders. Set '
                                '`main_loop.on_unused_sources` to `ignore` in order to suppress this warning. '
                                'Extra sources: %s', unused_sources)
                self._extra_sources_warned = True
            elif self._on_unused_sources == 'error':
                raise ValueError('Some sources provided by the stream do not match net placeholders. Set'
                                 '`main_loop.on_unused_sources` to `warn` in order to suppress this error.\n'
                                 'Extra sources: {}'.format(unused_sources))

        if missing_sources:
            raise ValueError('Stream does not provide all required sources. Missing sources: {}'
                             .format(missing_sources))

    def _run_epoch(self, stream: AbstractDataset.Stream, train: bool, stream_type: str) -> AbstractDataset.Batch:
        """
        Iterate through the stream
        :param stream: Iterable stream
        :param train: if set to true, the network will be trained
        :param stream_type: {train, valid, test}
        :return: epoch summary results
        """
        processed_batch_count = 0
        summed_results = defaultdict(float)

        while True:
            try:
                with Timer('read_batch_{}'.format(stream_type), self._epoch_profile):
                    batch = next(stream)
            except StopIteration:
                break

            if self._fixed_batch_size:
                if len(batch[list(batch.keys())[0]]) != self._fixed_batch_size:
                    logging.debug('Incomplete batch skipped')
                    continue

            self._check_sources(batch)

            with Timer('eval_batch_{}'.format(stream_type), self._epoch_profile):
                batch_result = self._net.run(batch=batch, train=train)

            with Timer('after_batch_hooks_{}'.format(stream_type), self._epoch_profile):
                for hook in self._hooks:
                    hook.after_batch(stream_type=stream_type, results=batch_result)

            for name, value in batch_result.items():
                try:
                    summed_results[name] += value
                except Exception as ex:
                    raise ValueError('Cannot sum results `{}`'.format(name)) from ex

            processed_batch_count += 1

        if processed_batch_count == 0:
            raise ValueError('No data in stream `{}`'.format(stream_type))

        for name in summed_results.keys():
            summed_results[name] /= processed_batch_count

        return summed_results

    def train_by_stream(self, stream: AbstractDataset.Stream) -> AbstractDataset.Batch:
        """Given a stream and batch size, train the network on this stream."""

        return self._run_epoch(stream=stream, train=True, stream_type='train')

    def evaluate_stream(self, stream: AbstractDataset.Stream, stream_type: str) -> AbstractDataset.Batch:
        """Given a stream and batch size, evaluate the network on this stream."""

        return self._run_epoch(stream=stream, train=False, stream_type=stream_type)

    def run(self, run_test_stream: bool) -> None:
        """
        Start the main loop
        :param run_test_stream: should the test stream be evaluated?
        """

        try:
            epoch_id = 0

            for hook in self._hooks:
                hook.before_training()

            valid_results = self.evaluate_stream(stream=self._dataset.create_valid_stream(), stream_type='valid')
            test_results = self.evaluate_stream(stream=self._dataset.create_test_stream(), stream_type='test') \
                if run_test_stream else None

            for hook in self._hooks:
                hook.before_first_epoch(valid_results=valid_results, test_results=test_results)

            while True:
                self._epoch_profile = {}
                epoch_id += 1

                train_results = self.train_by_stream(stream=self._dataset.create_train_stream())
                valid_results = self.evaluate_stream(stream=self._dataset.create_valid_stream(), stream_type='valid')
                test_results = self.evaluate_stream(stream=self._dataset.create_test_stream(), stream_type='test') \
                    if run_test_stream else None

                with Timer('after_epoch_hooks', self._epoch_profile):
                    for hook in self._hooks:
                        hook.after_epoch(epoch_id=epoch_id, train_results=train_results, valid_results=valid_results,
                                         test_results=test_results)

                for hook in self._hooks:
                    hook.after_epoch_profile(epoch_id=epoch_id, profile=self._epoch_profile)

        except TrainingTerminated as ex:
            logging.info('Training terminated by a hook: %s', ex)
        except KeyboardInterrupt:
            logging.warning('Training terminated by a keyboard interrupt')
            sys.exit(2)

        for hook in self._hooks:
            hook.after_training()
