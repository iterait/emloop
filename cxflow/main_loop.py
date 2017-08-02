"""
cxflow main loop for training nets.

The MainLoop requires AbstractNet, AbstractDataset and a list of AbstractHooks.
Having all that, it manages iterating through streams, training and hooks execution.
"""
import sys
import logging
import typing
from collections import OrderedDict

from .datasets import AbstractDataset
from .nets.abstract_net import AbstractNet
from .hooks.abstract_hook import AbstractHook, TrainingTerminated
from .utils.profile import Timer


class MainLoop:   # pylint: disable=too-many-instance-attributes
    """Train the network, manage hooks etc."""

    UNUSED_SOURCE_ACTIONS = {'ignore', 'warn', 'error'}

    def __init__(self,   # pylint: disable=too-many-arguments
                 net: AbstractNet, dataset: AbstractDataset,
                 hooks: typing.Iterable[AbstractHook]=(),
                 extra_streams: typing.List[str]=(),  # pylint: disable=invalid-sequence-index
                 on_unused_sources: str = 'warn',
                 fixed_batch_size: int = None,
                 skip_zeroth_epoch: bool = False):
        """
        :param net: trained network
        :param dataset: loaded dataset
        :param hooks: a sequence of hooks
        :param extra_streams: a sequence of additional stream names to be evaluated
        :param on_unused_sources: action to take when stream provides unused sources {'ignore', 'warn', 'error'}
        :param fixed_batch_size: if specified, main_loop removes all batches that do not have the specified size
        """
        assert on_unused_sources in MainLoop.UNUSED_SOURCE_ACTIONS

        self._net = net
        self._dataset = dataset
        self._hooks = hooks
        self._on_unused_sources = on_unused_sources
        self._fixed_batch_size = fixed_batch_size
        self._extra_sources_warned = False
        self._epoch_profile = {}
        self._extra_streams = extra_streams
        self._all_streams = list(self._extra_streams) + ['train']
        self._skip_zeroth_epoch = skip_zeroth_epoch

    def _create_epoch_data(self):
        """Create empty epoch data double dict."""
        return OrderedDict([(stream_name, OrderedDict()) for stream_name in self._all_streams])

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

    def _run_epoch(self, stream: AbstractDataset.Stream, train: bool, stream_name: str) -> None:
        """
        Iterate through the stream
        :param stream: Iterable stream
        :param train: if set to true, the network will be trained
        :param stream_name: {train} or any specified
        """
        while True:
            event_name = 'read_batch_{}'.format(stream_name)
            try:
                with Timer(event_name, self._epoch_profile):
                    batch_input = next(stream)
            except StopIteration:
                # remove the last recorded event which is created by the StopIteration exception
                self._epoch_profile[event_name].pop()
                break

            if self._fixed_batch_size:
                if len(batch_input[list(batch_input.keys())[0]]) != self._fixed_batch_size:
                    logging.debug('Incomplete batch skipped')
                    continue

            self._check_sources(batch_input)

            with Timer('eval_batch_{}'.format(stream_name), self._epoch_profile):
                batch_output = self._net.run(batch=batch_input, train=train)
            assert set(batch_input.keys()).isdisjoint(set(batch_output)), 'Batch inputs and outputs must not overlap.'

            with Timer('after_batch_hooks_{}'.format(stream_name), self._epoch_profile):
                for hook in self._hooks:
                    hook.after_batch(stream_name=stream_name, batch_data={**batch_input, **batch_output})

    def train_by_stream(self, stream: AbstractDataset.Stream) -> None:
        """Train the network with the given stream."""

        self._run_epoch(stream=stream, train=True, stream_name='train')

    def evaluate_stream(self, stream: AbstractDataset.Stream, stream_name: str) -> None:
        """Evaluate the network with the given stream."""

        self._run_epoch(stream=stream, train=False, stream_name=stream_name)

    def get_stream(self, stream_name: str) -> AbstractDataset.Stream:
        """
        Get a stream iterator with the given name.

        :param stream_name: name of the stream

        Raises:
            AttributeError: if the dataset does not provide the function creating the stream
        """
        try:
            stream_fn_name = 'create_{}_stream'.format(stream_name)
            return getattr(self._dataset, stream_fn_name)()
        except AttributeError as ex:
            raise AttributeError('The dataset does not have a function for creating a stream named `{}`. '
                                 'The function has to be named `{}`.'.format(stream_name, stream_fn_name)) from ex

    def run(self) -> None:
        """Run the main loop."""

        try:
            epoch_id = 0

            # Initialization: before_training
            for hook in self._hooks:
                hook.before_training()

            # Zeroth epoch: after_epoch
            if not self._skip_zeroth_epoch:
                for stream_name in self._all_streams:
                    self.evaluate_stream(stream=self.get_stream(stream_name), stream_name=stream_name)

                epoch_data = self._create_epoch_data()
                for hook in self._hooks:
                    hook.after_epoch(epoch_id=epoch_id, epoch_data=epoch_data)

            # Training loop: after_epoch, after_epoch_profile
            while True:
                epoch_id += 1
                self._epoch_profile = {}
                epoch_data = self._create_epoch_data()

                self.train_by_stream(stream=self._dataset.create_train_stream())
                for stream_name in self._extra_streams:
                    self.evaluate_stream(stream=self.get_stream(stream_name), stream_name=stream_name)

                with Timer('after_epoch_hooks', self._epoch_profile):
                    for hook in self._hooks:
                        hook.after_epoch(epoch_id=epoch_id, epoch_data=epoch_data)

                for hook in self._hooks:
                    hook.after_epoch_profile(epoch_id=epoch_id, profile=self._epoch_profile,
                                             extra_streams=self._extra_streams)

        except TrainingTerminated as ex:
            logging.info('Training terminated by a hook: %s', ex)
        except KeyboardInterrupt:
            logging.warning('Training terminated by a keyboard interrupt')
            sys.exit(2)

        # After training: after_training
        for hook in self._hooks:
            hook.after_training()
